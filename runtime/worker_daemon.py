#!/usr/bin/env python3
"""Queue-driven worker daemon for scale benchmarks.

The daemon connects to the protocol broker, repeatedly requests tasks from the
work queue, executes them, and publishes structured result messages.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_protocol.agent import Agent
from agent_protocol.messages import Message, MessageType


DEFAULT_IDLE_POLL_SECONDS = 2.0
DEFAULT_ASSIGNMENT_POLL_SECONDS = 0.5


def _now_ms() -> int:
    return int(time.time() * 1000)


def _resolve_task_root(work_root: Path, benchmark_id: str, task_id: str, shared_workspace: Any) -> Path:
    if shared_workspace:
        return Path(str(shared_workspace))
    return work_root / benchmark_id / task_id


def _resolve_ref_path(task_root: Path, ref: str) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    return task_root / candidate


def _read_structured_input(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {"value": data}
    return {"raw_text": text}


class WorkerDaemon:
    def __init__(
        self,
        worker_id: str,
        broker_router: str,
        broker_sub: str,
        work_root: Path,
        max_concurrent_local: int,
        default_executor: str,
        default_max_iterations: int,
        default_max_runtime_seconds: int,
        default_agent_model: Optional[str],
        idle_poll_seconds: float,
        assignment_poll_seconds: float,
    ) -> None:
        self.worker_id = worker_id
        self.broker_router = broker_router
        self.broker_sub = broker_sub
        self.work_root = work_root
        self.max_concurrent_local = max(1, max_concurrent_local)
        self.default_executor = default_executor
        self.default_max_iterations = default_max_iterations
        self.default_max_runtime_seconds = default_max_runtime_seconds
        self.default_agent_model = default_agent_model
        self.idle_poll_seconds = idle_poll_seconds
        self.assignment_poll_seconds = assignment_poll_seconds

        self._lock = threading.Lock()
        self._inbox: deque[Message] = deque()
        self._stop = threading.Event()
        self._stopped = False

        self._agent = Agent(
            agent_id=self.worker_id,
            broker_router=self.broker_router,
            broker_sub=self.broker_sub,
            topics=["benchmark.results"],
            message_handler=self._message_handler,
            enable_logging=False,
            heartbeat_interval_seconds=float(os.environ.get("PROTOCOL_HEARTBEAT_INTERVAL_SECONDS", "5")),
            heartbeat_enabled=True,
        )

        self._pool = ThreadPoolExecutor(max_workers=self.max_concurrent_local)
        self._inflight: Dict[str, Future] = {}
        self._last_no_tasks_at = 0.0
        self._assignment_requests = 0
        self._completed_count = 0

    def _message_handler(self, message: Message) -> None:
        with self._lock:
            self._inbox.append(message)

    def start(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._agent.start()
        time.sleep(0.2)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        self._pool.shutdown(wait=True)
        self._agent.stop()

    def run(self) -> None:
        print(
            f"[worker-daemon] id={self.worker_id} router={self.broker_router} "
            f"sub={self.broker_sub} max_local={self.max_concurrent_local}",
            flush=True,
        )

        while not self._stop.is_set():
            self._drain_inbox()
            self._collect_finished()
            self._request_more_tasks_if_capacity()
            time.sleep(0.05)

    def _request_more_tasks_if_capacity(self) -> None:
        if len(self._inflight) >= self.max_concurrent_local:
            return

        now = time.time()
        if now - self._last_no_tasks_at < self.idle_poll_seconds:
            return

        self._agent.request_task()
        self._assignment_requests += 1
        time.sleep(self.assignment_poll_seconds)

    def _drain_inbox(self) -> None:
        messages: List[Message] = []
        with self._lock:
            while self._inbox:
                messages.append(self._inbox.popleft())

        for msg in messages:
            if msg.message_type != MessageType.TASK_ASSIGN:
                continue

            payload = msg.payload
            if payload.get("status") == "no_tasks":
                self._last_no_tasks_at = time.time()
                continue

            task_id = payload.get("task_id")
            if not task_id or task_id in self._inflight:
                continue

            future = self._pool.submit(self._process_assignment, payload)
            self._inflight[task_id] = future

    def _collect_finished(self) -> None:
        done_ids: List[str] = []
        for task_id, fut in self._inflight.items():
            if not fut.done():
                continue
            done_ids.append(task_id)
            try:
                fut.result()
            except Exception as exc:
                print(f"[worker-daemon] task {task_id} worker exception: {exc}", flush=True)

        for task_id in done_ids:
            self._inflight.pop(task_id, None)

    def _process_assignment(self, assignment_payload: Dict[str, Any]) -> None:
        task_id = assignment_payload["task_id"]
        task_payload = assignment_payload.get("payload", {})
        benchmark_id = task_payload.get("benchmark_id", "unknown")
        task_type = task_payload.get("task_type", "unknown")
        result_topic = task_payload.get("result_topic", "benchmark.results")
        lease_id = assignment_payload.get("lease_id")
        lease_timeout_seconds = float(assignment_payload.get("lease_timeout_seconds", 60))

        started_ms = _now_ms()
        status = "success"
        error = ""
        worker_result: Dict[str, Any] = {}
        renew_stop = threading.Event()
        renew_thread: Optional[threading.Thread] = None

        try:
            if lease_id:
                renew_interval = max(2.0, min(15.0, lease_timeout_seconds * 0.4))

                def _renew_loop() -> None:
                    while not renew_stop.wait(renew_interval):
                        try:
                            self._agent.renew_task(task_id, lease_id)
                        except Exception:
                            # Renew failures are observable through broker redelivery;
                            # task execution should continue and report result.
                            pass

                renew_thread = threading.Thread(target=_renew_loop, daemon=True)
                renew_thread.start()

            executor = task_payload.get("executor", self.default_executor)
            if executor == "agent":
                worker_result = self._execute_agent_task(task_id, task_payload)
                returncode = worker_result.get("returncode")
                if returncode not in (None, 0):
                    stderr_tail = str(worker_result.get("stderr_tail", "")).strip()
                    detail = f" (stderr: {stderr_tail[-300:]})" if stderr_tail else ""
                    raise RuntimeError(f"agent subprocess exited with code {returncode}{detail}")
                expected_output = str(task_payload.get("output_ref", "") or "").strip()
                if expected_output and not bool(worker_result.get("output_exists")):
                    raise RuntimeError(
                        f"agent task did not produce expected output_ref '{expected_output}'"
                    )
            elif executor == "direct_wiki":
                worker_result = self._execute_direct_wiki_task(task_id, task_payload)
            elif executor == "local_reduce":
                worker_result = self._execute_local_reduce_task(task_id, task_payload)
            elif executor == "python_handler":
                worker_result = self._execute_python_handler_task(task_id, task_payload)
            else:
                raise ValueError(f"Unsupported executor '{executor}'")
        except Exception as exc:
            status = "failure"
            error = f"{type(exc).__name__}: {exc}"
            worker_result = {
                "stdout_tail": "",
                "stderr_tail": traceback.format_exc()[-4000:],
                "output_path": "",
                "output_exists": False,
                "output_preview": "",
                "returncode": None,
            }
        finally:
            renew_stop.set()
            if renew_thread:
                renew_thread.join(timeout=2)

        ended_ms = _now_ms()
        duration_ms = ended_ms - started_ms

        task_result = {
            "result_type": "task_result",
            "benchmark_id": benchmark_id,
            "task_id": task_id,
            "client_task_id": task_payload.get("client_task_id", task_id),
            "task_type": task_type,
            "worker_id": self.worker_id,
            "status": status,
            "error": error,
            "lease_id": lease_id or "",
            "started_at_ms": started_ms,
            "ended_at_ms": ended_ms,
            "duration_ms": duration_ms,
            "executor": task_payload.get("executor", self.default_executor),
            **worker_result,
        }

        if status == "success":
            self._agent.complete_task(
                task_id,
                result={"status": status, "duration_ms": duration_ms},
                lease_id=lease_id,
            )
        else:
            self._agent.fail_task(
                task_id,
                error={"error": error, "duration_ms": duration_ms},
                lease_id=lease_id,
            )
        self._agent.send_data(task_result, topic=result_topic)

        self._completed_count += 1
        print(
            f"[worker-daemon] completed task={task_id} status={status} "
            f"duration_ms={duration_ms} completed_total={self._completed_count}",
            flush=True,
        )

    def _execute_agent_task(self, task_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        project_root = PROJECT_ROOT
        benchmark_id = task_payload.get("benchmark_id", "unknown")
        shared_workspace = task_payload.get("shared_workspace")
        task_root = _resolve_task_root(self.work_root, benchmark_id, task_id, shared_workspace)
        task_root.mkdir(parents=True, exist_ok=True)

        instructions = task_payload["instructions"]
        max_iterations = int(task_payload.get("max_iterations", self.default_max_iterations))
        max_runtime_seconds = int(task_payload.get("max_runtime_seconds", self.default_max_runtime_seconds))
        model_name = task_payload.get("agent_model", self.default_agent_model)

        env = os.environ.copy()
        env["PROTOCOL_ENABLED"] = "false"
        env["MAX_ITERATIONS"] = str(max_iterations)
        env["MAX_RUNTIME_SECONDS"] = str(max_runtime_seconds)
        env["SHARED_WORKSPACE"] = str(task_root)
        if model_name:
            env["AGENT_MODEL"] = model_name

        agent_mode = str(env.get("COLLAB_AGENT_MODE", "native")).strip().lower()
        entry_script = "runtime/byoa_runner.py" if agent_mode == "adapter" else "runtime/agent_main.py"
        entry_binary = ""
        if agent_mode == "adapter":
            entry_binary = str(env.get("COLLAB_BYOA_RUNNER_BINARY", "")).strip()
        else:
            entry_binary = str(env.get("COLLAB_AGENT_MAIN_BINARY", "")).strip()

        if entry_binary and Path(entry_binary).exists():
            cmd = [entry_binary, instructions]
        else:
            cmd = [sys.executable, entry_script, instructions]

        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output_ref = task_payload.get("output_ref", "")
        output_path = task_root / output_ref if output_ref else None
        output_exists = bool(output_path and output_path.exists())
        output_preview = ""
        if output_exists and output_path is not None:
            output_preview = output_path.read_text(errors="replace")[:2000]

        prompt_tokens = 0
        response_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        for line in proc.stdout.splitlines():
            if "Tokens:" not in line:
                continue
            token_match = re.search(r"Tokens:\s*([\d,]+)\s*\(([\d,]+) in / ([\d,]+) out\)\s*\| Cost:\s*\$([0-9.]+)", line)
            if token_match:
                total_tokens = int(token_match.group(1).replace(",", ""))
                prompt_tokens = int(token_match.group(2).replace(",", ""))
                response_tokens = int(token_match.group(3).replace(",", ""))
                total_cost = float(token_match.group(4))

        return {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-2000:],
            "output_path": str(output_path) if output_path else "",
            "output_exists": output_exists,
            "output_preview": output_preview,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
        }

    def _execute_direct_wiki_task(self, task_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        benchmark_id = task_payload.get("benchmark_id", "unknown")
        title = task_payload["title"]
        page_title = quote(title.replace(" ", "_"), safe="_")
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"

        task_root = _resolve_task_root(
            self.work_root,
            benchmark_id,
            task_id,
            task_payload.get("shared_workspace"),
        )
        task_root.mkdir(parents=True, exist_ok=True)
        output_ref = task_payload.get("output_ref", f"results/{task_id}.json")
        output_path = task_root / output_ref
        output_path.parent.mkdir(parents=True, exist_ok=True)

        request = Request(
            api_url,
            headers={
                "User-Agent": "epsilon-scale-benchmark/1.0 (+https://example.local)",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        item = {
            "task_id": task_id,
            "title": payload.get("title", title),
            "url": payload.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "summary": payload.get("extract", ""),
            "description": payload.get("description", ""),
            "last_updated": payload.get("timestamp", ""),
        }
        output_path.write_text(json.dumps(item, indent=2), encoding="utf-8")

        return {
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_path": str(output_path),
            "output_exists": True,
            "output_preview": json.dumps(item)[:2000],
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def _execute_local_reduce_task(self, task_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        benchmark_id = task_payload.get("benchmark_id", "unknown")
        task_root = _resolve_task_root(
            self.work_root,
            benchmark_id,
            task_id,
            task_payload.get("shared_workspace"),
        )
        task_root.mkdir(parents=True, exist_ok=True)

        output_ref = task_payload.get("output_ref", f"results/{task_id}.json")
        output_path = _resolve_ref_path(task_root, output_ref)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        operation = str(task_payload.get("operation", "reduce") or "reduce").strip().lower()
        if operation == "map":
            input_text = str(task_payload.get("input_text", "") or "")
            input_ref = str(task_payload.get("input_ref", "") or "")
            if not input_text and input_ref:
                source_path = _resolve_ref_path(task_root, input_ref)
                input_text = source_path.read_text(encoding="utf-8", errors="replace")
            normalized = " ".join(input_text.split())
            words = normalized.split()
            item = {
                "task_id": task_payload.get("client_task_id", task_id),
                "operation": "map",
                "input_ref": input_ref,
                "payload": task_payload.get("payload", {}),
                "word_count": len(words),
                "summary": " ".join(words[:24]),
            }
        elif operation == "reduce":
            input_refs = [str(ref) for ref in task_payload.get("input_refs", [])]
            children = [_read_structured_input(_resolve_ref_path(task_root, ref)) for ref in input_refs]
            summary_parts: List[str] = []
            total_word_count = 0
            source_task_ids: List[str] = []
            for child in children:
                summary = str(child.get("summary") or child.get("aggregate_summary") or child.get("raw_text") or "").strip()
                if summary:
                    summary_parts.append(summary)
                total_word_count += int(child.get("word_count", 0) or child.get("total_word_count", 0) or 0)
                task_label = str(child.get("task_id") or child.get("reduce_label") or "").strip()
                if task_label:
                    source_task_ids.append(task_label)
            item = {
                "task_id": task_payload.get("client_task_id", task_id),
                "operation": "reduce",
                "reduce_label": str(task_payload.get("reduce_label", "") or ""),
                "child_count": len(children),
                "input_refs": input_refs,
                "source_task_ids": source_task_ids,
                "total_word_count": total_word_count,
                "aggregate_summary": " | ".join(summary_parts[:16]),
            }
        else:
            raise ValueError(f"Unsupported local_reduce operation '{operation}'")

        output_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
        return {
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_path": str(output_path),
            "output_exists": True,
            "output_preview": json.dumps(item)[:2000],
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    def _execute_python_handler_task(self, task_id: str, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        benchmark_id = task_payload.get("benchmark_id", "unknown")
        task_root = _resolve_task_root(
            self.work_root,
            benchmark_id,
            task_id,
            task_payload.get("shared_workspace"),
        )
        task_root.mkdir(parents=True, exist_ok=True)

        handler_path = str(task_payload.get("handler", "") or "").strip()
        if not handler_path or ":" not in handler_path:
            raise ValueError("python_handler tasks require 'handler' in module:function format")
        module_name, function_name = handler_path.split(":", 1)
        module = importlib.import_module(module_name)
        handler = getattr(module, function_name, None)
        if handler is None or not callable(handler):
            raise ValueError(f"Handler '{handler_path}' is not callable")

        result = handler(task_id=task_id, task_payload=task_payload, task_root=task_root)
        if not isinstance(result, dict):
            raise ValueError(f"Handler '{handler_path}' returned non-dict result")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue-driven worker daemon")
    parser.add_argument("--worker-id", default=f"worker-{int(time.time())}")
    parser.add_argument("--broker-router", default=os.environ.get("BROKER_ROUTER", "tcp://localhost:5555"))
    parser.add_argument("--broker-sub", default=os.environ.get("BROKER_SUB", "tcp://localhost:5556"))
    parser.add_argument("--work-root", default=os.environ.get("WORKER_WORK_ROOT", "runs/worker-local"))
    parser.add_argument("--max-concurrent-local", type=int, default=1)
    parser.add_argument(
        "--default-executor",
        choices=["agent", "direct_wiki", "local_reduce", "python_handler"],
        default="agent",
    )
    parser.add_argument("--default-max-iterations", type=int, default=10)
    parser.add_argument("--default-max-runtime-seconds", type=int, default=120)
    parser.add_argument("--default-agent-model", default=os.environ.get("AGENT_MODEL"))
    parser.add_argument("--idle-poll-seconds", type=float, default=DEFAULT_IDLE_POLL_SECONDS)
    parser.add_argument("--assignment-poll-seconds", type=float, default=DEFAULT_ASSIGNMENT_POLL_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daemon = WorkerDaemon(
        worker_id=args.worker_id,
        broker_router=args.broker_router,
        broker_sub=args.broker_sub,
        work_root=Path(args.work_root),
        max_concurrent_local=args.max_concurrent_local,
        default_executor=args.default_executor,
        default_max_iterations=args.default_max_iterations,
        default_max_runtime_seconds=args.default_max_runtime_seconds,
        default_agent_model=args.default_agent_model,
        idle_poll_seconds=args.idle_poll_seconds,
        assignment_poll_seconds=args.assignment_poll_seconds,
    )

    def _handle_signal(sig: int, _frame: Any) -> None:
        print(f"[worker-daemon] signal={sig} stopping", flush=True)
        daemon.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    daemon.start()
    try:
        daemon.run()
    finally:
        daemon.stop()


if __name__ == "__main__":
    main()
