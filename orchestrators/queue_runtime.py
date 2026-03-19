"""Shared queue runtime helpers for dependency-aware broker execution."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

from agent_protocol.agent import Agent
from agent_protocol.messages import Message, MessageType


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class QueueNodeSpec:
    """One queue-executed node in a dependency graph."""

    node_id: str
    role: str
    task_type: str
    payload: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    kind: str = "map"


class ResultCollector:
    def __init__(self) -> None:
        self._inbox: Deque[Message] = deque()
        self._lock = threading.Lock()

    def handler(self, message: Message) -> None:
        with self._lock:
            self._inbox.append(message)

    def drain(self) -> List[Message]:
        out: List[Message] = []
        with self._lock:
            while self._inbox:
                out.append(self._inbox.popleft())
        return out


def _ready_node_ids(nodes: Dict[str, QueueNodeSpec], pending: Set[str], completed: Set[str]) -> List[str]:
    ready: List[str] = []
    for node_id in sorted(pending):
        deps = set(nodes[node_id].depends_on)
        if deps.issubset(completed):
            ready.append(node_id)
    return ready


def _stream_process_output(proc: subprocess.Popen[str], prefix: str) -> None:
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        text = line.rstrip()
        if text:
            print(f"[{prefix}] {text}", flush=True)


def start_worker_daemons(
    worker_count: int,
    broker_router: str,
    broker_sub: str,
    worker_root: str,
    default_max_iterations: int,
    default_max_runtime_seconds: int,
    default_agent_model: str,
    max_concurrent_local: int,
    default_executor: str = "agent",
) -> List[Dict[str, Any]]:
    workers: List[Dict[str, Any]] = []
    base_env = os.environ.copy()
    base_env["PYTHONIOENCODING"] = "utf-8"

    for idx in range(worker_count):
        worker_id = f"queue-worker-{idx + 1}-{int(time.time() * 1000)}"
        cmd = [
            sys.executable,
            "runtime/worker_daemon.py",
            "--worker-id",
            worker_id,
            "--broker-router",
            broker_router,
            "--broker-sub",
            broker_sub,
            "--work-root",
            worker_root,
            "--max-concurrent-local",
            str(max_concurrent_local),
            "--default-executor",
            default_executor,
            "--default-max-iterations",
            str(default_max_iterations),
            "--default-max-runtime-seconds",
            str(default_max_runtime_seconds),
            "--default-agent-model",
            default_agent_model,
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=base_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=_stream_process_output,
            args=(proc, f"QUEUE:{worker_id}"),
            daemon=True,
        )
        thread.start()
        workers.append({"worker_id": worker_id, "proc": proc, "thread": thread})
    return workers


def stop_worker_daemons(workers: List[Dict[str, Any]]) -> None:
    for worker in workers:
        proc = worker["proc"]
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + 5.0
    for worker in workers:
        proc = worker["proc"]
        remaining = max(0.0, deadline - time.time())
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_queue_plan(
    *,
    nodes: List[QueueNodeSpec],
    coordinator: Agent,
    collector: ResultCollector,
    run_id: str,
    result_topic: str,
    global_timeout_seconds: int,
    max_stage_retries: int = 0,
    progress_label: str = "QUEUE",
) -> Dict[str, Any]:
    node_map: Dict[str, QueueNodeSpec] = {node.node_id: node for node in nodes}
    pending: Set[str] = set(node_map.keys())
    completed: Set[str] = set()
    node_attempts: Dict[str, int] = {node_id: 0 for node_id in node_map}
    in_flight_by_client: Dict[str, str] = {}
    stage_history: List[Dict[str, Any]] = []
    failures: List[str] = []
    results_by_node: Dict[str, Dict[str, Any]] = {}

    progress_next = time.time() + 5.0
    deadline = float("inf") if global_timeout_seconds <= 0 else time.time() + global_timeout_seconds

    while pending:
        if time.time() > deadline:
            failures.append("global_timeout")
            print(f"[{progress_label}] Global timeout reached.", flush=True)
            break

        ready = [
            node_id
            for node_id in _ready_node_ids(node_map, pending, completed)
            if node_id not in in_flight_by_client.values()
        ]

        for node_id in ready:
            spec = node_map[node_id]
            node_attempts[node_id] += 1
            attempt = node_attempts[node_id]
            client_task_id = f"{node_id}__a{attempt}"
            payload = {
                **spec.payload,
                "description": spec.payload.get("description", spec.role),
                "benchmark_id": run_id,
                "client_task_id": client_task_id,
                "task_type": spec.task_type,
                "result_topic": result_topic,
                "node_id": node_id,
                "attempt": attempt,
            }
            coordinator.submit_task(payload)
            in_flight_by_client[client_task_id] = node_id
            print(f"[{progress_label}] queued {node_id} attempt {attempt}", flush=True)

        for message in collector.drain():
            if message.message_type != MessageType.DATA:
                continue
            payload = message.payload
            if not isinstance(payload, dict):
                continue
            if payload.get("result_type") != "task_result":
                continue
            if payload.get("benchmark_id") != run_id:
                continue

            client_task_id = payload.get("client_task_id")
            node_id = in_flight_by_client.pop(str(client_task_id), None)
            if not node_id:
                continue

            attempt = node_attempts.get(node_id, 1)
            status = str(payload.get("status", "")).strip().lower()
            returncode = int(payload.get("returncode", 1) if payload.get("returncode") is not None else 1)

            if status == "success" and returncode == 0:
                pending.discard(node_id)
                completed.add(node_id)
                results_by_node[node_id] = payload
                stage_history.append(
                    {
                        "stage_id": node_id,
                        "attempt": attempt,
                        "decision": "pass",
                        "reason": "queue_worker_returncode_0",
                    }
                )
                print(f"[{progress_label}] completed {node_id} attempt {attempt}", flush=True)
                continue

            reason = (
                f"status={status} returncode={returncode} "
                f"stderr_tail={str(payload.get('stderr_tail', ''))[-400:]}"
            ).strip()

            if attempt <= max_stage_retries:
                spec = node_map[node_id]
                instructions = spec.payload.get("instructions")
                if isinstance(instructions, str):
                    spec.payload["instructions"] = (
                        f"{instructions}\n\n"
                        f"{progress_label} RETRY FEEDBACK (attempt {attempt}):\n{reason}\n"
                        "Address this directly before calling done."
                    )
                stage_history.append(
                    {
                        "stage_id": node_id,
                        "attempt": attempt,
                        "decision": "retry",
                        "reason": reason,
                    }
                )
                print(f"[{progress_label}] retrying {node_id} (attempt {attempt + 1})", flush=True)
                continue

            pending.discard(node_id)
            failures.append(f"{node_id}:exit_{returncode}")
            results_by_node[node_id] = payload
            stage_history.append(
                {
                    "stage_id": node_id,
                    "attempt": attempt,
                    "decision": "fail",
                    "reason": reason,
                }
            )
            print(f"[{progress_label}] failed {node_id} after {attempt} attempt(s)", flush=True)

        ready_after = _ready_node_ids(node_map, pending, completed)
        if not in_flight_by_client and not ready_after and pending:
            unresolved = {node_id: node_map[node_id].depends_on for node_id in sorted(pending)}
            failures.append(f"dependency_blocked:{unresolved}")
            print(f"[{progress_label}] No ready or inflight stages. Unresolved dependencies: {unresolved}", flush=True)
            break

        now = time.time()
        if now >= progress_next:
            print(
                f"[{progress_label}] progress completed={len(completed)}/{len(node_map)} "
                f"inflight={len(in_flight_by_client)}",
                flush=True,
            )
            progress_next = now + 5.0

        time.sleep(0.1)

    return {
        "completed_ids": sorted(completed),
        "pending_ids": sorted(pending),
        "build_failures": failures,
        "stage_history": stage_history,
        "results_by_node": results_by_node,
        "build_passed": len(failures) == 0 and len(pending) == 0,
    }
