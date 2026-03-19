#!/usr/bin/env python3
"""
Work-queue orchestrator: decomposes a task into a dependency DAG, submits ready
stages to a broker queue, and lets worker daemons pull/execute tasks.

Usage:
  python orchestrators/work_queue_orchestrator.py "Build a URL shortener"
  python orchestrators/work_queue_orchestrator.py --prompts challenge_prompts.json --prompt 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from agent_protocol.agent import Agent
from agent_protocol.broker import MessageBroker
from agent_protocol.messages import Message, MessageType
from orchestrators.dag_orchestrator import (
    _broker_port_from_endpoint,
    _truthy,
    build_agent_task,
    build_fix_defs,
    build_qa_task,
    call_assign_fixes,
    call_orchestrator,
    normalize_decomposition_result,
    read_qa_report,
    run_wave,
)
from orchestrators.patterns import resolve_pattern


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


def _ready_node_ids(nodes: Dict[str, Dict[str, Any]], pending: Set[str], completed: Set[str]) -> List[str]:
    ready = []
    for node_id in sorted(pending):
        deps = set(nodes[node_id].get("depends_on", []))
        if deps.issubset(completed):
            ready.append(node_id)
    return ready


def _stream_process_output(proc: subprocess.Popen, prefix: str) -> None:
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        text = line.rstrip()
        if text:
            print(f"[{prefix}] {text}", flush=True)


def _start_worker_daemons(
    worker_count: int,
    broker_router: str,
    broker_sub: str,
    worker_root: str,
    default_max_iterations: int,
    default_max_runtime_seconds: int,
    default_agent_model: str,
    max_concurrent_local: int,
) -> List[Dict[str, Any]]:
    workers: List[Dict[str, Any]] = []
    base_env = os.environ.copy()
    base_env["PYTHONIOENCODING"] = "utf-8"

    for idx in range(worker_count):
        worker_id = f"wq-worker-{idx + 1}-{int(time.time() * 1000)}"
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
            "agent",
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
            args=(proc, f"WQ:{worker_id}"),
            daemon=True,
        )
        thread.start()
        workers.append({"worker_id": worker_id, "proc": proc, "thread": thread})
    return workers


def _stop_worker_daemons(workers: List[Dict[str, Any]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Work-queue multi-agent orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--branch", help="Git branch to work on")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all agents/broker")
    parser.add_argument("--pattern", help="Optional pattern label; must resolve to WORK_QUEUE for this entrypoint")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
        idx = (args.prompt or 1) - 1
        entry = prompts[idx]
        print(f"Loaded prompt #{idx + 1}: {entry['name']}")
        args.task = entry["prompt"]

    if not args.task:
        parser.error("Provide a task string or use --prompts/--prompt.")

    if args.pattern:
        resolved = resolve_pattern(args.pattern).pattern
        if resolved != "work_queue":
            parser.error(
                f"work_queue_orchestrator.py only supports WORK_QUEUE pattern, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved

    return args


def _print_workspace(shared_dir: str) -> None:
    print("\n  Shared workspace contents:")
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            print(f"    {rel} ({size} bytes)")


def main() -> None:
    args = parse_args()
    task = args.task
    branch = args.branch
    shared_dir_arg = args.shared_dir

    manifest_path = os.path.join(PROJECT_ROOT, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]
    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL", settings["model"])
    agent_model = os.environ.get("AGENT_MODEL", settings["model"])

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "40"))
    max_runtime = int(os.environ.get("MAX_RUNTIME_SECONDS", "300"))
    max_waves = int(os.environ.get("MAX_WAVES", "3"))
    run_indefinitely = _truthy(os.environ.get("RESI_RUN_INDEFINITELY"))
    qa_enabled = run_indefinitely or max_waves > 0
    qa_iterations = int(os.environ.get("QA_ITERATIONS", "30"))
    fix_iterations = int(os.environ.get("FIX_ITERATIONS", "15"))
    fix_runtime = os.environ.get("FIX_RUNTIME_SECONDS", "120")

    executor = (args.executor or os.environ.get("COLLAB_EXECUTOR", "host")).strip().lower()
    if executor != "host":
        raise RuntimeError("work_queue pattern currently supports only --executor host.")

    worker_count = int(os.environ.get("WORK_QUEUE_WORKERS", "3"))
    worker_max_concurrent = int(os.environ.get("WORK_QUEUE_MAX_CONCURRENT_LOCAL", "1"))
    max_stage_retries = int(os.environ.get("WORK_QUEUE_STAGE_RETRIES", "1"))
    global_timeout_seconds = int(os.environ.get("WORK_QUEUE_GLOBAL_TIMEOUT_SECONDS", "1800"))
    if run_indefinitely:
        global_timeout_seconds = 0

    print("=" * 70)
    print("  WORK_QUEUE ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Pattern: {os.environ.get('COLLAB_PATTERN', 'work_queue')}")
    print(f"Orchestrator model: {orchestrator_model}")
    print(f"Agent model: {agent_model}")
    print(f"Executor: {executor}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    print(f"Workers: {worker_count} | Local concurrency: {worker_max_concurrent}")
    print(
        f"Stage retries: {max_stage_retries} | Global timeout: "
        f"{'unlimited' if run_indefinitely else f'{global_timeout_seconds}s'}"
    )
    if run_indefinitely:
        print("Mode: unlimited iterations/runtime/QA-fix waves (until completion or manual stop)")
    print(f"QA waves: {'unlimited' if run_indefinitely else max_waves} | QA iters: {qa_iterations} | Fix iters: {fix_iterations}")
    if branch:
        print(f"Branch: {branch}")
    print()

    print("[ORCHESTRATOR] Decomposing task...\n")
    result = call_orchestrator(orchestrator_model, task)

    try:
        nodes_list = normalize_decomposition_result(result)
    except Exception as exc:
        print(f"[ORCHESTRATOR] Invalid decomposition output: {exc}")
        print(f"[ORCHESTRATOR] Raw output: {json.dumps(result, indent=2)[:4000]}")
        raise

    nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes_list}

    print(f"[ORCHESTRATOR] Created {len(nodes_list)} queued stage(s):")
    for node in nodes_list:
        deps = f" (depends on: {', '.join(node['depends_on'])})" if node["depends_on"] else ""
        print(f"  - {node['id']} [{node.get('task_type', 'build')}]: {node['role']}{deps}")
    print()

    if shared_dir_arg:
        shared_dir = shared_dir_arg
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"shared-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[SHARED] Workspace: {shared_dir}")

    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")
    broker_router = os.environ.get("BROKER_ROUTER", "tcp://localhost:5555")
    broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")

    broker: Optional[MessageBroker] = None
    workers: List[Dict[str, Any]] = []
    coordinator: Optional[Agent] = None

    run_id = f"wq-{int(time.time() * 1000)}"
    result_topic = f"work_queue.results.{run_id}"
    collector = ResultCollector()

    build_failures: List[str] = []
    stage_history: List[Dict[str, Any]] = []
    qa_passed = False
    qa_waves_run = 0

    pending: Set[str] = set(nodes.keys())
    completed: Set[str] = set()
    node_attempts: Dict[str, int] = {node_id: 0 for node_id in nodes}
    in_flight_by_client: Dict[str, str] = {}

    progress_next = time.time() + 5.0
    deadline = float("inf") if global_timeout_seconds <= 0 else time.time() + global_timeout_seconds

    try:
        if broker_mode == "host":
            router_port = _broker_port_from_endpoint(broker_router, 5555)
            sub_port = _broker_port_from_endpoint(broker_sub, 5556)
            broker = MessageBroker(
                router_port=router_port,
                pub_port=sub_port,
                enable_logging=False,
            )
            broker.start()
            time.sleep(0.5)
            print(f"[BROKER] Started on :{router_port}/:{sub_port}")
        else:
            print("[BROKER] Connecting to existing broker")

        coordinator = Agent(
            agent_id=f"work-queue-coordinator-{int(time.time())}",
            broker_router=broker_router,
            broker_sub=broker_sub,
            topics=[result_topic],
            message_handler=collector.handler,
            enable_logging=False,
        )
        coordinator.start()
        time.sleep(0.3)

        worker_root = os.path.join(PROJECT_ROOT, "runs", "worker-local")
        workers = _start_worker_daemons(
            worker_count=worker_count,
            broker_router=broker_router,
            broker_sub=broker_sub,
            worker_root=worker_root,
            default_max_iterations=max_iterations,
            default_max_runtime_seconds=max_runtime,
            default_agent_model=agent_model,
            max_concurrent_local=worker_max_concurrent,
        )
        time.sleep(0.5)
        print(f"[WORKERS] Started {len(workers)} worker daemon(s)")

        while pending:
            if time.time() > deadline:
                build_failures.append("global_timeout")
                print("[WORK_QUEUE] Global timeout reached.")
                break

            ready = [
                node_id
                for node_id in _ready_node_ids(nodes, pending, completed)
                if node_id not in in_flight_by_client.values()
            ]

            for node_id in ready:
                node = nodes[node_id]
                node_attempts[node_id] += 1
                attempt = node_attempts[node_id]
                wrapped_task = build_agent_task(node, list(nodes.values()), branch=branch)

                client_task_id = f"{node_id}__a{attempt}"
                payload = {
                    "description": f"Queue stage {node_id} ({node.get('task_type', 'build')})",
                    "benchmark_id": run_id,
                    "client_task_id": client_task_id,
                    "task_type": node.get("task_type", "build"),
                    "instructions": wrapped_task,
                    "result_topic": result_topic,
                    "executor": "agent",
                    "max_iterations": max_iterations,
                    "max_runtime_seconds": max_runtime,
                    "agent_model": agent_model,
                    "shared_workspace": shared_dir,
                    "node_id": node_id,
                    "attempt": attempt,
                }
                coordinator.submit_task(payload)
                in_flight_by_client[client_task_id] = node_id
                print(f"[WORK_QUEUE] queued {node_id} attempt {attempt}")

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
                node_id = in_flight_by_client.pop(client_task_id, None)
                if not node_id:
                    continue

                attempt = node_attempts.get(node_id, 1)
                status = str(payload.get("status", "")).strip().lower()
                returncode = int(payload.get("returncode", 1) if payload.get("returncode") is not None else 1)

                if status == "success" and returncode == 0:
                    pending.discard(node_id)
                    completed.add(node_id)
                    stage_history.append(
                        {
                            "stage_id": node_id,
                            "attempt": attempt,
                            "decision": "pass",
                            "reason": "queue_worker_returncode_0",
                        }
                    )
                    print(f"[WORK_QUEUE] completed {node_id} attempt {attempt}")
                    continue

                reason = (
                    f"status={status} returncode={returncode} "
                    f"stderr_tail={str(payload.get('stderr_tail', ''))[-400:]}"
                ).strip()

                if attempt <= max_stage_retries:
                    node = nodes[node_id]
                    node["task"] = (
                        f"{node['task']}\n\n"
                        f"WORK_QUEUE RETRY FEEDBACK (attempt {attempt}):\n{reason}\n"
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
                    print(f"[WORK_QUEUE] retrying {node_id} (attempt {attempt + 1})")
                    continue

                pending.discard(node_id)
                build_failures.append(f"{node_id}:exit_{returncode}")
                stage_history.append(
                    {
                        "stage_id": node_id,
                        "attempt": attempt,
                        "decision": "fail",
                        "reason": reason,
                    }
                )
                print(f"[WORK_QUEUE] failed {node_id} after {attempt} attempt(s)")

            ready_after = _ready_node_ids(nodes, pending, completed)
            if not in_flight_by_client and not ready_after and pending:
                unresolved = {node_id: nodes[node_id].get("depends_on", []) for node_id in sorted(pending)}
                build_failures.append(f"dependency_blocked:{json.dumps(unresolved)}")
                print(f"[WORK_QUEUE] No ready or inflight stages. Unresolved dependencies: {unresolved}")
                break

            now = time.time()
            if now >= progress_next:
                print(
                    f"[WORK_QUEUE] progress completed={len(completed)}/{len(nodes)} inflight={len(in_flight_by_client)}",
                    flush=True,
                )
                progress_next = now + 5.0

            time.sleep(0.1)

        _stop_worker_daemons(workers)
        workers = []

        build_passed = len(build_failures) == 0 and len(pending) == 0

        if qa_enabled:
            wave_num = 0
            while run_indefinitely or wave_num < max_waves:
                qa_waves_run += 1
                report_path = os.path.join(shared_dir, "qa_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)

                qa_task = build_qa_task(task, list(nodes.values()), shared_dir)
                run_wave(
                    [{"id": "qa", "task": qa_task}],
                    f"QA-{wave_num + 1}",
                    agent_model,
                    qa_iterations,
                    fix_runtime,
                    shared_dir,
                    {
                        "executor": "host",
                        "broker_router": broker_router,
                        "broker_sub": broker_sub,
                        "shared_workspace_path": shared_dir,
                    },
                )

                report = read_qa_report(shared_dir)
                print(f"\n[QA-{wave_num + 1}] Status: {report['status']}")
                print(f"[QA-{wave_num + 1}] Summary: {report['summary']}")
                if report["status"] == "pass":
                    qa_passed = True
                    break

                if not run_indefinitely and wave_num == max_waves - 1:
                    break

                assignments = call_assign_fixes(orchestrator_model, report, list(nodes.values()))

                fix_defs = build_fix_defs(assignments, list(nodes.values()), report)
                run_wave(
                    fix_defs,
                    f"FIX-{wave_num + 1}",
                    agent_model,
                    fix_iterations,
                    fix_runtime,
                    shared_dir,
                    {
                        "executor": "host",
                        "broker_router": broker_router,
                        "broker_sub": broker_sub,
                        "shared_workspace_path": shared_dir,
                    },
                )
                wave_num += 1
    finally:
        _stop_worker_daemons(workers)
        if coordinator:
            coordinator.stop()
        if broker:
            broker.stop()

    build_passed = len(build_failures) == 0 and len(pending) == 0
    final_passed = build_passed and (qa_passed if qa_enabled else True)

    print("\n" + "=" * 70)
    print("  WORK_QUEUE COMPLETE")
    print("=" * 70)
    print(f"  Build result: {'PASSED' if build_passed else 'FAILED'}")
    if qa_enabled:
        print(f"  QA result: {'PASSED' if qa_passed else 'FAILED'}")
    else:
        print("  QA result: SKIPPED (MAX_WAVES=0)")

    _print_workspace(shared_dir)

    workspace_files = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            workspace_files.append(
                {"path": os.path.relpath(path, shared_dir), "size_bytes": os.path.getsize(path)}
            )

    last_qa_report = None
    report_path = os.path.join(shared_dir, "qa_report.json")
    if qa_enabled and os.path.exists(report_path):
        with open(report_path) as f:
            last_qa_report = json.load(f)

    summary = {
        "status": "pass" if final_passed else "fail",
        "pattern": "work_queue",
        "workspace": shared_dir,
        "task": task,
        "nodes": list(nodes.values()),
        "build_passed": build_passed,
        "build_failures": build_failures,
        "completed_stage_ids": sorted(completed),
        "pending_stage_ids": sorted(pending),
        "stage_history": stage_history,
        "worker_count": worker_count,
        "worker_max_concurrent_local": worker_max_concurrent,
        "qa_waves_run": qa_waves_run,
        "qa_passed": qa_passed,
        "qa_skipped": not qa_enabled,
        "run_indefinitely": run_indefinitely,
        "qa_report": last_qa_report,
        "files": workspace_files,
    }
    summary_path = os.path.join(shared_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Run summary written to: {summary_path}")

    sys.exit(0 if final_passed else 1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda _sig, _frame: sys.exit(130))
    main()
