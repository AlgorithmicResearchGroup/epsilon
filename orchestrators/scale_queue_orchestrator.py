#!/usr/bin/env python3
"""Shared entrypoint logic for manifest-backed scale topologies."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from agent_protocol.agent import Agent
from agent_protocol.broker import MessageBroker
from orchestrators.dag_orchestrator import _broker_port_from_endpoint, _truthy
from orchestrators.queue_runtime import ResultCollector, run_queue_plan, start_worker_daemons, stop_worker_daemons
from orchestrators.scale_topologies import (
    ManifestValidationError,
    build_map_reduce_nodes,
    build_sharded_queue_nodes,
    load_task_manifest,
    manifest_to_dict,
)


def parse_args(expected_pattern: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{expected_pattern} manifest-backed queue orchestrator")
    parser.add_argument("task", nargs="?", help="Optional task label for the overall run")
    parser.add_argument("--task-manifest", required=True, help="JSON manifest describing the large-scale workload")
    parser.add_argument("--branch", help="Optional git branch context to prepend to agent tasks")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all workers")
    parser.add_argument("--pattern", help=f"Must resolve to {expected_pattern}")
    args = parser.parse_args()

    if args.pattern:
        from orchestrators.patterns import resolve_pattern

        resolved = resolve_pattern(args.pattern).pattern
        if resolved != expected_pattern:
            parser.error(
                f"{expected_pattern}_orchestrator.py only supports {expected_pattern}, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved

    return args


def _load_runtime_settings() -> Dict[str, Any]:
    manifest_path = os.path.join(PROJECT_ROOT, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]
    return {
        "orchestrator_model": os.environ.get("ORCHESTRATOR_MODEL", settings["model"]),
        "agent_model": os.environ.get("AGENT_MODEL", settings["model"]),
    }


def _build_nodes_for_pattern(
    pattern: str,
    task_manifest_path: str,
    *,
    shared_workspace: str,
    worker_count: int,
) -> Tuple[List[Any], str, Dict[str, Any], Dict[str, Any]]:
    manifest = load_task_manifest(task_manifest_path, pattern)
    if pattern == "sharded_queue":
        shard_count = manifest.shard_count or min(8, max(1, worker_count))
        nodes, final_output_ref, metadata = build_sharded_queue_nodes(
            manifest,
            shard_count=shard_count,
            shared_workspace=shared_workspace,
        )
    elif pattern == "map_reduce":
        nodes, final_output_ref, metadata = build_map_reduce_nodes(
            manifest,
            shared_workspace=shared_workspace,
        )
    else:
        raise ValueError(f"Unsupported scale pattern '{pattern}'")
    return nodes, final_output_ref, metadata, manifest_to_dict(manifest)


def _prepend_branch_context(nodes: List[Any], branch: str) -> None:
    if not branch:
        return
    prefix = f"Work on git branch: {branch}\n\n"
    for node in nodes:
        instructions = node.payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            node.payload["instructions"] = prefix + instructions


def main_for_pattern(pattern: str) -> None:
    args = parse_args(pattern)
    runtime = _load_runtime_settings()
    agent_model = runtime["agent_model"]

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "40"))
    max_runtime = int(os.environ.get("MAX_RUNTIME_SECONDS", "300"))
    worker_count = int(os.environ.get("WORK_QUEUE_WORKERS", "3"))
    worker_max_concurrent = int(os.environ.get("WORK_QUEUE_MAX_CONCURRENT_LOCAL", "1"))
    max_stage_retries = int(os.environ.get("WORK_QUEUE_STAGE_RETRIES", "1"))
    global_timeout_seconds = int(os.environ.get("WORK_QUEUE_GLOBAL_TIMEOUT_SECONDS", "1800"))
    if _truthy(os.environ.get("RESI_RUN_INDEFINITELY")):
        global_timeout_seconds = 0

    executor = (args.executor or os.environ.get("COLLAB_EXECUTOR", "host")).strip().lower()
    if executor != "host":
        raise RuntimeError(f"{pattern} currently supports only --executor host.")

    if args.shared_dir:
        shared_dir = args.shared_dir
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"{pattern}-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)

    try:
        nodes, final_output_ref, metadata, manifest_dict = _build_nodes_for_pattern(
            pattern,
            args.task_manifest,
            shared_workspace=shared_dir,
            worker_count=worker_count,
        )
    except ManifestValidationError as exc:
        raise SystemExit(str(exc)) from exc

    _prepend_branch_context(nodes, args.branch or "")

    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")
    broker_router = os.environ.get("BROKER_ROUTER", "tcp://localhost:5555")
    broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
    worker_root = os.path.join(PROJECT_ROOT, "runs", "worker-local")

    print("=" * 70)
    print(f"  {pattern.upper()} ORCHESTRATOR")
    print("=" * 70)
    print(f"Task label: {args.task or pattern}")
    print(f"Manifest: {args.task_manifest}")
    print(f"Pattern: {pattern}")
    print(f"Agent model: {agent_model}")
    print(f"Executor: {executor}")
    print(f"Workers: {worker_count} | Local concurrency: {worker_max_concurrent}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    print(f"Stage retries: {max_stage_retries} | Global timeout: {global_timeout_seconds or 'unlimited'}")
    print(f"Graph nodes: {len(nodes)}")
    for key, value in metadata.items():
        if key == "shards":
            print(f"Topology shards: {len(value)}")
            continue
        print(f"{key.replace('_', ' ').title()}: {value}")
    print(f"Shared workspace: {shared_dir}")
    print()

    broker: MessageBroker | None = None
    coordinator: Agent | None = None
    workers: List[Dict[str, Any]] = []
    collector = ResultCollector()

    run_id = f"{pattern}-{int(time.time() * 1000)}"
    result_topic = f"{pattern}.results.{run_id}"

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
            agent_id=f"{pattern}-coordinator-{int(time.time())}",
            broker_router=broker_router,
            broker_sub=broker_sub,
            topics=[result_topic],
            message_handler=collector.handler,
            enable_logging=False,
        )
        coordinator.start()
        time.sleep(0.3)

        workers = start_worker_daemons(
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

        queue_result = run_queue_plan(
            nodes=nodes,
            coordinator=coordinator,
            collector=collector,
            run_id=run_id,
            result_topic=result_topic,
            global_timeout_seconds=global_timeout_seconds,
            max_stage_retries=max_stage_retries,
            progress_label=pattern.upper(),
        )
    finally:
        stop_worker_daemons(workers)
        if coordinator is not None:
            coordinator.stop()
        if broker is not None:
            broker.stop()

    final_output_path = os.path.join(shared_dir, final_output_ref)
    summary = {
        "status": "pass" if queue_result["build_passed"] else "fail",
        "pattern": pattern,
        "task": args.task or pattern,
        "manifest_path": args.task_manifest,
        "manifest": manifest_dict,
        "workspace": shared_dir,
        "metadata": metadata,
        "final_output_ref": final_output_ref,
        "final_output_path": final_output_path,
        "final_output_exists": os.path.exists(final_output_path),
        "completed_stage_ids": queue_result["completed_ids"],
        "pending_stage_ids": queue_result["pending_ids"],
        "build_failures": queue_result["build_failures"],
        "stage_history": queue_result["stage_history"],
        "node_count": len(nodes),
        "map_count": len([node for node in nodes if node.kind == "map"]),
        "reduce_count": len([node for node in nodes if node.kind == "reduce"]),
        "worker_count": worker_count,
        "worker_max_concurrent_local": worker_max_concurrent,
    }
    summary_path = os.path.join(shared_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  {pattern.upper()} COMPLETE")
    print("=" * 70)
    print(f"Build result: {'PASSED' if queue_result['build_passed'] else 'FAILED'}")
    print(f"Final output: {final_output_path}")
    print(f"Run summary: {summary_path}")

    sys.exit(0 if queue_result["build_passed"] else 1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda _sig, _frame: sys.exit(130))
    raise SystemExit("Use sharded_queue_orchestrator.py or map_reduce_orchestrator.py directly.")
