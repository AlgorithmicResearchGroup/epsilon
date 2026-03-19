#!/usr/bin/env python3
"""Scale benchmark harness for queue-based large-agent topologies."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_protocol.agent import Agent
from agent_protocol.broker import MessageBroker
from orchestrators.queue_runtime import QueueNodeSpec, ResultCollector, run_queue_plan
from orchestrators.scale_topologies import (
    ManifestSpec,
    build_local_reduce_items,
    build_local_reduce_payload,
    build_map_reduce_nodes,
    build_sharded_queue_nodes,
)


DEFAULT_SAMPLE_TITLES = [
    "Artificial_intelligence",
    "Compiler",
    "Python_(programming_language)",
    "C_(programming_language)",
    "Operating_system",
    "Distributed_computing",
    "TCP/IP",
    "Machine_learning",
    "Database",
    "Computer_security",
    "Linux",
    "Unix",
    "Neural_network",
    "Open-source_software",
    "Software_testing",
    "Formal_verification",
    "Natural_language_processing",
    "Graph_theory",
    "Data_structure",
    "Algorithm",
]

COMPILER_COMPONENTS = [
    "lexer-identifiers",
    "lexer-numeric-literals",
    "lexer-string-literals",
    "lexer-keywords",
    "parser-expression-precedence",
    "parser-control-flow",
    "parser-function-declarations",
    "ast-node-definitions",
    "semantic-type-checking",
    "semantic-scope-resolution",
    "ir-three-address-code",
    "ir-constant-folding",
    "codegen-x86-register-allocation",
    "codegen-stack-frame-layout",
    "codegen-function-calls",
    "runtime-memory-intrinsics",
    "diagnostics-error-reporting",
    "diagnostics-source-spans",
    "tests-parser-golden-cases",
    "tests-codegen-smoke-cases",
]


def _now_ms() -> int:
    return int(time.time() * 1000)


def parse_zmq_tcp_url(url: str) -> Tuple[str, int]:
    if not url.startswith("tcp://"):
        raise ValueError(f"Unsupported broker URL '{url}'")
    host_port = url[len("tcp://"):]
    if ":" not in host_port:
        raise ValueError(f"Expected host:port in '{url}'")
    host, port = host_port.rsplit(":", 1)
    return host, int(port)


def load_titles(dataset: Optional[str]) -> List[str]:
    if not dataset:
        return list(DEFAULT_SAMPLE_TITLES)

    path = Path(dataset)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        titles: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    titles.append(item)
                elif isinstance(item, dict) and "title" in item:
                    titles.append(str(item["title"]))
        if titles:
            return titles
        raise ValueError(f"No titles found in JSON dataset '{dataset}'")

    titles = [line.strip() for line in text.splitlines() if line.strip()]
    if not titles:
        raise ValueError(f"No titles found in dataset '{dataset}'")
    return titles


def build_wiki_nodes(
    *,
    task_count: int,
    titles: List[str],
    executor: str,
    max_iterations: int,
    max_runtime_seconds: int,
    agent_model: Optional[str],
) -> List[QueueNodeSpec]:
    nodes: List[QueueNodeSpec] = []
    for i in range(task_count):
        title = titles[i % len(titles)]
        client_task_id = f"wiki-{i + 1:05d}"
        output_ref = f"results/{client_task_id}.json"
        page_path = quote(title.replace(" ", "_"), safe="_()")
        page_url = f"https://en.wikipedia.org/wiki/{page_path}"
        instructions = (
            f"Benchmark task id: {client_task_id}.\n"
            f"Goal: fetch and summarize one Wikipedia page.\n"
            f"Page URL: {page_url}\n\n"
            "Required steps:\n"
            "1. Use fetch_url to fetch the page URL.\n"
            "2. Build a JSON object with keys: task_id, title, source_url, summary, key_points.\n"
            "3. key_points must be an array of 3-7 concise bullets from the fetched page.\n"
            f"4. Write that JSON to file path: {output_ref}\n"
            "5. Call done with a short summary that includes the output path.\n"
            "Do not use run_bash for this task unless a tool fails unexpectedly."
        )
        payload: Dict[str, Any] = {
            "description": f"Wikipedia summary for {title}",
            "task_type": "wiki",
            "title": title,
            "instructions": instructions,
            "output_ref": output_ref,
            "input_ref": page_url,
            "executor": executor,
            "max_iterations": max_iterations,
            "max_runtime_seconds": max_runtime_seconds,
        }
        if agent_model:
            payload["agent_model"] = agent_model
        nodes.append(
            QueueNodeSpec(
                node_id=client_task_id,
                role=f"Wikipedia summary for {title}",
                task_type="wiki",
                payload=payload,
                kind="map",
            )
        )
    return nodes


def build_compiler_nodes(
    *,
    task_count: int,
    max_iterations: int,
    max_runtime_seconds: int,
    agent_model: Optional[str],
) -> List[QueueNodeSpec]:
    nodes: List[QueueNodeSpec] = []
    for i in range(task_count):
        component = COMPILER_COMPONENTS[i % len(COMPILER_COMPONENTS)]
        client_task_id = f"compiler-{i + 1:05d}"
        output_ref = f"compiler_parts/{component}-{i + 1:05d}.md"
        instructions = (
            f"Benchmark task id: {client_task_id}.\n"
            f"Component focus: {component}.\n\n"
            "Produce a concrete compiler micro-deliverable in Markdown with these sections:\n"
            "1. Component objective and boundaries\n"
            "2. Proposed C interfaces (function signatures, structs, enums)\n"
            "3. Edge cases and failure modes\n"
            "4. Minimal test cases (input -> expected behavior)\n"
            "5. Integration dependencies on other compiler components\n\n"
            f"Write file: {output_ref}\n"
            "Call done with a summary containing component name and output path."
        )
        payload: Dict[str, Any] = {
            "description": f"Compiler microtask for {component}",
            "component": component,
            "instructions": instructions,
            "output_ref": output_ref,
            "executor": "agent",
            "max_iterations": max_iterations,
            "max_runtime_seconds": max_runtime_seconds,
        }
        if agent_model:
            payload["agent_model"] = agent_model
        nodes.append(
            QueueNodeSpec(
                node_id=client_task_id,
                role=f"Compiler microtask for {component}",
                task_type="compiler",
                payload=payload,
                kind="map",
            )
        )
    return nodes


def build_local_reduce_nodes(
    *,
    topology: str,
    task_count: int,
    output_root: str,
    shard_count: int,
    reduce_arity: int,
    shared_workspace: str,
) -> Tuple[List[QueueNodeSpec], Dict[str, Any]]:
    items = build_local_reduce_items(task_count)
    manifest = ManifestSpec(
        pattern=topology,
        task_type="local_reduce",
        output_root=output_root,
        items=items,
        map_task_template="unused",
        reduce_task_template="unused",
        shard_count=shard_count,
        reduce_arity=reduce_arity,
        map_executor="local_reduce",
        reduce_executor="local_reduce",
    )

    if topology == "work_queue":
        nodes: List[QueueNodeSpec] = []
        for item in items:
            output_ref = f"{output_root}/maps/{item.item_id}.json"
            nodes.append(
                QueueNodeSpec(
                    node_id=f"map-{item.item_id}",
                    role=f"Local reduce map {item.item_id}",
                    task_type="map",
                    payload=build_local_reduce_payload(
                        description=f"Local map for {item.item_id}",
                        output_ref=output_ref,
                        shared_workspace=shared_workspace,
                        input_text=item.input_text,
                        payload=item.payload,
                        reduce_label=item.item_id,
                        operation="map",
                    ),
                    kind="map",
                )
            )
        return nodes, {"pattern": topology, "map_count": len(nodes), "reduce_count": 0}

    if topology == "sharded_queue":
        nodes, _, metadata = build_sharded_queue_nodes(
            manifest,
            shard_count=shard_count,
            shared_workspace=shared_workspace,
        )
        return nodes, metadata

    nodes, _, metadata = build_map_reduce_nodes(
        manifest,
        shared_workspace=shared_workspace,
    )
    return nodes, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale benchmark harness")
    parser.add_argument("--benchmark", choices=["wiki", "compiler", "local_reduce"], default="wiki")
    parser.add_argument("--topology", choices=["work_queue", "sharded_queue", "map_reduce"], default="work_queue")
    parser.add_argument("--task-count", type=int, default=100)
    parser.add_argument("--dataset", help="Wikipedia title dataset (line-delimited or JSON)")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--global-timeout-seconds", type=int, default=1800)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--reduce-arity", type=int, default=8)

    parser.add_argument("--broker-router", default=os.environ.get("BROKER_ROUTER", "tcp://localhost:5555"))
    parser.add_argument("--broker-sub", default=os.environ.get("BROKER_SUB", "tcp://localhost:5556"))
    parser.add_argument("--start-broker", action="store_true")
    parser.add_argument("--broker-bind-address", default=os.environ.get("BROKER_BIND_ADDRESS", "*"))
    parser.add_argument(
        "--broker-heartbeat-timeout-seconds",
        type=float,
        default=float(os.environ.get("BROKER_HEARTBEAT_TIMEOUT_SECONDS", "30")),
    )
    parser.add_argument(
        "--broker-lease-timeout-seconds",
        type=float,
        default=float(os.environ.get("BROKER_LEASE_TIMEOUT_SECONDS", "60")),
    )
    parser.add_argument(
        "--broker-sweep-interval-seconds",
        type=float,
        default=float(os.environ.get("BROKER_SWEEP_INTERVAL_SECONDS", "1")),
    )
    parser.add_argument(
        "--broker-max-redeliveries",
        type=int,
        default=int(os.environ.get("BROKER_MAX_REDELIVERIES", "5")),
    )
    parser.add_argument(
        "--broker-max-fail-retries",
        type=int,
        default=int(os.environ.get("BROKER_MAX_FAIL_RETRIES", "0")),
    )
    parser.add_argument(
        "--broker-redelivery-backoff-base-seconds",
        type=float,
        default=float(os.environ.get("BROKER_REDELIVERY_BACKOFF_BASE_SECONDS", "0")),
    )
    parser.add_argument(
        "--broker-redelivery-backoff-max-seconds",
        type=float,
        default=float(os.environ.get("BROKER_REDELIVERY_BACKOFF_MAX_SECONDS", "30")),
    )

    parser.add_argument("--executor", choices=["agent", "direct_wiki", "local_reduce"], default="agent")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-runtime-seconds", type=int, default=120)
    parser.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL"))

    parser.add_argument("--coordinator-id", default=f"benchmark-coordinator-{int(time.time())}")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_ms = _now_ms()
    benchmark_id = f"{args.benchmark}-{start_ms}"
    result_topic = f"benchmark.results.{benchmark_id}"

    out_dir = Path(args.output_dir) / f"scale-{benchmark_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.topology != "work_queue" and args.benchmark != "local_reduce":
        raise SystemExit(
            "Topology comparisons beyond work_queue are currently supported only for --benchmark local_reduce."
        )

    if args.benchmark == "wiki":
        effective_executor = args.executor
        nodes = build_wiki_nodes(
            task_count=args.task_count,
            titles=load_titles(args.dataset),
            executor=effective_executor,
            max_iterations=args.max_iterations,
            max_runtime_seconds=args.max_runtime_seconds,
            agent_model=args.agent_model,
        )
        topology_metadata = {"pattern": args.topology, "map_count": len(nodes), "reduce_count": 0}
    elif args.benchmark == "compiler":
        effective_executor = "agent"
        nodes = build_compiler_nodes(
            task_count=args.task_count,
            max_iterations=args.max_iterations,
            max_runtime_seconds=args.max_runtime_seconds,
            agent_model=args.agent_model,
        )
        topology_metadata = {"pattern": args.topology, "map_count": len(nodes), "reduce_count": 0}
    else:
        effective_executor = "local_reduce"
        nodes, topology_metadata = build_local_reduce_nodes(
            topology=args.topology,
            task_count=args.task_count,
            output_root="results",
            shard_count=args.shard_count,
            reduce_arity=args.reduce_arity,
            shared_workspace=str(out_dir),
        )

    submitted_node_ids = {node.node_id for node in nodes}
    broker: Optional[MessageBroker] = None
    if args.start_broker:
        _, router_port = parse_zmq_tcp_url(args.broker_router)
        _, sub_port = parse_zmq_tcp_url(args.broker_sub)
        broker = MessageBroker(
            router_port=router_port,
            pub_port=sub_port,
            bind_address=args.broker_bind_address,
            enable_logging=False,
            heartbeat_timeout_seconds=args.broker_heartbeat_timeout_seconds,
            lease_timeout_seconds=args.broker_lease_timeout_seconds,
            sweep_interval_seconds=args.broker_sweep_interval_seconds,
            max_redeliveries=args.broker_max_redeliveries,
            max_fail_retries=args.broker_max_fail_retries,
            redelivery_backoff_base_seconds=args.broker_redelivery_backoff_base_seconds,
            redelivery_backoff_max_seconds=args.broker_redelivery_backoff_max_seconds,
        )
        broker.start()
        time.sleep(0.3)

    collector = ResultCollector()
    coordinator = Agent(
        agent_id=args.coordinator_id,
        broker_router=args.broker_router,
        broker_sub=args.broker_sub,
        topics=[result_topic],
        message_handler=collector.handler,
        enable_logging=False,
    )
    coordinator.start()
    time.sleep(0.3)

    print("=" * 70)
    print("SCALE BENCHMARK")
    print("=" * 70)
    print(f"Benchmark ID: {benchmark_id}")
    print(f"Benchmark type: {args.benchmark}")
    print(f"Topology: {args.topology}")
    print(f"Input items: {args.task_count}")
    print(f"Submitted queue tasks: {len(nodes)}")
    print(f"Result topic: {result_topic}")
    print(f"Broker router: {args.broker_router}")
    print(f"Broker sub: {args.broker_sub}")
    print(f"Output dir: {out_dir}")
    print()

    print("Workers should run this command:")
    print(
        "  "
        f"python runtime/worker_daemon.py --worker-id <id> "
        f"--broker-router {args.broker_router} --broker-sub {args.broker_sub}"
    )
    print()

    queue_samples: List[Dict[str, Any]] = []
    sampler_stop = threading.Event()
    sampler_thread: Optional[threading.Thread] = None
    if broker is not None:

        def _sample_queue() -> None:
            while not sampler_stop.wait(0.25):
                stats = broker.get_stats()
                queue_samples.append(
                    {
                        "timestamp_ms": _now_ms(),
                        "tasks_pending": stats.get("tasks_pending", 0),
                        "tasks_submitted": stats.get("tasks_submitted", 0),
                        "tasks_assigned": stats.get("tasks_assigned", 0),
                        "connected_agents": stats.get("connected_agents", 0),
                    }
                )

        sampler_thread = threading.Thread(target=_sample_queue, daemon=True)
        sampler_thread.start()

    submitted_at_ms = _now_ms()
    try:
        queue_result = run_queue_plan(
            nodes=nodes,
            coordinator=coordinator,
            collector=collector,
            run_id=benchmark_id,
            result_topic=result_topic,
            global_timeout_seconds=args.global_timeout_seconds,
            max_stage_retries=0,
            progress_label="BENCH",
        )
        finished_at_ms = _now_ms()
    finally:
        sampler_stop.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=1.0)
        coordinator.stop()
        if broker is not None:
            broker.stop()

    results_by_node = queue_result["results_by_node"]
    completed_results = list(results_by_node.values())
    success = [r for r in completed_results if r.get("status") == "success"]
    failures = [r for r in completed_results if r.get("status") != "success"]
    missing_ids = sorted(submitted_node_ids - set(results_by_node))
    timed_out = not queue_result["build_passed"] and "global_timeout" in queue_result["build_failures"]

    durations_ms = [int(r.get("duration_ms", 0)) for r in completed_results if int(r.get("duration_ms", 0)) > 0]
    p50_ms = int(statistics.median(durations_ms)) if durations_ms else 0
    p95_ms = int(statistics.quantiles(durations_ms, n=20)[-1]) if len(durations_ms) >= 20 else 0

    elapsed_seconds = max(1.0, (finished_at_ms - submitted_at_ms) / 1000.0)
    throughput_per_min = (len(completed_results) / elapsed_seconds) * 60.0
    input_throughput_per_min = (args.task_count / elapsed_seconds) * 60.0

    total_prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in completed_results)
    total_response_tokens = sum(int(r.get("response_tokens", 0)) for r in completed_results)
    total_tokens = sum(int(r.get("total_tokens", 0)) for r in completed_results)
    total_cost = sum(float(r.get("estimated_cost_usd", 0.0)) for r in completed_results)

    report = {
        "benchmark_id": benchmark_id,
        "benchmark": args.benchmark,
        "topology": args.topology,
        "input_item_count": args.task_count,
        "submitted_task_count": len(nodes),
        "submitted_at_ms": submitted_at_ms,
        "finished_at_ms": finished_at_ms,
        "elapsed_seconds": elapsed_seconds,
        "timed_out": timed_out,
        "executor": effective_executor,
        "completed": len(completed_results),
        "success_count": len(success),
        "failure_count": len(failures),
        "missing_count": len(missing_ids),
        "missing_client_task_ids": missing_ids,
        "throughput_tasks_per_min": throughput_per_min,
        "throughput_input_items_per_min": input_throughput_per_min,
        "latency_ms": {"p50": p50_ms, "p95": p95_ms},
        "tokens": {
            "prompt": total_prompt_tokens,
            "response": total_response_tokens,
            "total": total_tokens,
        },
        "estimated_cost_usd": total_cost,
        "topology_metadata": topology_metadata,
        "build_failures": queue_result["build_failures"],
        "broker": {
            "router": args.broker_router,
            "sub": args.broker_sub,
            "started_by_harness": bool(broker),
        },
        "result_topic": result_topic,
    }

    report_path = out_dir / "scale_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    results_path = out_dir / "task_results.ndjson"
    with results_path.open("w", encoding="utf-8") as f:
        for item in sorted(completed_results, key=lambda x: x.get("client_task_id", "")):
            f.write(json.dumps(item) + "\n")

    queue_path = out_dir / "queue_samples.ndjson"
    with queue_path.open("w", encoding="utf-8") as f:
        for item in queue_samples:
            f.write(json.dumps(item) + "\n")

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Completed: {len(completed_results)}/{len(submitted_node_ids)}")
    print(f"Success: {len(success)} | Failures: {len(failures)} | Missing: {len(missing_ids)}")
    print(f"Throughput: {throughput_per_min:.2f} queue tasks/min")
    print(f"Input throughput: {input_throughput_per_min:.2f} items/min")
    print(f"Latency p50/p95: {p50_ms}ms / {p95_ms}ms")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Report: {report_path}")
    print(f"Results: {results_path}")
    print()

    raise SystemExit(0 if queue_result["build_passed"] else 1)


if __name__ == "__main__":
    main()
