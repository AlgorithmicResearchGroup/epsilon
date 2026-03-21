#!/usr/bin/env python3
"""Iterative population-search orchestrator for recursive improvement workloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from agent_protocol.agent import Agent
from agent_protocol.broker import MessageBroker
from orchestrators.dag_orchestrator import _broker_port_from_endpoint
from orchestrators.patterns import resolve_pattern
from orchestrators.population_search import (
    CandidatePlan,
    PopulationSearchManifest,
    build_candidate_prompt,
    build_generation_selection,
    build_generation_zero_plans,
    build_review_prompt,
    load_population_manifest,
    manifest_to_dict,
    score_improved,
    sort_candidate_results,
)
from orchestrators.queue_runtime import QueueNodeSpec, ResultCollector, run_queue_plan, start_worker_daemons, stop_worker_daemons
from orchestrators.scale_topologies import ManifestValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="population_search recursive improvement orchestrator")
    parser.add_argument("task", nargs="?", help="Optional task label for the overall run")
    parser.add_argument("--task-manifest", required=True, help="JSON manifest describing the population-search workload")
    parser.add_argument("--branch", help="Optional git branch context to prepend to agent tasks")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all workers")
    parser.add_argument("--pattern", help="Must resolve to population_search")
    args = parser.parse_args()

    if args.pattern:
        resolved = resolve_pattern(args.pattern).pattern
        if resolved != "population_search":
            parser.error(
                "population_search_orchestrator.py only supports population_search. "
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
        "agent_model": os.environ.get("AGENT_MODEL", settings["model"]),
    }


def _resolve_shared_path(shared_dir: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return shared_dir / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepend_branch_context(nodes: List[QueueNodeSpec], branch: str) -> None:
    if not branch:
        return
    prefix = f"Work on git branch: {branch}\n\n"
    for node in nodes:
        instructions = node.payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            node.payload["instructions"] = prefix + instructions


def _candidate_solution_ref(manifest: PopulationSearchManifest, generation_index: int, candidate_id: str) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/candidates/{candidate_id}/solution.py"


def _candidate_request_ref(manifest: PopulationSearchManifest, generation_index: int, candidate_id: str) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/candidates/{candidate_id}/request.json"


def _candidate_result_ref(manifest: PopulationSearchManifest, generation_index: int, candidate_id: str) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/candidates/{candidate_id}/result.json"


def _leaderboard_ref(manifest: PopulationSearchManifest, generation_index: int) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/leaderboard.json"


def _leaderboard_csv_ref(manifest: PopulationSearchManifest, generation_index: int) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/leaderboard.csv"


def _brief_input_ref(manifest: PopulationSearchManifest, generation_index: int) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/brief-input.json"


def _brief_output_ref(manifest: PopulationSearchManifest, generation_index: int) -> str:
    return f"{manifest.output_root}/generation-{generation_index:03d}/generation_brief.md"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_candidate_request(
    *,
    manifest: PopulationSearchManifest,
    plan: CandidatePlan,
    shared_dir: Path,
    current_best_score: Optional[float],
    previous_leaderboard_ref: str,
    previous_brief_ref: str,
    top_results: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    top_excerpt = []
    for item in top_results[: manifest.review_top_k]:
        top_excerpt.append(
            {
                "candidate_id": item.get("candidate_id", ""),
                "score": item.get(manifest.score_field),
                "correctness_passed": bool(item.get("correctness_passed")),
                "strategy": item.get("strategy", ""),
                "solution_ref": item.get("solution_ref", ""),
                "result_ref": item.get("result_ref", ""),
            }
        )

    payload = {
        "candidate_id": plan.candidate_id,
        "generation_index": plan.generation_index,
        "strategy": plan.strategy,
        "task_ref": manifest.task_ref,
        "current_best_score": current_best_score,
        "score_field": manifest.score_field,
        "score_direction": manifest.score_direction,
        "previous_leaderboard_ref": previous_leaderboard_ref,
        "generation_brief_ref": previous_brief_ref,
        "parent_candidate_ids": list(plan.parent_candidate_ids),
        "parent_solution_refs": list(plan.parent_solution_refs),
        "top_solution_refs": [str(item.get("solution_ref", "") or "") for item in top_results[: manifest.review_top_k]],
        "top_result_refs": [str(item.get("result_ref", "") or "") for item in top_results[: manifest.review_top_k]],
        "leaderboard_excerpt": top_excerpt,
    }
    request_path = _resolve_shared_path(shared_dir, _candidate_request_ref(manifest, plan.generation_index, plan.candidate_id))
    _write_json(request_path, payload)
    return payload


def _build_candidate_nodes(
    *,
    manifest: PopulationSearchManifest,
    plans: List[CandidatePlan],
    shared_dir: Path,
    current_best_score: Optional[float],
    previous_leaderboard_ref: str,
    previous_brief_ref: str,
    top_results: List[Mapping[str, Any]],
) -> List[QueueNodeSpec]:
    nodes: List[QueueNodeSpec] = []
    for plan in plans:
        request_payload = _build_candidate_request(
            manifest=manifest,
            plan=plan,
            shared_dir=shared_dir,
            current_best_score=current_best_score,
            previous_leaderboard_ref=previous_leaderboard_ref,
            previous_brief_ref=previous_brief_ref,
            top_results=top_results,
        )
        input_ref = _candidate_request_ref(manifest, plan.generation_index, plan.candidate_id)
        output_ref = _candidate_solution_ref(manifest, plan.generation_index, plan.candidate_id)
        nodes.append(
            QueueNodeSpec(
                node_id=f"candidate-{plan.candidate_id}",
                role=f"Candidate {plan.candidate_id}",
                task_type="candidate",
                payload={
                    "description": f"Candidate {plan.candidate_id}",
                    "executor": "agent",
                    "instructions": build_candidate_prompt(
                        manifest,
                        input_ref=input_ref,
                        output_ref=output_ref,
                        candidate_id=plan.candidate_id,
                        generation_index=plan.generation_index,
                        strategy=plan.strategy,
                    ),
                    "output_ref": output_ref,
                    "input_ref": input_ref,
                    "shared_workspace": str(shared_dir),
                    "candidate_id": plan.candidate_id,
                    "generation_index": plan.generation_index,
                    "strategy": plan.strategy,
                    "parent_candidate_ids": list(plan.parent_candidate_ids),
                    "parent_solution_refs": list(plan.parent_solution_refs),
                    "request_payload": request_payload,
                    **manifest.candidate_payload,
                },
                kind="candidate",
            )
        )
    return nodes


def _build_evaluation_nodes(
    *,
    manifest: PopulationSearchManifest,
    candidate_records: List[Mapping[str, Any]],
    shared_dir: Path,
) -> List[QueueNodeSpec]:
    nodes: List[QueueNodeSpec] = []
    for record in candidate_records:
        if not bool(record.get("candidate_succeeded")):
            continue
        solution_ref = str(record.get("solution_ref", "") or "")
        if not solution_ref:
            continue
        candidate_id = str(record.get("candidate_id", "") or "")
        generation_index = int(record.get("generation_index", 0) or 0)
        result_ref = _candidate_result_ref(manifest, generation_index, candidate_id)
        nodes.append(
            QueueNodeSpec(
                node_id=f"evaluate-{candidate_id}",
                role=f"Evaluate {candidate_id}",
                task_type="evaluate",
                payload={
                    "description": f"Evaluate {candidate_id}",
                    "executor": "python_handler",
                    "handler": manifest.evaluation_handler,
                    "operation": "evaluate_candidate",
                    "output_ref": result_ref,
                    "task_ref": manifest.task_ref,
                    "solution_ref": solution_ref,
                    "candidate_id": candidate_id,
                    "generation_index": generation_index,
                    "strategy": str(record.get("strategy", "") or ""),
                    "parent_candidate_ids": list(record.get("parent_candidate_ids", []) or []),
                    "shared_workspace": str(shared_dir),
                    **manifest.evaluation_payload,
                },
                kind="evaluate",
            )
        )
    return nodes


def _write_leaderboard_csv(path: Path, rows: List[Mapping[str, Any]], score_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generation_index",
        "candidate_id",
        "strategy",
        "correctness_passed",
        "status",
        score_field,
        "runtime_seconds",
        "speedup_vs_baseline",
        "code_hash",
        "solution_ref",
        "result_ref",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generation_index": row.get("generation_index", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "strategy": row.get("strategy", ""),
                    "correctness_passed": row.get("correctness_passed", False),
                    "status": row.get("status", ""),
                    score_field: row.get(score_field, ""),
                    "runtime_seconds": row.get("runtime_seconds", ""),
                    "speedup_vs_baseline": row.get("speedup_vs_baseline", ""),
                    "code_hash": row.get("code_hash", ""),
                    "solution_ref": row.get("solution_ref", ""),
                    "result_ref": row.get("result_ref", ""),
                    "error": row.get("error", ""),
                }
            )


def _build_brief_node(
    *,
    manifest: PopulationSearchManifest,
    generation_index: int,
    shared_dir: Path,
    sorted_records: List[Mapping[str, Any]],
) -> QueueNodeSpec:
    input_ref = _brief_input_ref(manifest, generation_index)
    output_ref = _brief_output_ref(manifest, generation_index)
    brief_payload = {
        "generation_index": generation_index,
        "score_field": manifest.score_field,
        "score_direction": manifest.score_direction,
        "top_candidates": [
            {
                "candidate_id": item.get("candidate_id", ""),
                "strategy": item.get("strategy", ""),
                "correctness_passed": bool(item.get("correctness_passed")),
                "score": item.get(manifest.score_field),
                "runtime_seconds": item.get("runtime_seconds"),
                "speedup_vs_baseline": item.get("speedup_vs_baseline"),
                "solution_ref": item.get("solution_ref", ""),
                "result_ref": item.get("result_ref", ""),
                "error": item.get("error", ""),
                "parent_candidate_ids": item.get("parent_candidate_ids", []),
            }
            for item in sorted_records[: manifest.review_top_k]
        ],
    }
    _write_json(_resolve_shared_path(shared_dir, input_ref), brief_payload)
    return QueueNodeSpec(
        node_id=f"brief-g{generation_index:03d}",
        role=f"Generation {generation_index} brief",
        task_type="brief",
        payload={
            "description": f"Generation {generation_index} brief",
            "executor": "agent",
            "instructions": build_review_prompt(
                manifest,
                input_ref=input_ref,
                output_ref=output_ref,
                generation_index=generation_index,
            ),
            "output_ref": output_ref,
            "input_ref": input_ref,
            "shared_workspace": str(shared_dir),
            **manifest.review_payload,
        },
        kind="brief",
    )


def _aggregate_queue_metrics(queue_result: Mapping[str, Any]) -> Dict[str, Any]:
    prompt_tokens = 0
    response_tokens = 0
    total_tokens = 0
    estimated_cost_usd = 0.0
    for payload in queue_result.get("results_by_node", {}).values():
        prompt_tokens += int(payload.get("prompt_tokens", 0) or 0)
        response_tokens += int(payload.get("response_tokens", 0) or 0)
        total_tokens += int(payload.get("total_tokens", 0) or 0)
        estimated_cost_usd += float(payload.get("estimated_cost_usd", 0.0) or 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
    }


def _candidate_records_from_run(
    *,
    manifest: PopulationSearchManifest,
    plans: List[CandidatePlan],
    queue_result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    payloads = queue_result.get("results_by_node", {})
    for plan in plans:
        node_id = f"candidate-{plan.candidate_id}"
        payload = payloads.get(node_id, {})
        candidate_succeeded = bool(payload.get("status") == "success" and payload.get("returncode", 1) == 0)
        record = {
            "candidate_id": plan.candidate_id,
            "generation_index": plan.generation_index,
            "strategy": plan.strategy,
            "parent_candidate_ids": list(plan.parent_candidate_ids),
            "parent_solution_refs": list(plan.parent_solution_refs),
            "solution_ref": _candidate_solution_ref(manifest, plan.generation_index, plan.candidate_id),
            "result_ref": _candidate_result_ref(manifest, plan.generation_index, plan.candidate_id),
            "candidate_succeeded": candidate_succeeded,
            "status": "candidate_pass" if candidate_succeeded else "candidate_failure",
            "correctness_passed": False,
            manifest.score_field: 0.0,
            "score": 0.0,
            "runtime_seconds": None,
            "speedup_vs_baseline": 0.0,
            "code_hash": "",
            "error": str(payload.get("error", "") or ""),
        }
        records.append(record)
    return records


def _apply_evaluation_results(
    *,
    manifest: PopulationSearchManifest,
    shared_dir: Path,
    candidate_records: List[Dict[str, Any]],
    evaluation_queue_result: Mapping[str, Any],
) -> None:
    payloads = evaluation_queue_result.get("results_by_node", {})
    records_by_id = {record["candidate_id"]: record for record in candidate_records}
    for node_id, payload in payloads.items():
        if not node_id.startswith("evaluate-"):
            continue
        candidate_id = node_id[len("evaluate-") :]
        record = records_by_id.get(candidate_id)
        if record is None:
            continue
        output_ref = record["result_ref"]
        output_payload = _read_json_if_exists(_resolve_shared_path(shared_dir, output_ref))
        if output_payload is None:
            record["status"] = "evaluation_failure"
            record["error"] = str(payload.get("error", "") or "evaluation output missing")
            continue
        record.update(output_payload)
        record["status"] = str(output_payload.get("status", "evaluated") or "evaluated")
        score_value = _coerce_float(output_payload.get(manifest.score_field))
        record[manifest.score_field] = 0.0 if score_value is None else score_value
        generic_score = _coerce_float(output_payload.get("score"))
        record["score"] = 0.0 if generic_score is None else generic_score


def _select_best_result(
    sorted_records: List[Mapping[str, Any]],
    manifest: PopulationSearchManifest,
) -> Optional[Dict[str, Any]]:
    if not sorted_records:
        return None
    return dict(sorted_records[0])


def _write_final_outputs(
    *,
    manifest: PopulationSearchManifest,
    shared_dir: Path,
    all_results: List[Mapping[str, Any]],
    generation_history: List[Mapping[str, Any]],
    stop_reason: str,
    workspace_summary: Dict[str, Any],
) -> Dict[str, Any]:
    final_dir = _resolve_shared_path(shared_dir, manifest.output_root)
    final_dir.mkdir(parents=True, exist_ok=True)

    sorted_all = sort_candidate_results(all_results, manifest)
    best_result = _select_best_result(sorted_all, manifest)
    leaderboard_path = final_dir / "leaderboard.json"
    leaderboard_csv_path = final_dir / "leaderboard.csv"
    history_path = final_dir / "generation_history.json"
    best_solution_path = final_dir / "best_solution.py"
    best_result_path = final_dir / "best_result.json"

    _write_json(
        leaderboard_path,
        {
            "score_field": manifest.score_field,
            "score_direction": manifest.score_direction,
            "candidate_count": len(sorted_all),
            "best_candidate_id": best_result.get("candidate_id", "") if best_result else "",
            "best_score": best_result.get(manifest.score_field) if best_result else None,
            "candidates": sorted_all,
        },
    )
    _write_leaderboard_csv(leaderboard_csv_path, sorted_all, manifest.score_field)
    _write_json(history_path, {"generations": generation_history, "stop_reason": stop_reason})

    if best_result:
        best_solution_source = _resolve_shared_path(shared_dir, str(best_result.get("solution_ref", "") or ""))
        if best_solution_source.exists():
            shutil.copyfile(best_solution_source, best_solution_path)
        _write_json(best_result_path, best_result)

    return {
        **workspace_summary,
        "leaderboard_path": str(leaderboard_path),
        "leaderboard_csv_path": str(leaderboard_csv_path),
        "generation_history_path": str(history_path),
        "best_solution_path": str(best_solution_path) if best_solution_path.exists() else "",
        "best_result_path": str(best_result_path) if best_result_path.exists() else "",
        "best_candidate_id": best_result.get("candidate_id", "") if best_result else "",
        "best_score": best_result.get(manifest.score_field) if best_result else None,
    }


def main() -> None:
    args = parse_args()
    runtime = _load_runtime_settings()
    agent_model = runtime["agent_model"]

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "40"))
    max_runtime = int(os.environ.get("MAX_RUNTIME_SECONDS", "300"))
    worker_count = int(os.environ.get("WORK_QUEUE_WORKERS", "3"))
    worker_max_concurrent = int(os.environ.get("WORK_QUEUE_MAX_CONCURRENT_LOCAL", "1"))
    max_stage_retries = int(os.environ.get("WORK_QUEUE_STAGE_RETRIES", "1"))

    executor = (args.executor or os.environ.get("COLLAB_EXECUTOR", "host")).strip().lower()
    if executor != "host":
        raise RuntimeError("population_search currently supports only --executor host.")

    if args.shared_dir:
        shared_dir = Path(args.shared_dir)
        shared_dir.mkdir(parents=True, exist_ok=True)
    else:
        shared_dir = Path(PROJECT_ROOT) / "runs" / f"population-search-{int(time.time() * 1000)}"
        shared_dir.mkdir(parents=True)

    try:
        manifest = load_population_manifest(args.task_manifest)
    except ManifestValidationError as exc:
        raise SystemExit(str(exc)) from exc

    task_path = _resolve_shared_path(shared_dir, manifest.task_ref)
    if not task_path.exists():
        raise SystemExit(f"population_search task_ref does not exist: {manifest.task_ref}")

    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")
    broker_router = os.environ.get("BROKER_ROUTER", "tcp://localhost:5555")
    broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
    worker_root = os.path.join(PROJECT_ROOT, "runs", "worker-local")

    print("=" * 70)
    print("  POPULATION_SEARCH ORCHESTRATOR")
    print("=" * 70)
    print(f"Task label: {args.task or 'population_search'}")
    print(f"Manifest: {args.task_manifest}")
    print(f"Task ref: {manifest.task_ref}")
    print(f"Agent model: {agent_model}")
    print(f"Population size: {manifest.population_size}")
    print(f"Generations: {manifest.max_generations} | Patience: {manifest.patience_generations}")
    print(f"Workers: {worker_count} | Local concurrency: {worker_max_concurrent}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    print(f"Shared workspace: {shared_dir}")
    print()

    broker: MessageBroker | None = None
    coordinator: Agent | None = None
    workers: List[Dict[str, Any]] = []
    collector = ResultCollector()

    run_id = f"population-search-{int(time.time() * 1000)}"
    result_topic = f"population_search.results.{run_id}"
    generation_history: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    stop_reason = "max_generations_reached"
    current_best_score: Optional[float] = None
    best_result: Optional[Dict[str, Any]] = None
    no_improvement_generations = 0
    previous_leaderboard_ref = ""
    previous_brief_ref = ""
    top_results: List[Mapping[str, Any]] = []
    queue_metrics = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    started_at = time.time()

    try:
        if broker_mode == "host":
            router_port = _broker_port_from_endpoint(broker_router, 5555)
            sub_port = _broker_port_from_endpoint(broker_sub, 5556)
            broker = MessageBroker(router_port=router_port, pub_port=sub_port, enable_logging=False)
            broker.start()
            time.sleep(0.5)
            print(f"[BROKER] Started on :{router_port}/:{sub_port}")
        else:
            print("[BROKER] Connecting to existing broker")

        coordinator = Agent(
            agent_id=f"population-search-coordinator-{int(time.time())}",
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

        plans = build_generation_zero_plans(manifest)
        for generation_index in range(manifest.max_generations):
            elapsed = time.time() - started_at
            if elapsed >= manifest.max_wall_time_seconds:
                stop_reason = "max_wall_time_seconds_reached"
                break

            candidate_nodes = _build_candidate_nodes(
                manifest=manifest,
                plans=plans,
                shared_dir=shared_dir,
                current_best_score=current_best_score,
                previous_leaderboard_ref=previous_leaderboard_ref,
                previous_brief_ref=previous_brief_ref,
                top_results=top_results,
            )
            _prepend_branch_context(candidate_nodes, args.branch or "")
            candidate_result = run_queue_plan(
                nodes=candidate_nodes,
                coordinator=coordinator,
                collector=collector,
                run_id=run_id,
                result_topic=result_topic,
                global_timeout_seconds=max(1, int(manifest.max_wall_time_seconds - elapsed)),
                max_stage_retries=max_stage_retries,
                progress_label=f"POPSEARCH:g{generation_index:03d}:CAND",
            )
            metrics = _aggregate_queue_metrics(candidate_result)
            for key in queue_metrics:
                queue_metrics[key] += metrics[key]

            candidate_records = _candidate_records_from_run(
                manifest=manifest,
                plans=plans,
                queue_result=candidate_result,
            )

            evaluation_nodes = _build_evaluation_nodes(
                manifest=manifest,
                candidate_records=candidate_records,
                shared_dir=shared_dir,
            )
            evaluation_result = {"build_failures": [], "stage_history": [], "results_by_node": {}, "build_passed": True}
            if evaluation_nodes:
                evaluation_result = run_queue_plan(
                    nodes=evaluation_nodes,
                    coordinator=coordinator,
                    collector=collector,
                    run_id=run_id,
                    result_topic=result_topic,
                    global_timeout_seconds=max(1, int(manifest.max_wall_time_seconds - (time.time() - started_at))),
                    max_stage_retries=max_stage_retries,
                    progress_label=f"POPSEARCH:g{generation_index:03d}:EVAL",
                )
                metrics = _aggregate_queue_metrics(evaluation_result)
                for key in queue_metrics:
                    queue_metrics[key] += metrics[key]

            _apply_evaluation_results(
                manifest=manifest,
                shared_dir=shared_dir,
                candidate_records=candidate_records,
                evaluation_queue_result=evaluation_result,
            )

            sorted_records = sort_candidate_results(candidate_records, manifest)
            leaderboard_payload = {
                "generation_index": generation_index,
                "score_field": manifest.score_field,
                "score_direction": manifest.score_direction,
                "candidate_count": len(sorted_records),
                "best_candidate_id": sorted_records[0].get("candidate_id", "") if sorted_records else "",
                "best_score": sorted_records[0].get(manifest.score_field) if sorted_records else None,
                "candidates": sorted_records,
            }
            _write_json(_resolve_shared_path(shared_dir, _leaderboard_ref(manifest, generation_index)), leaderboard_payload)
            _write_leaderboard_csv(
                _resolve_shared_path(shared_dir, _leaderboard_csv_ref(manifest, generation_index)),
                sorted_records,
                manifest.score_field,
            )

            brief_ref = ""
            brief_node = _build_brief_node(
                manifest=manifest,
                generation_index=generation_index,
                shared_dir=shared_dir,
                sorted_records=sorted_records,
            )
            _prepend_branch_context([brief_node], args.branch or "")
            brief_result = run_queue_plan(
                nodes=[brief_node],
                coordinator=coordinator,
                collector=collector,
                run_id=run_id,
                result_topic=result_topic,
                global_timeout_seconds=max(1, int(manifest.max_wall_time_seconds - (time.time() - started_at))),
                max_stage_retries=0,
                progress_label=f"POPSEARCH:g{generation_index:03d}:BRIEF",
            )
            metrics = _aggregate_queue_metrics(brief_result)
            for key in queue_metrics:
                queue_metrics[key] += metrics[key]
            brief_path = _resolve_shared_path(shared_dir, _brief_output_ref(manifest, generation_index))
            if brief_path.exists():
                brief_ref = _brief_output_ref(manifest, generation_index)

            current_generation_best = _select_best_result(sorted_records, manifest)
            current_generation_best_score = (
                _coerce_float(current_generation_best.get(manifest.score_field)) if current_generation_best else None
            )
            improved = score_improved(
                previous_best_score=current_best_score,
                current_best_score=current_generation_best_score,
                manifest=manifest,
            )
            if improved:
                current_best_score = current_generation_best_score
                best_result = dict(current_generation_best) if current_generation_best else best_result
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            selection = build_generation_selection(
                sorted_records=sorted_records,
                manifest=manifest,
                generation_index=generation_index,
            )
            generation_summary = {
                "generation_index": generation_index,
                "candidate_count": len(plans),
                "best_candidate_id": current_generation_best.get("candidate_id", "") if current_generation_best else "",
                "best_score": current_generation_best_score,
                "improved": improved,
                "candidate_build_failures": list(candidate_result.get("build_failures", [])),
                "evaluation_build_failures": list(evaluation_result.get("build_failures", [])),
                "brief_build_failures": list(brief_result.get("build_failures", [])),
                "leaderboard_ref": _leaderboard_ref(manifest, generation_index),
                "leaderboard_csv_ref": _leaderboard_csv_ref(manifest, generation_index),
                "brief_ref": brief_ref,
                "selection": {
                    "elite_candidate_ids": selection["elite_candidate_ids"],
                    "diversity_candidate_ids": selection["diversity_candidate_ids"],
                },
            }
            generation_history.append(generation_summary)
            all_results.extend(sorted_records)

            previous_leaderboard_ref = _leaderboard_ref(manifest, generation_index)
            previous_brief_ref = brief_ref
            top_results = sorted_records[: manifest.review_top_k]

            if no_improvement_generations >= manifest.patience_generations:
                stop_reason = "patience_exhausted"
                break

            plans = selection["plans"]
        else:
            stop_reason = "max_generations_reached"
    finally:
        stop_worker_daemons(workers)
        if coordinator is not None:
            coordinator.stop()
        if broker is not None:
            broker.stop()

    final_payload = _write_final_outputs(
        manifest=manifest,
        shared_dir=shared_dir,
        all_results=all_results,
        generation_history=generation_history,
        stop_reason=stop_reason,
        workspace_summary={
            "status": (
                "pass"
                if best_result
                and bool(best_result.get("correctness_passed"))
                and _resolve_shared_path(shared_dir, str(best_result.get("solution_ref", "") or "")).exists()
                else "fail"
            ),
            "pattern": "population_search",
            "task": args.task or "population_search",
            "manifest_path": args.task_manifest,
            "manifest": manifest_to_dict(manifest),
            "workspace": str(shared_dir),
            "generation_count": len(generation_history),
            "worker_count": worker_count,
            "worker_max_concurrent_local": worker_max_concurrent,
            "stop_reason": stop_reason,
            "queue_metrics": {
                "prompt_tokens": queue_metrics["prompt_tokens"],
                "response_tokens": queue_metrics["response_tokens"],
                "total_tokens": queue_metrics["total_tokens"],
                "estimated_cost_usd": round(float(queue_metrics["estimated_cost_usd"]), 6),
            },
        },
    )

    summary_path = shared_dir / "run_summary.json"
    _write_json(summary_path, final_payload)

    print("\n" + "=" * 70)
    print("  POPULATION_SEARCH COMPLETE")
    print("=" * 70)
    print(f"Build result: {'PASSED' if final_payload['status'] == 'pass' else 'FAILED'}")
    print(f"Best score: {final_payload.get('best_score')}")
    print(f"Best solution: {final_payload.get('best_solution_path', '')}")
    print(f"Run summary: {summary_path}")

    sys.exit(0 if final_payload["status"] == "pass" else 1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda _sig, _frame: sys.exit(130))
    main()
