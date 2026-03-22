"""Workflow helpers for the population-search CSV optimization demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from examples.population_search_csv.local_tasks import compute_expected_aggregate, evaluate_solution_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It describes one optimization attempt for the same CSV aggregation task. The JSON includes:
    - `task_ref`: a path to the canonical task spec
    - `parent_solution_refs`: optional prior solutions worth learning from
    - `generation_brief_ref`: optional markdown brief from the previous generation
    - `leaderboard_excerpt`: top scoring candidates so far

    Write a Python module to `{output_ref}` that defines exactly:

    ```python
    def aggregate_orders(csv_path: str) -> dict:
        ...
    ```

    Requirements:
    - Use only the Python standard library.
    - Preserve the exact output contract documented in the task spec.
    - Favor correctness first, then runtime.
    - You may inspect any referenced solution files and the generation brief before writing code.
    - Do not write any files besides `{output_ref}`.
    - After writing the module, run `python -m py_compile {output_ref}` with `run_bash`. If it fails, fix the code and validate again before calling `done`.
    - Keep the module self-contained: no network, no subprocesses, no caches on disk.
    - Write the module only, then call `done`.
    """
).strip()

REVIEW_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It contains the top results for one generation of the same optimization task.
    Inspect the referenced solution files and write concise markdown to `{output_ref}` with exactly these sections:

    ## What Worked
    ## What Hurt
    ## Next Moves

    Rules:
    - Ground the brief in the actual top candidates and their scores.
    - Point out correctness failures separately from performance issues.
    - Keep it short and concrete so the next generation can act on it.
    - Write markdown only, then call `done`.
    """
).strip()

BASELINE_SOLUTION = textwrap.dedent(
    """
    import csv


    def aggregate_orders(csv_path: str) -> dict:
        region_totals = {}
        customer_totals = {}
        product_units = {}
        total_revenue = 0.0
        row_count = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            for row in reader:
                row_count += 1
                customer_id = row["customer_id"]
                region = row["region"]
                product = row["product"]
                quantity = int(row["quantity"])
                unit_price = float(row["unit_price"])
                discount_pct = float(row["discount_pct"])
                revenue = quantity * unit_price * (1.0 - (discount_pct / 100.0))
                total_revenue += revenue
                region_totals[region] = region_totals.get(region, 0.0) + revenue
                customer_totals[customer_id] = customer_totals.get(customer_id, 0.0) + revenue
                product_units[product] = product_units.get(product, 0) + quantity

        top_customers = [
            {"customer_id": customer_id, "revenue": round(revenue, 6)}
            for customer_id, revenue in sorted(customer_totals.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
        return {
            "row_count": row_count,
            "total_revenue": round(total_revenue, 6),
            "region_totals": {key: round(region_totals[key], 6) for key in sorted(region_totals)},
            "top_customers": top_customers,
            "product_units": {key: product_units[key] for key in sorted(product_units)},
        }
    """
).strip() + "\n"


@dataclass(frozen=True)
class DemoConfig:
    population_size: int = 8
    elite_count: int = 2
    diversity_count: int = 2
    fresh_count: int = 2
    review_top_k: int = 3
    max_generations: int = 4
    max_wall_time_seconds: int = 1200
    min_improvement_delta: float = 0.02
    patience_generations: int = 2
    worker_count: int = 4
    max_iterations: int = 18
    max_runtime_seconds: int = 180
    agent_model: str = "openai/gpt-5.4"
    seed: int = 17
    correctness_rows: tuple[int, int] = (80, 220)
    performance_rows: tuple[int, int] = (6000, 12000)
    performance_repetitions: int = 2


def _validate_model_credentials(model_name: str) -> None:
    provider = str(model_name or "").split("/", 1)[0].strip().lower()
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is required to run the population-search demo with model '{model_name}'."
        )
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"ANTHROPIC_API_KEY is required to run the population-search demo with model '{model_name}'."
        )


def _make_row(index: int, *, seed: int) -> List[str]:
    regions = ("north", "south", "east", "west", "central")
    products = ("alpha", "beta", "gamma", "delta", "omega", "sigma")
    customer = f"cust-{((index * 37) + seed) % 400:04d}"
    region = regions[((index * 5) + seed) % len(regions)]
    product = products[((index * 7) + seed) % len(products)]
    quantity = 1 + (((index * 11) + seed) % 5)
    unit_price = f"{3.50 + ((((index * 17) + seed) % 250) / 10.0):.2f}"
    discount_pct = str((0, 5, 10, 15, 20)[((index * 13) + seed) % 5])
    timestamp = f"2025-01-{((index % 27) + 1):02d}T{(index % 24):02d}:{(index % 60):02d}:00"
    return [customer, region, product, str(quantity), unit_price, discount_pct, timestamp]


def _write_dataset(path: Path, *, row_count: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        sink.write("customer_id,region,product,quantity,unit_price,discount_pct,timestamp\n")
        for index in range(row_count):
            sink.write(",".join(_make_row(index, seed=seed)) + "\n")


def materialize_problem_bundle(*, run_dir: Path, config: DemoConfig) -> Dict[str, Any]:
    input_dir = run_dir / "input"
    datasets_dir = input_dir / "datasets"
    expected_dir = input_dir / "expected"
    baseline_path = input_dir / "baseline_solution.py"
    task_ref = "input/problem.json"
    problem_path = run_dir / task_ref

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(BASELINE_SOLUTION, encoding="utf-8")

    correctness_cases = []
    for ordinal, row_count in enumerate(config.correctness_rows, start=1):
        csv_path = datasets_dir / f"correctness-{ordinal:02d}.csv"
        expected_path = expected_dir / f"correctness-{ordinal:02d}.json"
        _write_dataset(csv_path, row_count=row_count, seed=config.seed + ordinal)
        module = {"csv_ref": str(csv_path.relative_to(run_dir)), "expected_ref": str(expected_path.relative_to(run_dir))}
        correctness_cases.append(
            {
                "case_id": f"correctness-{ordinal:02d}",
                "csv_ref": module["csv_ref"],
                "expected_ref": module["expected_ref"],
            }
        )
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(json.dumps(compute_expected_aggregate(csv_path), indent=2), encoding="utf-8")

    performance_cases = []
    for ordinal, row_count in enumerate(config.performance_rows, start=1):
        csv_path = datasets_dir / f"performance-{ordinal:02d}.csv"
        _write_dataset(csv_path, row_count=row_count, seed=config.seed + 100 + ordinal)
        performance_cases.append(
            {
                "case_id": f"performance-{ordinal:02d}",
                "csv_ref": str(csv_path.relative_to(run_dir)),
                "repetitions": config.performance_repetitions,
            }
        )

    problem_spec = {
        "title": "CSV aggregation optimization",
        "function_name": "aggregate_orders",
        "task_summary": (
            "Optimize a deterministic Python function that aggregates an orders CSV into summary statistics. "
            "Preserve exact output shape and improve runtime over the provided baseline."
        ),
        "output_contract": {
            "row_count": "int",
            "total_revenue": "float rounded to 6 decimals",
            "region_totals": "mapping of region to rounded revenue",
            "top_customers": "top 5 customers by revenue, sorted desc then id asc",
            "product_units": "mapping of product to integer unit totals",
        },
        "baseline_solution_ref": str(baseline_path.relative_to(run_dir)),
        "correctness_cases": correctness_cases,
        "performance_cases": performance_cases,
        "baseline_runtime_seconds": 0.0,
    }
    problem_path.write_text(json.dumps(problem_spec, indent=2), encoding="utf-8")

    baseline_eval = evaluate_solution_file(solution_path=baseline_path, problem_spec_path=problem_path)
    problem_spec["baseline_runtime_seconds"] = baseline_eval["runtime_seconds"]
    problem_spec["baseline_speedup"] = 1.0
    problem_spec["baseline_score"] = 1.0
    problem_spec["baseline_metrics"] = {
        "runtime_seconds": baseline_eval["runtime_seconds"],
        "speedup_vs_baseline": baseline_eval["speedup_vs_baseline"],
        "code_size_bytes": baseline_eval["code_size_bytes"],
    }
    problem_path.write_text(json.dumps(problem_spec, indent=2), encoding="utf-8")

    return {
        "task_ref": task_ref,
        "problem_path": str(problem_path),
        "baseline_solution_path": str(baseline_path),
        "baseline_runtime_seconds": baseline_eval["runtime_seconds"],
    }


def build_manifest(*, run_dir: Path, config: DemoConfig, task_ref: str) -> Path:
    manifest = {
        "task_ref": task_ref,
        "output_root": "population_search_csv",
        "candidate_task_template": CANDIDATE_PROMPT,
        "review_task_template": REVIEW_PROMPT,
        "evaluation_handler": "examples.population_search_csv.local_tasks:run_task",
        "evaluation_payload": {},
        "population_size": config.population_size,
        "elite_count": config.elite_count,
        "diversity_count": config.diversity_count,
        "fresh_count": config.fresh_count,
        "review_top_k": config.review_top_k,
        "score_field": "speedup_vs_baseline",
        "score_direction": "maximize",
        "max_generations": config.max_generations,
        "max_wall_time_seconds": config.max_wall_time_seconds,
        "min_improvement_delta": config.min_improvement_delta,
        "patience_generations": config.patience_generations,
    }
    manifest_path = run_dir / "population_search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_population_search(
    *,
    manifest_path: Path,
    shared_dir: Path,
    config: DemoConfig,
) -> Dict[str, Any]:
    _validate_model_credentials(config.agent_model)
    env = os.environ.copy()
    env["MAX_WAVES"] = "0"
    env["WORK_QUEUE_WORKERS"] = str(config.worker_count)
    env["WORK_QUEUE_MAX_CONCURRENT_LOCAL"] = "1"
    env["MAX_ITERATIONS"] = str(config.max_iterations)
    env["MAX_RUNTIME_SECONDS"] = str(config.max_runtime_seconds)
    env["AGENT_MODEL"] = config.agent_model

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "orchestrate.py"),
        "--pattern",
        "population_search",
        "--task-manifest",
        str(manifest_path),
        "--shared-dir",
        str(shared_dir),
        "population_search_csv",
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    return json.loads((shared_dir / "run_summary.json").read_text(encoding="utf-8"))


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    config = DemoConfig(
        population_size=args.population_size,
        elite_count=args.elite_count,
        diversity_count=args.diversity_count,
        fresh_count=args.fresh_count,
        review_top_k=args.review_top_k,
        max_generations=args.max_generations,
        max_wall_time_seconds=args.max_wall_time_seconds,
        min_improvement_delta=args.min_improvement_delta,
        patience_generations=args.patience_generations,
        worker_count=args.worker_count,
        max_iterations=args.max_iterations,
        max_runtime_seconds=args.max_runtime_seconds,
        agent_model=args.agent_model,
        seed=args.seed,
    )

    run_dir = Path(args.run_dir) if args.run_dir else PROJECT_ROOT / "runs" / f"population-search-csv-{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = materialize_problem_bundle(run_dir=run_dir, config=config)
    manifest_path = build_manifest(run_dir=run_dir, config=config, task_ref=bundle["task_ref"])
    run_summary = run_population_search(manifest_path=manifest_path, shared_dir=run_dir, config=config)

    final_summary = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "problem_path": bundle["problem_path"],
        "baseline_solution_path": bundle["baseline_solution_path"],
        "baseline_runtime_seconds": bundle["baseline_runtime_seconds"],
        "best_solution_path": run_summary.get("best_solution_path", ""),
        "best_result_path": run_summary.get("best_result_path", ""),
        "leaderboard_path": run_summary.get("leaderboard_path", ""),
        "generation_history_path": run_summary.get("generation_history_path", ""),
        "best_score": run_summary.get("best_score"),
        "generation_count": run_summary.get("generation_count", 0),
        "stop_reason": run_summary.get("stop_reason", ""),
        "queue_metrics": run_summary.get("queue_metrics", {}),
    }
    summary_path = run_dir / "demo_summary.json"
    summary_path.write_text(json.dumps({"final_summary": final_summary, "run_summary": run_summary}, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "final_summary": final_summary, "run_summary": run_summary}


def parse_args() -> argparse.Namespace:
    defaults = DemoConfig()
    parser = argparse.ArgumentParser(description="Run the population-search CSV optimization demo")
    parser.add_argument("--run-dir")
    parser.add_argument("--population-size", type=int, default=defaults.population_size)
    parser.add_argument("--elite-count", type=int, default=defaults.elite_count)
    parser.add_argument("--diversity-count", type=int, default=defaults.diversity_count)
    parser.add_argument("--fresh-count", type=int, default=defaults.fresh_count)
    parser.add_argument("--review-top-k", type=int, default=defaults.review_top_k)
    parser.add_argument("--max-generations", type=int, default=defaults.max_generations)
    parser.add_argument("--max-wall-time-seconds", type=int, default=defaults.max_wall_time_seconds)
    parser.add_argument("--min-improvement-delta", type=float, default=defaults.min_improvement_delta)
    parser.add_argument("--patience-generations", type=int, default=defaults.patience_generations)
    parser.add_argument("--worker-count", type=int, default=defaults.worker_count)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--max-runtime-seconds", type=int, default=defaults.max_runtime_seconds)
    parser.add_argument("--agent-model", default=defaults.agent_model)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()
