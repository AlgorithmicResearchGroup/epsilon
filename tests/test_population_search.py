from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.population_search_csv.local_tasks import evaluate_solution_file
from examples.population_search_csv.workflow import DemoConfig, build_manifest, materialize_problem_bundle
from orchestrators.patterns import resolve_pattern
from orchestrators.population_search import (
    build_generation_selection,
    load_population_manifest,
    score_improved,
    sort_candidate_results,
)
from orchestrators.scale_topologies import ManifestValidationError
from runtime.worker_daemon import WorkerDaemon


OPTIMIZED_SOLUTION = """import csv


def aggregate_orders(csv_path: str) -> dict:
    region_totals = {}
    customer_totals = {}
    product_units = {}
    total_revenue = 0.0
    row_count = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        next(reader, None)
        for customer_id, region, product, quantity_s, unit_price_s, discount_pct_s, _timestamp in reader:
            row_count += 1
            quantity = int(quantity_s)
            revenue = quantity * float(unit_price_s) * (1.0 - (float(discount_pct_s) * 0.01))
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


def test_population_search_pattern_registry_exposes_aliases():
    assert resolve_pattern("population_search").pattern == "population_search"
    assert resolve_pattern("population").pattern == "population_search"


def test_population_manifest_rejects_invalid_seed_counts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_ref": "input/problem.json",
                "output_root": "population",
                "candidate_task_template": "candidate {input_ref} {output_ref}",
                "review_task_template": "review {input_ref} {output_ref}",
                "evaluation_handler": "examples.population_search_csv.local_tasks:run_task",
                "population_size": 4,
                "elite_count": 2,
                "diversity_count": 2,
                "fresh_count": 2,
                "review_top_k": 2,
                "score_field": "score",
                "score_direction": "maximize",
                "max_generations": 3,
                "max_wall_time_seconds": 300,
                "min_improvement_delta": 0.01,
                "patience_generations": 2,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError):
        load_population_manifest(str(manifest_path))


def test_generation_selection_keeps_elites_diversity_and_fresh():
    manifest = load_population_manifest_from_dict(
        {
            "task_ref": "input/problem.json",
            "output_root": "population",
            "candidate_task_template": "candidate {input_ref} {output_ref}",
            "review_task_template": "review {input_ref} {output_ref}",
            "evaluation_handler": "examples.population_search_csv.local_tasks:run_task",
            "population_size": 6,
            "elite_count": 2,
            "diversity_count": 1,
            "fresh_count": 1,
            "review_top_k": 2,
            "score_field": "speedup_vs_baseline",
            "score_direction": "maximize",
            "max_generations": 3,
            "max_wall_time_seconds": 300,
            "min_improvement_delta": 0.01,
            "patience_generations": 2,
        }
    )
    sorted_records = sort_candidate_results(
        [
            {"candidate_id": "a", "correctness_passed": True, "speedup_vs_baseline": 1.8, "solution_ref": "a.py", "code_hash": "h1"},
            {"candidate_id": "b", "correctness_passed": True, "speedup_vs_baseline": 1.7, "solution_ref": "b.py", "code_hash": "h1"},
            {"candidate_id": "c", "correctness_passed": True, "speedup_vs_baseline": 1.6, "solution_ref": "c.py", "code_hash": "h2"},
            {"candidate_id": "d", "correctness_passed": True, "speedup_vs_baseline": 1.5, "solution_ref": "d.py", "code_hash": "h3"},
        ],
        manifest,
    )

    selection = build_generation_selection(sorted_records=sorted_records, manifest=manifest, generation_index=0)

    assert selection["elite_candidate_ids"] == ["a", "b"]
    assert selection["diversity_candidate_ids"] == ["c"]
    plans = selection["plans"]
    assert len(plans) == 6
    assert [plan.strategy for plan in plans].count("fresh") == 1
    assert [plan.strategy for plan in plans].count("offspring") == 2


def test_score_improved_uses_delta():
    manifest = load_population_manifest_from_dict(
        {
            "task_ref": "input/problem.json",
            "output_root": "population",
            "candidate_task_template": "candidate {input_ref} {output_ref}",
            "review_task_template": "review {input_ref} {output_ref}",
            "evaluation_handler": "examples.population_search_csv.local_tasks:run_task",
            "population_size": 4,
            "elite_count": 1,
            "diversity_count": 1,
            "fresh_count": 1,
            "review_top_k": 2,
            "score_field": "score",
            "score_direction": "maximize",
            "max_generations": 3,
            "max_wall_time_seconds": 300,
            "min_improvement_delta": 0.05,
            "patience_generations": 2,
        }
    )
    assert score_improved(previous_best_score=None, current_best_score=1.0, manifest=manifest) is True
    assert score_improved(previous_best_score=1.0, current_best_score=1.02, manifest=manifest) is False
    assert score_improved(previous_best_score=1.0, current_best_score=1.06, manifest=manifest) is True


def test_csv_evaluator_scores_optimized_solution(tmp_path):
    config = DemoConfig(correctness_rows=(80, 140), performance_rows=(6000, 9000), performance_repetitions=1)
    bundle = materialize_problem_bundle(run_dir=tmp_path, config=config)
    optimized_path = tmp_path / "input" / "optimized_solution.py"
    optimized_path.write_text(OPTIMIZED_SOLUTION, encoding="utf-8")

    baseline_result = evaluate_solution_file(
        solution_path=Path(bundle["baseline_solution_path"]),
        problem_spec_path=Path(bundle["problem_path"]),
    )
    optimized_result = evaluate_solution_file(
        solution_path=optimized_path,
        problem_spec_path=Path(bundle["problem_path"]),
    )

    assert baseline_result["correctness_passed"] is True
    assert optimized_result["correctness_passed"] is True
    assert optimized_result["speedup_vs_baseline"] > 1.0


def test_population_search_workflow_builds_manifest(tmp_path):
    config = DemoConfig()
    bundle = materialize_problem_bundle(run_dir=tmp_path, config=config)
    manifest_path = build_manifest(run_dir=tmp_path, config=config, task_ref=bundle["task_ref"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_ref"] == "input/problem.json"
    assert manifest["score_field"] == "speedup_vs_baseline"
    assert manifest["evaluation_handler"] == "examples.population_search_csv.local_tasks:run_task"


def test_python_handler_evaluates_candidate(tmp_path):
    config = DemoConfig(correctness_rows=(40, 60), performance_rows=(1200, 1600), performance_repetitions=1)
    bundle = materialize_problem_bundle(run_dir=tmp_path, config=config)
    solution_path = tmp_path / "input" / "optimized_solution.py"
    solution_path.write_text(OPTIMIZED_SOLUTION, encoding="utf-8")

    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.work_root = tmp_path

    result = WorkerDaemon._execute_python_handler_task(
        daemon,
        "task-1",
        {
            "benchmark_id": "bench",
            "shared_workspace": str(tmp_path),
            "handler": "examples.population_search_csv.local_tasks:run_task",
            "operation": "evaluate_candidate",
            "output_ref": "population_search_csv/candidate-result.json",
            "task_ref": "input/problem.json",
            "solution_ref": "input/optimized_solution.py",
            "candidate_id": "g000-c001",
            "generation_index": 0,
            "strategy": "fresh",
            "parent_candidate_ids": [],
        },
    )
    assert result["output_exists"] is True
    payload = json.loads((tmp_path / "population_search_csv" / "candidate-result.json").read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "g000-c001"
    assert payload["correctness_passed"] is True
    assert payload["solution_ref"] == "input/optimized_solution.py"


def load_population_manifest_from_dict(payload: dict) -> Any:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_population_manifest(str(manifest_path))
