"""Deterministic evaluator helpers for the population-search CSV demo."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _resolve_ref_path(task_root: Path, ref: str) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    return task_root / candidate


def _round_money(value: float) -> float:
    return round(float(value), 6)


def compute_expected_aggregate(csv_path: Path) -> Dict[str, Any]:
    region_totals: Dict[str, float] = {}
    customer_totals: Dict[str, float] = {}
    product_units: Dict[str, int] = {}
    total_revenue = 0.0
    row_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            row_count += 1
            region = str(row["region"])
            customer_id = str(row["customer_id"])
            product = str(row["product"])
            quantity = int(row["quantity"])
            unit_price = float(row["unit_price"])
            discount_pct = float(row["discount_pct"])
            revenue = quantity * unit_price * (1.0 - (discount_pct / 100.0))
            total_revenue += revenue
            region_totals[region] = region_totals.get(region, 0.0) + revenue
            customer_totals[customer_id] = customer_totals.get(customer_id, 0.0) + revenue
            product_units[product] = product_units.get(product, 0) + quantity

    top_customers = [
        {"customer_id": customer_id, "revenue": _round_money(revenue)}
        for customer_id, revenue in sorted(customer_totals.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    return {
        "row_count": row_count,
        "total_revenue": _round_money(total_revenue),
        "region_totals": {key: _round_money(region_totals[key]) for key in sorted(region_totals)},
        "top_customers": top_customers,
        "product_units": {key: int(product_units[key]) for key in sorted(product_units)},
    }


def _load_solution_module(solution_path: Path) -> Any:
    module_name = f"population_search_solution_{hashlib.sha1(str(solution_path).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import solution at {solution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_output(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("aggregate_orders must return a dict")
    normalized = {
        "row_count": int(value.get("row_count", 0)),
        "total_revenue": _round_money(float(value.get("total_revenue", 0.0))),
        "region_totals": {},
        "top_customers": [],
        "product_units": {},
    }
    for key, amount in dict(value.get("region_totals", {}) or {}).items():
        normalized["region_totals"][str(key)] = _round_money(float(amount))
    for item in list(value.get("top_customers", []) or [])[:5]:
        if not isinstance(item, dict):
            continue
        normalized["top_customers"].append(
            {
                "customer_id": str(item.get("customer_id", "")),
                "revenue": _round_money(float(item.get("revenue", 0.0))),
            }
        )
    for key, units in dict(value.get("product_units", {}) or {}).items():
        normalized["product_units"][str(key)] = int(units)
    normalized["region_totals"] = {key: normalized["region_totals"][key] for key in sorted(normalized["region_totals"])}
    normalized["product_units"] = {key: normalized["product_units"][key] for key in sorted(normalized["product_units"])}
    normalized["top_customers"] = sorted(
        normalized["top_customers"],
        key=lambda item: (-item["revenue"], item["customer_id"]),
    )[:5]
    return normalized


def evaluate_solution_file(
    *,
    solution_path: Path,
    problem_spec_path: Path,
) -> Dict[str, Any]:
    problem_spec = json.loads(problem_spec_path.read_text(encoding="utf-8"))
    function_name = str(problem_spec["function_name"])
    baseline_runtime_seconds = float(problem_spec.get("baseline_runtime_seconds", 0.0) or 0.0)
    run_root = problem_spec_path.parent.parent

    module = _load_solution_module(solution_path)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"Solution module must define callable '{function_name}'")

    correctness_checks: List[Dict[str, Any]] = []
    correctness_passed = True
    error = ""
    for case in list(problem_spec.get("correctness_cases", [])):
        csv_path = _resolve_ref_path(run_root, str(case["csv_ref"]))
        expected_path = _resolve_ref_path(run_root, str(case["expected_ref"]))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        try:
            actual = _normalize_output(function(str(csv_path)))
        except Exception as exc:  # pragma: no cover - exercised via caller expectations
            correctness_passed = False
            error = f"{type(exc).__name__}: {exc}"
            correctness_checks.append(
                {
                    "case_id": str(case.get("case_id", csv_path.name)),
                    "passed": False,
                    "error": error,
                }
            )
            break

        passed = actual == expected
        if not passed and not error:
            error = f"Output mismatch for {case.get('case_id', csv_path.name)}"
        correctness_checks.append(
            {
                "case_id": str(case.get("case_id", csv_path.name)),
                "passed": passed,
            }
        )
        correctness_passed = correctness_passed and passed
        if not correctness_passed:
            break

    runtime_seconds = None
    speedup_vs_baseline = 0.0
    performance_runs: List[Dict[str, Any]] = []
    if correctness_passed:
        perf_samples: List[float] = []
        for case in list(problem_spec.get("performance_cases", [])):
            csv_path = _resolve_ref_path(run_root, str(case["csv_ref"]))
            repetitions = max(1, int(case.get("repetitions", 1) or 1))
            timings: List[float] = []
            for _ in range(repetitions):
                start = time.perf_counter()
                _normalize_output(function(str(csv_path)))
                timings.append(time.perf_counter() - start)
            median_seconds = float(statistics.median(timings))
            performance_runs.append(
                {
                    "case_id": str(case.get("case_id", csv_path.name)),
                    "median_seconds": median_seconds,
                    "repetitions": repetitions,
                }
            )
            perf_samples.append(median_seconds)
        runtime_seconds = float(sum(perf_samples))
        if runtime_seconds > 0 and baseline_runtime_seconds > 0:
            speedup_vs_baseline = baseline_runtime_seconds / runtime_seconds

    code_text = solution_path.read_text(encoding="utf-8", errors="replace")
    return {
        "status": "evaluated" if correctness_passed else "incorrect",
        "correctness_passed": correctness_passed,
        "correctness_checks": correctness_checks,
        "runtime_seconds": runtime_seconds,
        "baseline_runtime_seconds": baseline_runtime_seconds,
        "speedup_vs_baseline": round(float(speedup_vs_baseline), 6),
        "score": round(float(speedup_vs_baseline), 6),
        "code_hash": hashlib.sha256(code_text.encode("utf-8")).hexdigest(),
        "code_size_bytes": len(code_text.encode("utf-8")),
        "performance_runs": performance_runs,
        "error": error,
    }


def run_task(*, task_id: str, task_payload: Dict[str, Any], task_root: Path) -> Dict[str, Any]:
    operation = str(task_payload.get("operation", "") or "").strip().lower()
    if operation != "evaluate_candidate":
        raise ValueError(f"Unsupported operation '{operation}'")

    output_ref = str(task_payload.get("output_ref", "") or "").strip()
    if not output_ref:
        raise ValueError("python_handler task requires 'output_ref'")
    output_path = _resolve_ref_path(task_root, output_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    solution_ref = str(task_payload.get("solution_ref", "") or "").strip()
    task_ref = str(task_payload.get("task_ref", "") or "").strip()
    if not solution_ref or not task_ref:
        raise ValueError("evaluate_candidate requires 'solution_ref' and 'task_ref'")

    evaluation = evaluate_solution_file(
        solution_path=_resolve_ref_path(task_root, solution_ref),
        problem_spec_path=_resolve_ref_path(task_root, task_ref),
    )
    payload = {
        "candidate_id": str(task_payload.get("candidate_id", "") or ""),
        "generation_index": int(task_payload.get("generation_index", 0) or 0),
        "strategy": str(task_payload.get("strategy", "") or ""),
        "parent_candidate_ids": list(task_payload.get("parent_candidate_ids", []) or []),
        "solution_ref": solution_ref,
        "result_ref": output_ref,
        **evaluation,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "output_path": str(output_path),
        "output_exists": True,
        "output_preview": json.dumps(payload)[:2000],
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
