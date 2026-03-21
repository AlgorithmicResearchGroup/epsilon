"""Helpers for iterative population-search orchestration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from orchestrators.scale_topologies import ManifestValidationError, render_task_template


@dataclass(frozen=True)
class PopulationSearchManifest:
    task_ref: str
    output_root: str
    candidate_task_template: str
    review_task_template: str
    evaluation_handler: str
    population_size: int
    elite_count: int
    diversity_count: int
    fresh_count: int
    review_top_k: int
    score_field: str
    score_direction: str
    max_generations: int
    max_wall_time_seconds: int
    min_improvement_delta: float
    patience_generations: int
    evaluation_payload: Dict[str, Any] = field(default_factory=dict)
    candidate_payload: Dict[str, Any] = field(default_factory=dict)
    review_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    generation_index: int
    strategy: str
    parent_candidate_ids: List[str] = field(default_factory=list)
    parent_solution_refs: List[str] = field(default_factory=list)


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"Manifest field '{key}' must be a non-empty string.")
    return value.strip()


def _require_object(data: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestValidationError(f"Manifest field '{key}' must be an object.")
    return dict(value)


def _require_int(data: Mapping[str, Any], key: str, *, minimum: int) -> int:
    try:
        value = int(data.get(key))
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"Manifest field '{key}' must be an integer.") from exc
    if value < minimum:
        raise ManifestValidationError(f"Manifest field '{key}' must be >= {minimum}.")
    return value


def _require_float(data: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> float:
    try:
        value = float(data.get(key))
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"Manifest field '{key}' must be a number.") from exc
    if value < minimum:
        raise ManifestValidationError(f"Manifest field '{key}' must be >= {minimum}.")
    return value


def load_population_manifest(path: str) -> PopulationSearchManifest:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestValidationError("Task manifest must be a JSON object.")

    score_direction = str(data.get("score_direction", "maximize") or "maximize").strip().lower()
    if score_direction not in {"maximize", "minimize"}:
        raise ManifestValidationError("Manifest field 'score_direction' must be 'maximize' or 'minimize'.")

    output_root = _require_string(data, "output_root").strip().strip("/")
    if not output_root:
        raise ManifestValidationError("Manifest field 'output_root' must not be empty.")

    population_size = _require_int(data, "population_size", minimum=2)
    elite_count = _require_int(data, "elite_count", minimum=1)
    diversity_count = _require_int(data, "diversity_count", minimum=0)
    fresh_count = _require_int(data, "fresh_count", minimum=0)
    if elite_count + diversity_count + fresh_count > population_size:
        raise ManifestValidationError(
            "Manifest fields 'elite_count' + 'diversity_count' + 'fresh_count' must be <= population_size."
        )

    return PopulationSearchManifest(
        task_ref=_require_string(data, "task_ref"),
        output_root=output_root,
        candidate_task_template=_require_string(data, "candidate_task_template"),
        review_task_template=_require_string(data, "review_task_template"),
        evaluation_handler=_require_string(data, "evaluation_handler"),
        population_size=population_size,
        elite_count=elite_count,
        diversity_count=diversity_count,
        fresh_count=fresh_count,
        review_top_k=_require_int(data, "review_top_k", minimum=1),
        score_field=_require_string(data, "score_field"),
        score_direction=score_direction,
        max_generations=_require_int(data, "max_generations", minimum=1),
        max_wall_time_seconds=_require_int(data, "max_wall_time_seconds", minimum=1),
        min_improvement_delta=_require_float(data, "min_improvement_delta", minimum=0.0),
        patience_generations=_require_int(data, "patience_generations", minimum=1),
        evaluation_payload=_require_object(data, "evaluation_payload"),
        candidate_payload=_require_object(data, "candidate_payload"),
        review_payload=_require_object(data, "review_payload"),
    )


def manifest_to_dict(manifest: PopulationSearchManifest) -> Dict[str, Any]:
    return asdict(manifest)


def build_generation_zero_plans(manifest: PopulationSearchManifest) -> List[CandidatePlan]:
    return [
        CandidatePlan(
            candidate_id=f"g000-c{index + 1:03d}",
            generation_index=0,
            strategy="fresh",
        )
        for index in range(manifest.population_size)
    ]


def _score_value(record: Mapping[str, Any], manifest: PopulationSearchManifest) -> float:
    raw = record.get(manifest.score_field)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    return value


def sort_candidate_results(
    records: Iterable[Mapping[str, Any]],
    manifest: PopulationSearchManifest,
) -> List[Dict[str, Any]]:
    direction = manifest.score_direction

    def _sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        correctness = 1 if bool(record.get("correctness_passed")) else 0
        score = _score_value(record, manifest)
        if direction == "maximize":
            score_key = -score
        else:
            score_key = score
        return (
            -correctness,
            score_key,
            str(record.get("candidate_id", "")),
        )

    return [dict(record) for record in sorted(records, key=_sort_key)]


def build_generation_selection(
    *,
    sorted_records: Sequence[Mapping[str, Any]],
    manifest: PopulationSearchManifest,
    generation_index: int,
) -> Dict[str, Any]:
    elites = [dict(record) for record in sorted_records[: manifest.elite_count]]
    seen_hashes = {str(record.get("code_hash", "") or "") for record in elites if str(record.get("code_hash", "") or "")}

    diversity: List[Dict[str, Any]] = []
    for record in sorted_records[manifest.elite_count :]:
        code_hash = str(record.get("code_hash", "") or "")
        if code_hash and code_hash in seen_hashes:
            continue
        diversity.append(dict(record))
        if code_hash:
            seen_hashes.add(code_hash)
        if len(diversity) >= manifest.diversity_count:
            break

    selected = elites + diversity
    next_generation = generation_index + 1
    plans: List[CandidatePlan] = []

    for index, record in enumerate(elites):
        plans.append(
            CandidatePlan(
                candidate_id=f"g{next_generation:03d}-c{index + 1:03d}",
                generation_index=next_generation,
                strategy="elite",
                parent_candidate_ids=[str(record.get("candidate_id", "") or "")],
                parent_solution_refs=[str(record.get("solution_ref", "") or "")],
            )
        )

    for index, record in enumerate(diversity, start=len(plans) + 1):
        plans.append(
            CandidatePlan(
                candidate_id=f"g{next_generation:03d}-c{index:03d}",
                generation_index=next_generation,
                strategy="diversity",
                parent_candidate_ids=[str(record.get("candidate_id", "") or "")],
                parent_solution_refs=[str(record.get("solution_ref", "") or "")],
            )
        )

    next_slot = len(plans) + 1
    for _ in range(manifest.fresh_count):
        plans.append(
            CandidatePlan(
                candidate_id=f"g{next_generation:03d}-c{next_slot:03d}",
                generation_index=next_generation,
                strategy="fresh",
            )
        )
        next_slot += 1

    parent_pool = elites or diversity or [dict(record) for record in sorted_records[:1]]
    while len(plans) < manifest.population_size:
        parent = parent_pool[(len(plans) - len(selected) - manifest.fresh_count) % len(parent_pool)]
        secondary_pool = selected or parent_pool
        secondary = secondary_pool[(len(plans)) % len(secondary_pool)]
        parent_ids = [str(parent.get("candidate_id", "") or "")]
        parent_solution_refs = [str(parent.get("solution_ref", "") or "")]
        secondary_id = str(secondary.get("candidate_id", "") or "")
        secondary_ref = str(secondary.get("solution_ref", "") or "")
        if secondary_id and secondary_id not in parent_ids:
            parent_ids.append(secondary_id)
        if secondary_ref and secondary_ref not in parent_solution_refs:
            parent_solution_refs.append(secondary_ref)
        plans.append(
            CandidatePlan(
                candidate_id=f"g{next_generation:03d}-c{next_slot:03d}",
                generation_index=next_generation,
                strategy="offspring",
                parent_candidate_ids=parent_ids,
                parent_solution_refs=parent_solution_refs,
            )
        )
        next_slot += 1

    return {
        "elite_candidate_ids": [str(record.get("candidate_id", "") or "") for record in elites],
        "diversity_candidate_ids": [str(record.get("candidate_id", "") or "") for record in diversity],
        "plans": plans,
    }


def score_improved(
    *,
    previous_best_score: Optional[float],
    current_best_score: Optional[float],
    manifest: PopulationSearchManifest,
) -> bool:
    if current_best_score is None:
        return False
    if previous_best_score is None:
        return True
    if manifest.score_direction == "maximize":
        return current_best_score >= previous_best_score + manifest.min_improvement_delta
    return current_best_score <= previous_best_score - manifest.min_improvement_delta


def build_candidate_prompt(
    manifest: PopulationSearchManifest,
    *,
    input_ref: str,
    output_ref: str,
    candidate_id: str,
    generation_index: int,
    strategy: str,
) -> str:
    return render_task_template(
        manifest.candidate_task_template,
        {
            "input_ref": input_ref,
            "output_ref": output_ref,
            "candidate_id": candidate_id,
            "generation_index": generation_index,
            "strategy": strategy,
        },
    )


def build_review_prompt(
    manifest: PopulationSearchManifest,
    *,
    input_ref: str,
    output_ref: str,
    generation_index: int,
) -> str:
    return render_task_template(
        manifest.review_task_template,
        {
            "input_ref": input_ref,
            "output_ref": output_ref,
            "generation_index": generation_index,
        },
    )
