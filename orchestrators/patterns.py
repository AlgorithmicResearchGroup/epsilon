"""Pattern registry for collaboration orchestration entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PatternSpec:
    pattern: str
    entry_script: str
    description: str


PATTERN_REGISTRY: Dict[str, PatternSpec] = {
    "dag": PatternSpec(
        pattern="dag",
        entry_script="orchestrators/dag_orchestrator.py",
        description="Dependency-ordered DAG waves with QA/fix loops.",
    ),
    "tree": PatternSpec(
        pattern="tree",
        entry_script="orchestrators/tree_orchestrator.py",
        description="Hierarchical team decomposition with branch merge and integration QA.",
    ),
    "pipeline": PatternSpec(
        pattern="pipeline",
        entry_script="orchestrators/pipeline_orchestrator.py",
        description="Linear stage-by-stage execution with QA/fix waves.",
    ),
    "supervisor": PatternSpec(
        pattern="supervisor",
        entry_script="orchestrators/supervisor_orchestrator.py",
        description="Supervisor-managed DAG with per-stage pass/retry/split decisions.",
    ),
    "work_queue": PatternSpec(
        pattern="work_queue",
        entry_script="orchestrators/work_queue_orchestrator.py",
        description="Queue-driven DAG scheduling with pull-based worker daemons.",
    ),
    "sharded_queue": PatternSpec(
        pattern="sharded_queue",
        entry_script="orchestrators/sharded_queue_orchestrator.py",
        description="Manifest-backed shard fan-out with shard-local reducers and final merge.",
    ),
    "map_reduce": PatternSpec(
        pattern="map_reduce",
        entry_script="orchestrators/map_reduce_orchestrator.py",
        description="Manifest-backed map/reduce tree with fixed-arity hierarchical aggregation.",
    ),
    "population_search": PatternSpec(
        pattern="population_search",
        entry_script="orchestrators/population_search_orchestrator.py",
        description="Iterative population search with generation scoring, briefs, and recursive improvement.",
    ),
}

DEFAULT_PATTERN = "dag"

PATTERN_ALIASES: Dict[str, str] = {
    "fanout": "dag",
    "fan-out": "dag",
    "hierarchy": "tree",
    "hierarchical": "tree",
    "linear": "pipeline",
    "staged": "pipeline",
    "managed": "supervisor",
    "adaptive": "supervisor",
    "queue": "work_queue",
    "work-queue": "work_queue",
    "pull": "work_queue",
    "pull_queue": "work_queue",
    "sharded": "sharded_queue",
    "sharded-queue": "sharded_queue",
    "mapreduce": "map_reduce",
    "map-reduce": "map_reduce",
    "population": "population_search",
    "evolution": "population_search",
    "evolutionary": "population_search",
}


class PatternConfigError(ValueError):
    """Raised when a pattern is unknown."""


def available_patterns() -> List[str]:
    return sorted(PATTERN_REGISTRY.keys())


def resolve_pattern(pattern: Optional[str] = None) -> PatternSpec:
    requested = (pattern or os.environ.get("COLLAB_PATTERN") or DEFAULT_PATTERN).strip().lower()
    requested = PATTERN_ALIASES.get(requested, requested)
    spec = PATTERN_REGISTRY.get(requested)
    if spec is None:
        valid = ", ".join(available_patterns())
        aliases = ", ".join(sorted(PATTERN_ALIASES.keys()))
        raise PatternConfigError(
            f"Unsupported pattern '{requested}'. Supported patterns: {valid}. Aliases: {aliases}"
        )
    return spec
