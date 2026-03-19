#!/usr/bin/env python3
"""Unified collaboration entrypoint with explicit pattern selection."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from orchestrators.patterns import PatternConfigError, available_patterns, resolve_pattern


AUTO_PATTERN = "auto"


def _normalize_pattern_token(value: str | None) -> str:
    token = (value or "").strip().lower()
    if not token:
        return AUTO_PATTERN
    if token in {"auto", "automatic"}:
        return AUTO_PATTERN
    return token


def _extract_task_text(passthrough: list[str]) -> str:
    cleaned: list[str] = []
    skip_next = False
    option_flags_with_values = {
        "--prompts",
        "--prompt",
        "--task-manifest",
        "--shared-dir",
        "--executor",
        "--max-iterations",
        "--max-runtime-seconds",
        "--max-waves",
        "--model",
        "--orchestrator-model",
    }
    for token in passthrough:
        if skip_next:
            skip_next = False
            continue
        if token in option_flags_with_values:
            skip_next = True
            continue
        if token.startswith("--"):
            continue
        cleaned.append(token)
    if cleaned:
        return " ".join(cleaned).strip().lower()
    return " ".join(passthrough).strip().lower()


def _auto_select_pattern(task_text: str) -> tuple[str, str]:
    text = task_text.lower()

    queue_keywords = {
        "queue",
        "work queue",
        "worker daemon",
        "pull-based",
        "lease",
        "broker",
        "dispatch",
        "requeue",
    }
    if any(keyword in text for keyword in queue_keywords):
        return "work_queue", "queue-style task cues"

    supervisor_keywords = {
        "reassign",
        "adaptive",
        "supervise",
        "recover",
        "retry failure",
        "navigate failures",
        "failed stages",
        "uncertain",
    }
    if any(keyword in text for keyword in supervisor_keywords):
        return "supervisor", "adaptive recovery cues"

    tree_keywords = {
        "multi-team",
        "multiple teams",
        "parallel teams",
        "large app",
        "full-stack platform",
        "e-commerce platform",
        "split into teams",
        "branch merge",
    }
    tree_component_hits = len(re.findall(r"\b(frontend|backend|mobile|api|database|integration|admin)\b", text))
    if any(keyword in text for keyword in tree_keywords) or tree_component_hits >= 4:
        return "tree", "broad multi-subsystem decomposition cues"

    pipeline_keywords = {
        "pipeline",
        "etl",
        "extract",
        "transform",
        "load",
        "staged",
        "sequential",
        "ingest",
        "validate then",
    }
    if any(keyword in text for keyword in pipeline_keywords):
        return "pipeline", "sequential stage cues"

    return "dag", "default dependency-ordered execution"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run collaborative orchestration with a selectable coordination pattern."
    )
    parser.add_argument(
        "--pattern",
        default=AUTO_PATTERN,
        help=(
            "Collaboration pattern (e.g. dag, tree, pipeline, supervisor, work_queue, "
            "sharded_queue, map_reduce, fanout, hierarchy, auto). Defaults to auto."
        ),
    )
    parser.add_argument(
        "--task-manifest",
        help="JSON task manifest for manifest-backed large-scale patterns such as sharded_queue or map_reduce.",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List available patterns and exit.",
    )
    return parser.parse_known_args()


def main():
    args, passthrough = parse_args()
    if args.task_manifest:
        passthrough = ["--task-manifest", args.task_manifest, *passthrough]

    if args.list_patterns:
        print("Available patterns:")
        print("  - auto: Task-aware automatic selection.")
        for pattern in available_patterns():
            spec = resolve_pattern(pattern)
            print(f"  - {spec.pattern}: {spec.description} ({spec.entry_script})")
        return

    requested_pattern = _normalize_pattern_token(args.pattern)
    if requested_pattern == AUTO_PATTERN:
        task_text = _extract_task_text(passthrough)
        selected_pattern, reason = _auto_select_pattern(task_text)
        print(f"[PATTERN] Auto-selected '{selected_pattern}' ({reason})", flush=True)
    else:
        selected_pattern = requested_pattern

    try:
        spec = resolve_pattern(selected_pattern)
    except PatternConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    project_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["COLLAB_PATTERN"] = spec.pattern
    cmd = [sys.executable, spec.entry_script, *passthrough]

    print(f"[PATTERN] Using '{spec.pattern}' via {spec.entry_script}", flush=True)
    result = subprocess.run(cmd, cwd=project_dir, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
