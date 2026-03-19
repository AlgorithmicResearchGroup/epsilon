"""Workflow helpers for the S2ORC benchmark scout example."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from examples.benchmark_scout.local_tasks import finalize_benchmark_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")

EXTRACTION_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It contains:
    - `paper_id`
    - `title`
    - `year`
    - `venue`
    - `external_ids`
    - `source_text`

    Produce a strict JSON file at path `{output_ref}` with exactly this top-level shape:
    {{
      "paper_id": "{payload_paper_id}",
      "title": "paper title",
      "year": 2024,
      "venue": "venue name or empty string",
      "summary": "1-3 sentence summary of the paper's benchmark/evaluation content",
      "benchmark_records": [
        {{
          "task": "task being measured",
          "dataset": "dataset or benchmark name",
          "metric": "metric name",
          "model_name": "model or system name",
          "score_text": "raw result text as written in the paper",
          "score_value": 0.0,
          "score_unit": "percent|f1|accuracy|pass@1|other|unknown",
          "evaluation_mode": "zero-shot|few-shot|fine-tuned|instruction-tuned|human-eval|other|unknown",
          "dataset_split": "test|validation|dev|train|unknown",
          "setup_notes": "short setup note that affects comparability",
          "confidence": 0.0,
          "evidence": ["1-3 short verbatim snippets"]
        }}
      ]
    }}

    Rules:
    - Use `read_file` to inspect the input JSON.
    - Write the output file with `write_file`.
    - After writing it, run `python -m json.tool {output_ref}` with `run_bash`. If validation fails, fix the file and validate again before calling `done`.
    - This workflow is only about benchmark/evaluation records. If the paper does not clearly report benchmark-style results, return an empty `benchmark_records` list.
    - Keep extraction high precision. Prefer fewer good records over many weak ones.
    - Extract at most 6 benchmark records.
    - `confidence` must be between 0.0 and 1.0.
    - `evidence` snippets must be short and copied from the source text.
    - If a snippet contains double quotes, escape them or replace them with single quotes so the JSON stays valid.
    - Do not include literal control characters in JSON string values.
    - Write valid JSON only, then call `done`.
    """
).strip()

ADJUDICATION_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It contains two benchmark result records that look similar enough that a comparison judgment is needed.
    Decide whether the two records are actually comparable.

    Write strict JSON to `{output_ref}` with exactly this top-level shape:
    {{
      "ambiguity_id": "{payload_ambiguity_id}",
      "record_ids": ["left record id", "right record id"],
      "decision": "comparable|not_comparable|uncertain",
      "canonical_task": "best shared task name if comparable, otherwise empty string",
      "canonical_dataset": "best shared dataset name if comparable, otherwise empty string",
      "canonical_metric": "best shared metric name if comparable, otherwise empty string",
      "confidence": 0.0,
      "supporting_evidence": ["1-4 short snippets from the ambiguity file"],
      "rationale": "short explanation"
    }}

    Rules:
    - Use `read_file` to inspect the ambiguity JSON.
    - Write the output file with `write_file`.
    - After writing it, run `python -m json.tool {output_ref}` with `run_bash`. If validation fails, fix the file and validate again before calling `done`.
    - Use `not_comparable` when benchmark setup, evaluation mode, split, or scope differs enough that the results should not sit in the same benchmark table without caveats.
    - Use `uncertain` if the evidence is incomplete.
    - `confidence` must be between 0.0 and 1.0.
    - Keep `supporting_evidence` short.
    - Write valid JSON only, then call `done`.
    """
).strip()


LLM_KEYWORDS: Mapping[str, int] = {
    "large language model": 8,
    "llm": 7,
    "language model": 5,
    "chatgpt": 8,
    "gpt-4": 8,
    "gpt-3": 7,
    "instruction tuning": 6,
    "in-context learning": 6,
    "few-shot": 5,
    "zero-shot": 5,
    "chain-of-thought": 6,
    "prompting": 4,
    "prompt": 2,
}

EVAL_KEYWORDS: Mapping[str, int] = {
    "benchmark": 6,
    "leaderboard": 6,
    "mmlu": 8,
    "truthfulqa": 8,
    "gsm8k": 8,
    "hellaswag": 8,
    "evaluation harness": 7,
    "eval": 2,
    "evaluation": 2,
}

TEXT_LLM_KEYWORDS: Mapping[str, int] = {
    "large language model": 9,
    "llm": 8,
    "language model": 6,
    "chatgpt": 8,
    "gpt-4": 8,
    "gpt-3": 7,
    "instruction tuning": 6,
    "in-context learning": 6,
    "few-shot": 5,
    "zero-shot": 5,
    "chain-of-thought": 6,
    "reasoning": 4,
    "chatbot": 5,
}

TEXT_BENCHMARK_KEYWORDS: Mapping[str, int] = {
    "benchmark": 6,
    "leaderboard": 6,
    "evaluation harness": 7,
    "mmlu": 9,
    "truthfulqa": 9,
    "gsm8k": 9,
    "hellaswag": 9,
    "humaneval": 9,
    "human eval": 8,
    "mbpp": 8,
    "bbh": 8,
    "big-bench": 8,
    "ifeval": 8,
    "winogrande": 8,
    "arc challenge": 8,
    "math benchmark": 6,
}

SECTION_PRIORITY: Mapping[str, int] = {
    "benchmark": 7,
    "evaluation": 7,
    "results": 6,
    "experiments": 6,
    "experimental setup": 5,
    "analysis": 4,
    "method": 2,
    "introduction": 1,
}

KEYWORD_PROFILES: Mapping[str, Dict[str, Any]] = {
    "llm_eval": {
        "llm": LLM_KEYWORDS,
        "eval": EVAL_KEYWORDS,
        "min_llm_score": 5,
        "min_eval_score": 2,
    },
    "text_llm_eval": {
        "llm": TEXT_LLM_KEYWORDS,
        "eval": TEXT_BENCHMARK_KEYWORDS,
        "min_llm_score": 7,
        "min_eval_score": 6,
    },
}


@dataclass(frozen=True)
class DemoConfig:
    corpus_root: str
    keyword_profile: str
    sample_size: int
    worker_count: int
    reduce_arity: int
    shard_count: int
    max_iterations: int
    max_runtime_seconds: int
    agent_model: str
    max_source_chars: int
    max_ambiguities: int
    min_year: int
    candidate_pool_size: int
    max_scan_files: int


def load_demo_config() -> DemoConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    defaults = payload["defaults"]
    return DemoConfig(
        corpus_root=str(dataset["corpus_root"]),
        keyword_profile=str(dataset["keyword_profile"]),
        sample_size=int(defaults["sample_size"]),
        worker_count=int(defaults["worker_count"]),
        reduce_arity=int(defaults["reduce_arity"]),
        shard_count=int(defaults["shard_count"]),
        max_iterations=int(defaults["max_iterations"]),
        max_runtime_seconds=int(defaults["max_runtime_seconds"]),
        agent_model=str(defaults["agent_model"]),
        max_source_chars=int(defaults["max_source_chars"]),
        max_ambiguities=int(defaults["max_ambiguities"]),
        min_year=int(defaults["min_year"]),
        candidate_pool_size=int(defaults["candidate_pool_size"]),
        max_scan_files=int(defaults["max_scan_files"]),
    )


def ensure_pyarrow() -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow"], check=True)


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\x00", " ")
    return " ".join(text.split())


def _score_keyword_map(text: str, weights: Mapping[str, int]) -> tuple[int, List[str]]:
    score = 0
    hits: List[str] = []
    lowered = text.casefold()
    for term, weight in weights.items():
        if term in lowered:
            score += weight
            hits.append(term)
    return score, hits


def _score_candidate_paper(
    *,
    title: str,
    abstract: str,
    year: int,
    min_year: int,
    keyword_profile: str,
) -> tuple[int, List[str], List[str]]:
    if year and year < min_year:
        return 0, [], []
    text = f"{title}\n{abstract}"
    profile = KEYWORD_PROFILES.get(keyword_profile)
    if profile is None:
        raise ValueError(f"Unsupported keyword profile '{keyword_profile}'")
    llm_score, llm_hits = _score_keyword_map(text, profile["llm"])
    eval_score, eval_hits = _score_keyword_map(text, profile["eval"])
    if llm_score < int(profile["min_llm_score"]) or eval_score < int(profile["min_eval_score"]):
        return 0, llm_hits, eval_hits
    recency_bonus = max(0, min(year - min_year, 6)) if year else 0
    return llm_score + eval_score + recency_bonus, llm_hits, eval_hits


def _select_candidates(
    *,
    corpus_root: Path,
    sample_size: int,
    sample_mode: str,
    sample_seed: int,
    min_year: int,
    keyword_profile: str,
    candidate_pool_size: int,
    max_scan_files: int,
) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_files = sorted(corpus_root.glob("*.parquet"))
    if max_scan_files > 0:
        parquet_files = parquet_files[:max_scan_files]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {corpus_root}")

    candidates: List[Dict[str, Any]] = []
    for file_index, parquet_path in enumerate(parquet_files):
        table = pq.read_table(
            parquet_path,
            columns=["corpus_id", "parsed_title", "metadata_title", "abstract", "year", "venue"],
        )
        for row_index, row in enumerate(table.to_pylist()):
            title = _clean_text(row.get("parsed_title") or row.get("metadata_title"))
            abstract = _clean_text(row.get("abstract"))
            year = int(row.get("year", 0) or 0)
            score, llm_hits, eval_hits = _score_candidate_paper(
                title=title,
                abstract=abstract,
                year=year,
                min_year=min_year,
                keyword_profile=keyword_profile,
            )
            if score <= 0 or not title:
                continue
            candidates.append(
                {
                    "file_path": str(parquet_path),
                    "row_index": row_index,
                    "corpus_id": str(row.get("corpus_id") or ""),
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "venue": _clean_text(row.get("venue")),
                    "score": score,
                    "llm_hits": llm_hits,
                    "eval_hits": eval_hits,
                    "file_ordinal": file_index,
                }
            )
            if len(candidates) >= candidate_pool_size:
                break
        if len(candidates) >= candidate_pool_size:
            break

    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Only found {len(candidates)} candidate papers under {corpus_root}; need at least {sample_size}."
        )

    ranked = sorted(
        candidates,
        key=lambda item: (-int(item["score"]), -int(item["year"]), item["file_ordinal"], int(item["row_index"])),
    )
    if sample_mode == "first":
        selected = ranked[:sample_size]
    elif sample_mode == "random":
        chooser = random.Random(sample_seed)
        selected = chooser.sample(ranked, sample_size)
        selected = sorted(selected, key=lambda item: (item["file_ordinal"], int(item["row_index"])))
    else:
        raise ValueError(f"Unsupported sample mode '{sample_mode}'")
    return selected


def _section_score(title: str, content: str) -> int:
    normalized_title = _clean_text(title).casefold()
    normalized_content = _clean_text(content).casefold()
    score = 0
    for token, weight in SECTION_PRIORITY.items():
        if token in normalized_title:
            score += weight
        elif token in normalized_content[:800]:
            score += max(1, weight - 2)
    return score


def _build_source_text(row: Mapping[str, Any], *, max_chars: int) -> str:
    parts: List[str] = []
    title = _clean_text(row.get("parsed_title") or row.get("metadata_title"))
    abstract = _clean_text(row.get("abstract"))
    year = int(row.get("year", 0) or 0)
    venue = _clean_text(row.get("venue"))
    if title:
        parts.append(f"Title: {title}")
    if year:
        parts.append(f"Year: {year}")
    if venue:
        parts.append(f"Venue: {venue}")
    if abstract:
        parts.append(f"Abstract:\n{abstract}")

    sections = row.get("sections") or []
    ranked_sections: List[tuple[int, int, str, str]] = []
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_title = _clean_text(section.get("title"))
        content = _clean_text(section.get("content"))
        if not content:
            continue
        ranked_sections.append((_section_score(section_title, content), index, section_title, content))
    ranked_sections.sort(key=lambda item: (-item[0], item[1]))
    if not ranked_sections and _clean_text(row.get("text")):
        ranked_sections.append((1, 0, "Paper Text", _clean_text(row.get("text"))))

    remaining = max_chars - sum(len(part) for part in parts) - (2 * len(parts))
    for score, _index, section_title, content in ranked_sections[:6]:
        if remaining <= 0:
            break
        excerpt = content[: min(2200, remaining)]
        if not excerpt:
            continue
        heading = section_title or f"Section (score={score})"
        block = f"{heading}:\n{excerpt}"
        parts.append(block)
        remaining = max_chars - sum(len(part) for part in parts) - (2 * len(parts))

    return "\n\n".join(parts)[:max_chars].strip()


def materialize_local_inputs(
    *,
    run_dir: Path,
    config: DemoConfig,
    sample_size: int,
    sample_mode: str,
    sample_seed: int,
) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    corpus_root = Path(config.corpus_root).expanduser().resolve()
    candidates = _select_candidates(
        corpus_root=corpus_root,
        sample_size=sample_size,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        min_year=config.min_year,
        keyword_profile=config.keyword_profile,
        candidate_pool_size=max(sample_size, config.candidate_pool_size),
        max_scan_files=config.max_scan_files,
    )
    candidate_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate_by_file[str(candidate["file_path"])].append(candidate)

    materialized_rows: List[Dict[str, Any]] = []
    for file_path, entries in candidate_by_file.items():
        table = pq.read_table(
            file_path,
            columns=[
                "corpus_id",
                "parsed_title",
                "metadata_title",
                "abstract",
                "sections",
                "text",
                "year",
                "venue",
                "DOI",
                "ArxIv",
                "url",
            ],
        )
        rows = table.to_pylist()
        for entry in entries:
            row = rows[int(entry["row_index"])]
            materialized_rows.append(
                {
                    "paper_id": str(row.get("corpus_id") or entry["corpus_id"]),
                    "title": _clean_text(row.get("parsed_title") or row.get("metadata_title") or entry["title"]),
                    "abstract": _clean_text(row.get("abstract") or entry["abstract"]),
                    "year": int(row.get("year", 0) or entry["year"] or 0),
                    "venue": _clean_text(row.get("venue") or entry["venue"]),
                    "external_ids": {
                        "doi": _clean_text(row.get("DOI")),
                        "arxiv": _clean_text(row.get("ArxIv")),
                        "url": _clean_text(row.get("url")),
                    },
                    "source_text": _build_source_text(row, max_chars=config.max_source_chars),
                    "slice_score": int(entry["score"]),
                    "slice_hits": {
                        "llm": entry["llm_hits"],
                        "eval": entry["eval_hits"],
                    },
                }
            )

    materialized_rows.sort(key=lambda item: (item["year"], item["paper_id"]))
    input_dir = run_dir / "input" / "papers"
    input_dir.mkdir(parents=True, exist_ok=True)
    papers_jsonl = run_dir / "input" / "papers.jsonl"
    materialized_inputs: List[Dict[str, Any]] = []
    with papers_jsonl.open("w", encoding="utf-8") as sink:
        for ordinal, row in enumerate(materialized_rows, start=1):
            doc_id = f"paper-{ordinal:04d}"
            paper_payload = {
                "doc_id": doc_id,
                **row,
            }
            path = input_dir / f"{doc_id}.json"
            path.write_text(json.dumps(paper_payload, indent=2), encoding="utf-8")
            sink.write(json.dumps(paper_payload) + "\n")
            materialized_inputs.append(
                {
                    "doc_id": doc_id,
                    "paper_id": row["paper_id"],
                    "input_ref": str(path),
                }
            )
    return materialized_inputs


def build_extraction_manifest(
    *,
    run_dir: Path,
    papers: Iterable[Dict[str, Any]],
    config: DemoConfig,
) -> Path:
    manifest = {
        "task_type": "benchmark_scout_extraction",
        "output_root": "phase1",
        "items": [
            {
                "id": paper["doc_id"],
                "input_ref": paper["input_ref"],
                "payload": {
                    "doc_id": paper["doc_id"],
                    "paper_id": paper["paper_id"],
                },
            }
            for paper in papers
        ],
        "map_task_template": EXTRACTION_PROMPT,
        "reduce_task_template": "unused",
        "reduce_arity": config.reduce_arity,
        "map_executor": "agent",
        "reduce_executor": "python_handler",
        "reduce_payload": {
            "handler": "examples.benchmark_scout.local_tasks:run_task",
            "operation": "merge_benchmark_records",
            "max_ambiguities": config.max_ambiguities,
        },
    }
    manifest_path = run_dir / "manifests" / "extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_ambiguity_inputs(run_dir: Path, ambiguity_candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ambiguity_dir = run_dir / "input" / "comparison_candidates"
    ambiguity_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for candidate in ambiguity_candidates:
        ambiguity_id = str(candidate["ambiguity_id"])
        path = ambiguity_dir / f"{ambiguity_id}.json"
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        items.append({"ambiguity_id": ambiguity_id, "input_ref": str(path)})
    return items


def build_adjudication_manifest(
    *,
    run_dir: Path,
    ambiguity_items: Iterable[Dict[str, Any]],
    config: DemoConfig,
) -> Path:
    manifest = {
        "task_type": "benchmark_scout_adjudication",
        "output_root": "phase2",
        "items": [
            {
                "id": item["ambiguity_id"],
                "input_ref": item["input_ref"],
                "payload": {"ambiguity_id": item["ambiguity_id"]},
            }
            for item in ambiguity_items
        ],
        "map_task_template": ADJUDICATION_PROMPT,
        "reduce_task_template": "unused",
        "shard_count": config.shard_count,
        "map_executor": "agent",
        "reduce_executor": "python_handler",
        "reduce_payload": {
            "handler": "examples.benchmark_scout.local_tasks:run_task",
            "operation": "bundle_outputs",
        },
    }
    manifest_path = run_dir / "manifests" / "adjudication_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def run_topology(
    *,
    pattern: str,
    manifest_path: Path,
    shared_dir: Path,
    config: DemoConfig,
    worker_count: int,
) -> Dict[str, Any]:
    _validate_model_credentials(config.agent_model)
    env = os.environ.copy()
    env["MAX_WAVES"] = "0"
    env["WORK_QUEUE_WORKERS"] = str(worker_count)
    env["WORK_QUEUE_MAX_CONCURRENT_LOCAL"] = "1"
    env["MAX_ITERATIONS"] = str(config.max_iterations)
    env["MAX_RUNTIME_SECONDS"] = str(config.max_runtime_seconds)
    env["AGENT_MODEL"] = config.agent_model

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "orchestrate.py"),
        "--pattern",
        pattern,
        "--task-manifest",
        str(manifest_path),
        "--shared-dir",
        str(shared_dir),
        pattern,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    return json.loads((shared_dir / "run_summary.json").read_text(encoding="utf-8"))


def _validate_model_credentials(model_name: str) -> None:
    provider = str(model_name or "").split("/", 1)[0].strip().lower()
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is required to run the benchmark scout demo with model '{model_name}'."
        )
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"ANTHROPIC_API_KEY is required to run the benchmark scout demo with model '{model_name}'."
        )


def parse_args() -> argparse.Namespace:
    defaults = load_demo_config()
    parser = argparse.ArgumentParser(description="Run the S2ORC benchmark scout demo")
    parser.add_argument("--corpus-root", default=defaults.corpus_root)
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--keyword-profile", choices=sorted(KEYWORD_PROFILES), default=defaults.keyword_profile)
    parser.add_argument("--worker-count", type=int, default=defaults.worker_count)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "runs" / f"benchmark-scout-{int(time.time() * 1000)}"))
    parser.add_argument("--agent-model", default=defaults.agent_model)
    parser.add_argument("--max-ambiguities", type=int, default=defaults.max_ambiguities)
    parser.add_argument("--reduce-arity", type=int, default=defaults.reduce_arity)
    parser.add_argument("--shard-count", type=int, default=defaults.shard_count)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--max-runtime-seconds", type=int, default=defaults.max_runtime_seconds)
    parser.add_argument("--max-source-chars", type=int, default=defaults.max_source_chars)
    parser.add_argument("--min-year", type=int, default=defaults.min_year)
    parser.add_argument("--candidate-pool-size", type=int, default=defaults.candidate_pool_size)
    parser.add_argument("--max-scan-files", type=int, default=defaults.max_scan_files)
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = load_demo_config()
    config = DemoConfig(
        corpus_root=str(args.corpus_root),
        keyword_profile=str(args.keyword_profile),
        sample_size=int(args.sample_size),
        worker_count=int(args.worker_count),
        reduce_arity=int(args.reduce_arity),
        shard_count=int(args.shard_count),
        max_iterations=int(args.max_iterations),
        max_runtime_seconds=int(args.max_runtime_seconds),
        agent_model=str(args.agent_model),
        max_source_chars=int(args.max_source_chars),
        max_ambiguities=int(args.max_ambiguities),
        min_year=int(args.min_year),
        candidate_pool_size=int(args.candidate_pool_size),
        max_scan_files=int(args.max_scan_files),
    )

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_pyarrow()

    papers = materialize_local_inputs(
        run_dir=run_dir,
        config=config,
        sample_size=config.sample_size,
        sample_mode=str(args.sample_mode),
        sample_seed=int(args.sample_seed),
    )
    extraction_manifest = build_extraction_manifest(run_dir=run_dir, papers=papers, config=config)
    phase1_workspace = run_dir / "phase1_workspace"
    phase1_summary = run_topology(
        pattern="map_reduce",
        manifest_path=extraction_manifest,
        shared_dir=phase1_workspace,
        config=config,
        worker_count=config.worker_count,
    )

    phase1_output = json.loads(Path(phase1_summary["final_output_path"]).read_text(encoding="utf-8"))
    ambiguity_items = write_ambiguity_inputs(run_dir, phase1_output.get("ambiguity_candidates", []))
    phase2_summary = None
    if ambiguity_items:
        adjudication_manifest = build_adjudication_manifest(
            run_dir=run_dir,
            ambiguity_items=ambiguity_items,
            config=config,
        )
        phase2_workspace = run_dir / "phase2_workspace"
        phase2_summary = run_topology(
            pattern="sharded_queue",
            manifest_path=adjudication_manifest,
            shared_dir=phase2_workspace,
            config=config,
            worker_count=config.worker_count,
        )

    final_summary = finalize_benchmark_results(
        run_dir=run_dir,
        phase1_summary=phase1_summary,
        phase2_summary=phase2_summary,
        corpus_root=config.corpus_root,
        keyword_profile=config.keyword_profile,
    )
    output = {
        "run_dir": str(run_dir),
        "sample_mode": str(args.sample_mode),
        "sample_seed": int(args.sample_seed),
        "phase1_summary": phase1_summary,
        "phase2_summary": phase2_summary,
        "final_summary": final_summary,
    }
    (run_dir / "demo_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
