"""Workflow helpers for the HF entity graph demo."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")

EXTRACTION_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It contains:
    - `doc_id`
    - `source_text`
    - `reference_summary`

    Produce a strict JSON file at path `{output_ref}` with exactly this top-level shape:
    {{
      "doc_id": "{payload_doc_id}",
      "summary": "1-3 sentence summary of the document cluster",
      "keywords": ["3-8 short keywords"],
      "entities": [
        {{
          "name": "entity name",
          "entity_type": "PERSON|ORG|GPE|EVENT|LAW|PRODUCT|WORK_OF_ART|OTHER",
          "aliases": ["surface forms from the document cluster"],
          "confidence": 0.0,
          "evidence": ["1-3 short verbatim snippets"]
        }}
      ],
      "relations": [
        {{
          "source": "entity name from entities",
          "target": "entity name from entities",
          "relation": "works_for|located_in|acquired|criticized|supports|owns|partnered_with|other",
          "confidence": 0.0,
          "evidence": ["1-2 short verbatim snippets"]
        }}
      ]
    }}

    Rules:
    - Use `read_file` to inspect the input JSON.
    - Write the output file with `write_file`.
    - After writing it, run `python -m json.tool {output_ref}` with `run_bash`. If validation fails, fix the file and validate again before calling `done`.
    - Keep entity extraction high precision: 3-8 salient entities only.
    - Every relation must reference entity names that appear in `entities`.
    - `confidence` must be between 0.0 and 1.0.
    - `evidence` snippets must be short and copied from the source text.
    - If an evidence snippet contains double quotes, escape them or replace them with single quotes so the JSON stays valid.
    - Do not include literal control characters in JSON string values.
    - Write valid JSON only, then call `done`.
    """
).strip()

ADJUDICATION_PROMPT = textwrap.dedent(
    """
    Read the JSON file at path `{input_ref}`.

    It contains one ambiguity candidate with two potentially matching entity clusters gathered from many documents.
    Decide whether the two clusters refer to the same real-world entity.

    Write strict JSON to `{output_ref}` with exactly this top-level shape:
    {{
      "ambiguity_id": "{payload_ambiguity_id}",
      "cluster_keys": ["left cluster key", "right cluster key"],
      "decision": "merge|keep_separate",
      "canonical_name": "best merged name if decision=merge, otherwise empty string",
      "entity_type": "PERSON|ORG|GPE|EVENT|LAW|PRODUCT|WORK_OF_ART|OTHER",
      "retained_aliases": ["aliases worth preserving"],
      "supporting_doc_ids": ["doc ids that support the decision"],
      "supporting_evidence": ["1-4 short snippets from the ambiguity file"],
      "confidence": 0.0,
      "rationale": "short explanation"
    }}

    Rules:
    - Use `read_file` to inspect the ambiguity JSON.
    - Write the output file with `write_file`.
    - After writing it, run `python -m json.tool {output_ref}` with `run_bash`. If validation fails, fix the file and validate again before calling `done`.
    - Prefer `keep_separate` unless the evidence clearly supports a merge.
    - `confidence` must be between 0.0 and 1.0.
    - Keep `supporting_evidence` short.
    - Write valid JSON only, then call `done`.
    """
).strip()


@dataclass(frozen=True)
class DemoConfig:
    repo_id: str
    repo_type: str
    filename: str
    sample_size: int
    worker_count: int
    reduce_arity: int
    shard_count: int
    max_iterations: int
    max_runtime_seconds: int
    agent_model: str
    max_source_chars: int
    max_ambiguities: int


def load_demo_config() -> DemoConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    defaults = payload["defaults"]
    return DemoConfig(
        repo_id=str(dataset["repo_id"]),
        repo_type=str(dataset["repo_type"]),
        filename=str(dataset["filename"]),
        sample_size=int(defaults["sample_size"]),
        worker_count=int(defaults["worker_count"]),
        reduce_arity=int(defaults["reduce_arity"]),
        shard_count=int(defaults["shard_count"]),
        max_iterations=int(defaults["max_iterations"]),
        max_runtime_seconds=int(defaults["max_runtime_seconds"]),
        agent_model=str(defaults["agent_model"]),
        max_source_chars=int(defaults["max_source_chars"]),
        max_ambiguities=int(defaults["max_ambiguities"]),
    )


def ensure_pyarrow() -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow"], check=True)


def _read_parquet_rows(parquet_path: str, sample_size: int) -> List[Dict[str, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["document", "summary"]).slice(0, sample_size)
    rows: List[Dict[str, str]] = []
    for index, item in enumerate(table.to_pylist(), start=1):
        rows.append(
            {
                "doc_id": f"doc-{index:04d}",
                "source_text": str(item.get("document", "") or ""),
                "reference_summary": str(item.get("summary", "") or ""),
            }
        )
    return rows


def _select_sample_rows(
    rows: List[Dict[str, Any]],
    *,
    sample_size: int,
    sample_mode: str,
    sample_seed: int,
) -> List[Dict[str, Any]]:
    if sample_size <= 0:
        return []
    bounded_size = min(sample_size, len(rows))
    normalized_mode = str(sample_mode or "first").strip().lower()
    if normalized_mode == "first":
        selected = rows[:bounded_size]
    elif normalized_mode == "random":
        chooser = random.Random(sample_seed)
        indices = chooser.sample(range(len(rows)), bounded_size)
        selected = [rows[index] for index in indices]
    else:
        raise ValueError(f"Unsupported sample mode '{sample_mode}'")

    return [
        {
            "doc_id": f"doc-{ordinal:04d}",
            "source_text": str(item.get("source_text", "") or ""),
            "reference_summary": str(item.get("reference_summary", "") or ""),
        }
        for ordinal, item in enumerate(selected, start=1)
    ]


def download_and_materialize_inputs(
    *,
    run_dir: Path,
    config: DemoConfig,
    sample_size: int,
    sample_mode: str,
    sample_seed: int,
) -> List[Dict[str, Any]]:
    ensure_pyarrow()
    parquet_path = hf_hub_download(
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        filename=config.filename,
    )
    all_rows = _read_parquet_rows(parquet_path, 1000000)
    rows = _select_sample_rows(
        all_rows,
        sample_size=sample_size,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    input_dir = run_dir / "input" / "documents"
    input_dir.mkdir(parents=True, exist_ok=True)
    documents_jsonl = run_dir / "input" / "documents.jsonl"

    materialized: List[Dict[str, Any]] = []
    with documents_jsonl.open("w", encoding="utf-8") as sink:
        for row in rows:
            normalized = {
                "doc_id": row["doc_id"],
                "source_text": row["source_text"][: config.max_source_chars].strip(),
                "reference_summary": row["reference_summary"].strip(),
            }
            doc_path = input_dir / f"{row['doc_id']}.json"
            doc_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            sink.write(json.dumps(normalized) + "\n")
            materialized.append({"doc_id": row["doc_id"], "input_ref": str(doc_path)})
    return materialized


def build_extraction_manifest(
    *,
    run_dir: Path,
    documents: Iterable[Dict[str, Any]],
    config: DemoConfig,
) -> Path:
    manifest = {
        "task_type": "hf_entity_graph_extraction",
        "output_root": "phase1",
        "items": [
            {
                "id": document["doc_id"],
                "input_ref": document["input_ref"],
                "payload": {"doc_id": document["doc_id"]},
            }
            for document in documents
        ],
        "map_task_template": EXTRACTION_PROMPT,
        "reduce_task_template": "unused",
        "reduce_arity": config.reduce_arity,
        "map_executor": "agent",
        "reduce_executor": "python_handler",
        "reduce_payload": {
            "handler": "examples.hf_entity_graph.local_tasks:run_task",
            "operation": "merge_entity_candidates",
            "max_ambiguities": config.max_ambiguities,
        },
    }
    manifest_path = run_dir / "manifests" / "extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_ambiguity_inputs(run_dir: Path, ambiguity_candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ambiguity_dir = run_dir / "input" / "ambiguities"
    ambiguity_dir.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for candidate in ambiguity_candidates:
        ambiguity_id = str(candidate["ambiguity_id"])
        path = ambiguity_dir / f"{ambiguity_id}.json"
        path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        items.append(
            {
                "ambiguity_id": ambiguity_id,
                "input_ref": str(path),
            }
        )
    return items


def build_adjudication_manifest(
    *,
    run_dir: Path,
    ambiguity_items: Iterable[Dict[str, Any]],
    config: DemoConfig,
) -> Path:
    manifest = {
        "task_type": "hf_entity_graph_adjudication",
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
            "handler": "examples.hf_entity_graph.local_tasks:run_task",
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
    summary_path = shared_dir / "run_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _validate_model_credentials(model_name: str) -> None:
    provider = str(model_name or "").split("/", 1)[0].strip().lower()
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            f"OPENAI_API_KEY is required to run the HF entity graph demo with model '{model_name}'."
        )
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            f"ANTHROPIC_API_KEY is required to run the HF entity graph demo with model '{model_name}'."
        )


def _flatten_bundle(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("kind") != "bundle_outputs":
        return [payload]
    out: List[Dict[str, Any]] = []
    for item in payload.get("outputs", []):
        if isinstance(item, dict):
            out.extend(_flatten_bundle(item))
    return out


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def finalize_entity_graph(
    *,
    run_dir: Path,
    phase1_summary: Dict[str, Any],
    phase2_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    phase1_output = Path(phase1_summary["final_output_path"])
    aggregate = json.loads(phase1_output.read_text(encoding="utf-8"))
    adjudications: List[Dict[str, Any]] = []
    if phase2_summary and phase2_summary.get("final_output_exists"):
        phase2_output = Path(phase2_summary["final_output_path"])
        adjudications = _flatten_bundle(json.loads(phase2_output.read_text(encoding="utf-8")))

    entity_candidates = aggregate.get("entity_candidates", [])
    relation_candidates = aggregate.get("relation_candidates", [])
    cluster_keys = [item["cluster_key"] for item in entity_candidates if item.get("cluster_key")]
    union_find = _UnionFind(cluster_keys)

    for adjudication in adjudications:
        if str(adjudication.get("decision", "")).strip().lower() != "merge":
            continue
        keys = [str(item).strip() for item in adjudication.get("cluster_keys", []) if str(item).strip()]
        if len(keys) != 2 or keys[0] not in union_find.parent or keys[1] not in union_find.parent:
            continue
        union_find.union(keys[0], keys[1])

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entity in entity_candidates:
        cluster_key = str(entity.get("cluster_key", "")).strip()
        if not cluster_key:
            continue
        grouped.setdefault(union_find.find(cluster_key), []).append(entity)

    merge_name_votes: Dict[str, Dict[str, int]] = {}
    merge_type_votes: Dict[str, Dict[str, int]] = {}
    merge_alias_votes: Dict[str, Dict[str, int]] = {}
    for adjudication in adjudications:
        if str(adjudication.get("decision", "")).strip().lower() != "merge":
            continue
        keys = [str(item).strip() for item in adjudication.get("cluster_keys", []) if str(item).strip()]
        if len(keys) != 2 or keys[0] not in union_find.parent or keys[1] not in union_find.parent:
            continue
        root = union_find.find(keys[0])
        canonical_name = str(adjudication.get("canonical_name", "")).strip()
        entity_type = str(adjudication.get("entity_type", "OTHER") or "OTHER").strip().upper()
        if canonical_name:
            merge_name_votes.setdefault(root, {})
            merge_name_votes[root][canonical_name] = int(merge_name_votes[root].get(canonical_name, 0)) + 1
        merge_type_votes.setdefault(root, {})
        merge_type_votes[root][entity_type] = int(merge_type_votes[root].get(entity_type, 0)) + 1
        for alias in adjudication.get("retained_aliases", []):
            text = str(alias).strip()
            if not text:
                continue
            merge_alias_votes.setdefault(root, {})
            merge_alias_votes[root][text] = int(merge_alias_votes[root].get(text, 0)) + 1

    entities: List[Dict[str, Any]] = []
    entity_id_by_root: Dict[str, str] = {}
    for index, (root, members) in enumerate(sorted(grouped.items()), start=1):
        entity_id = f"ent-{index:04d}"
        entity_id_by_root[root] = entity_id
        aliases: List[str] = []
        doc_ids: List[str] = []
        evidence: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {}
        mention_count = 0
        max_confidence = 0.0
        fallback_name = ""
        for member in members:
            aliases = list(dict.fromkeys([*aliases, *member.get("aliases", [])]))[:16]
            doc_ids = list(dict.fromkeys([*doc_ids, *member.get("doc_ids", [])]))
            evidence.extend(member.get("evidence", [])[:2])
            mention_count += int(member.get("mention_count", 0))
            max_confidence = max(max_confidence, float(member.get("max_confidence", 0.0) or 0.0))
            if len(str(member.get("representative_name", ""))) > len(fallback_name):
                fallback_name = str(member.get("representative_name", ""))
            for entity_type, count in member.get("entity_types", {}).items():
                type_counts[entity_type] = int(type_counts.get(entity_type, 0)) + int(count)

        voted_name = ""
        if root in merge_name_votes and merge_name_votes[root]:
            voted_name = sorted(merge_name_votes[root].items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0][0]
        canonical_name = voted_name or fallback_name or root

        merged_type_counts = dict(type_counts)
        for entity_type, count in merge_type_votes.get(root, {}).items():
            merged_type_counts[entity_type] = int(merged_type_counts.get(entity_type, 0)) + int(count)
        entity_type = sorted(merged_type_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if merged_type_counts else "OTHER"

        aliases = list(dict.fromkeys([canonical_name, *aliases, *merge_alias_votes.get(root, {}).keys()]))[:16]
        entities.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "aliases": aliases,
                "doc_ids": sorted(doc_ids),
                "mention_count": mention_count,
                "max_confidence": round(max_confidence, 3),
                "source_cluster_keys": sorted(member["cluster_key"] for member in members),
                "evidence": evidence[:6],
            }
        )

    relations_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for relation in relation_candidates:
        source_key = str(relation.get("source_key", "")).strip()
        target_key = str(relation.get("target_key", "")).strip()
        relation_name = str(relation.get("relation", "")).strip().lower()
        if not source_key or not target_key or not relation_name:
            continue
        if source_key not in union_find.parent or target_key not in union_find.parent:
            continue
        source_root = union_find.find(source_key)
        target_root = union_find.find(target_key)
        if source_root == target_root:
            continue
        source_id = entity_id_by_root[source_root]
        target_id = entity_id_by_root[target_root]
        key = (source_id, relation_name, target_id)
        candidate = relations_index.setdefault(
            key,
            {
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "relation": relation_name,
                "doc_ids": [],
                "mention_count": 0,
                "max_confidence": 0.0,
                "evidence": [],
            },
        )
        candidate["doc_ids"] = list(dict.fromkeys([*candidate["doc_ids"], *relation.get("doc_ids", [])]))
        candidate["mention_count"] += int(relation.get("mention_count", 0))
        candidate["max_confidence"] = max(float(candidate["max_confidence"]), float(relation.get("max_confidence", 0.0) or 0.0))
        candidate["evidence"].extend(relation.get("evidence", [])[:2])

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    document_summaries = aggregate.get("document_summaries", [])
    with (final_dir / "document_summaries.jsonl").open("w", encoding="utf-8") as sink:
        for item in document_summaries:
            sink.write(json.dumps(item) + "\n")

    entity_graph = {
        "dataset": {"repo_id": "Awesome075/multi_news_parquet", "split": "validation"},
        "entity_count": len(entities),
        "relation_count": len(relations_index),
        "entities": entities,
        "relations": sorted(relations_index.values(), key=lambda item: (item["source_entity_id"], item["relation"], item["target_entity_id"])),
    }
    (final_dir / "entity_graph.json").write_text(json.dumps(entity_graph, indent=2), encoding="utf-8")
    entity_index = {
        entity["entity_id"]: {
            "canonical_name": entity["canonical_name"],
            "entity_type": entity["entity_type"],
            "aliases": entity["aliases"],
        }
        for entity in entities
    }
    (final_dir / "entity_index.json").write_text(json.dumps(entity_index, indent=2), encoding="utf-8")

    merge_count = sum(1 for item in adjudications if str(item.get("decision", "")).strip().lower() == "merge")
    report = textwrap.dedent(
        f"""
        # HF Entity Graph Demo

        - Documents processed: {len(document_summaries)}
        - Candidate entities before adjudication: {len(entity_candidates)}
        - Candidate relations before adjudication: {len(relation_candidates)}
        - Ambiguity candidates: {len(aggregate.get('ambiguity_candidates', []))}
        - Adjudication tasks completed: {len(adjudications)}
        - Merge decisions accepted: {merge_count}
        - Final entities: {len(entities)}
        - Final relations: {len(relations_index)}

        ## Outputs

        - `final/entity_graph.json`
        - `final/entity_index.json`
        - `final/document_summaries.jsonl`
        """
    ).strip()
    (final_dir / "run_report.md").write_text(report + "\n", encoding="utf-8")

    return {
        "entity_graph_path": str(final_dir / "entity_graph.json"),
        "entity_index_path": str(final_dir / "entity_index.json"),
        "document_summaries_path": str(final_dir / "document_summaries.jsonl"),
        "run_report_path": str(final_dir / "run_report.md"),
        "entity_count": len(entities),
        "relation_count": len(relations_index),
        "ambiguity_count": len(aggregate.get("ambiguity_candidates", [])),
        "adjudication_count": len(adjudications),
        "merge_count": merge_count,
    }


def parse_args() -> argparse.Namespace:
    defaults = load_demo_config()
    parser = argparse.ArgumentParser(description="Run the HF entity graph demo")
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--worker-count", type=int, default=defaults.worker_count)
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "runs" / f"hf-entity-graph-{int(time.time() * 1000)}"))
    parser.add_argument("--agent-model", default=defaults.agent_model)
    parser.add_argument("--max-ambiguities", type=int, default=defaults.max_ambiguities)
    parser.add_argument("--reduce-arity", type=int, default=defaults.reduce_arity)
    parser.add_argument("--shard-count", type=int, default=defaults.shard_count)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--max-runtime-seconds", type=int, default=defaults.max_runtime_seconds)
    parser.add_argument("--max-source-chars", type=int, default=defaults.max_source_chars)
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = load_demo_config()
    config = DemoConfig(
        repo_id=defaults.repo_id,
        repo_type=defaults.repo_type,
        filename=defaults.filename,
        sample_size=int(args.sample_size),
        worker_count=int(args.worker_count),
        reduce_arity=int(args.reduce_arity),
        shard_count=int(args.shard_count),
        max_iterations=int(args.max_iterations),
        max_runtime_seconds=int(args.max_runtime_seconds),
        agent_model=str(args.agent_model),
        max_source_chars=int(args.max_source_chars),
        max_ambiguities=int(args.max_ambiguities),
    )

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = download_and_materialize_inputs(
        run_dir=run_dir,
        config=config,
        sample_size=config.sample_size,
        sample_mode=str(args.sample_mode),
        sample_seed=int(args.sample_seed),
    )
    extraction_manifest = build_extraction_manifest(run_dir=run_dir, documents=documents, config=config)
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

    final_summary = finalize_entity_graph(
        run_dir=run_dir,
        phase1_summary=phase1_summary,
        phase2_summary=phase2_summary,
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
