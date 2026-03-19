from __future__ import annotations

import json
from pathlib import Path

from examples.hf_entity_graph.local_tasks import run_task
from examples.hf_entity_graph.workflow import (
    DemoConfig,
    _validate_model_credentials,
    _select_sample_rows,
    build_adjudication_manifest,
    build_extraction_manifest,
    finalize_entity_graph,
)
from runtime.worker_daemon import WorkerDaemon


def _demo_config() -> DemoConfig:
    return DemoConfig(
        repo_id="Awesome075/multi_news_parquet",
        repo_type="dataset",
        filename="validation.parquet",
        sample_size=5,
        worker_count=2,
        reduce_arity=4,
        shard_count=2,
        max_iterations=8,
        max_runtime_seconds=90,
        agent_model="openai/gpt-5.2",
        max_source_chars=4000,
        max_ambiguities=10,
    )


def test_build_demo_manifests_include_expected_executors(tmp_path):
    config = _demo_config()
    extraction_manifest = build_extraction_manifest(
        run_dir=tmp_path,
        documents=[{"doc_id": "doc-0001", "input_ref": "/tmp/doc-0001.json"}],
        config=config,
    )
    extraction_payload = json.loads(extraction_manifest.read_text(encoding="utf-8"))
    assert extraction_payload["map_executor"] == "agent"
    assert extraction_payload["reduce_executor"] == "python_handler"
    assert extraction_payload["reduce_payload"]["operation"] == "merge_entity_candidates"

    adjudication_manifest = build_adjudication_manifest(
        run_dir=tmp_path,
        ambiguity_items=[{"ambiguity_id": "amb-0001", "input_ref": "/tmp/amb-0001.json"}],
        config=config,
    )
    adjudication_payload = json.loads(adjudication_manifest.read_text(encoding="utf-8"))
    assert adjudication_payload["map_executor"] == "agent"
    assert adjudication_payload["reduce_executor"] == "python_handler"
    assert adjudication_payload["reduce_payload"]["operation"] == "bundle_outputs"


def test_validate_model_credentials_requires_provider_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        _validate_model_credentials("openai/gpt-5.2")
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True


def test_select_sample_rows_random_is_seeded():
    rows = [
        {"doc_id": f"src-{index}", "source_text": f"text-{index}", "reference_summary": f"summary-{index}"}
        for index in range(10)
    ]

    first = _select_sample_rows(rows, sample_size=4, sample_mode="random", sample_seed=17)
    second = _select_sample_rows(rows, sample_size=4, sample_mode="random", sample_seed=17)
    third = _select_sample_rows(rows, sample_size=4, sample_mode="random", sample_seed=23)

    assert [item["source_text"] for item in first] == [item["source_text"] for item in second]
    assert [item["source_text"] for item in first] != [item["source_text"] for item in third]
    assert [item["doc_id"] for item in first] == ["doc-0001", "doc-0002", "doc-0003", "doc-0004"]


def test_merge_entity_candidates_creates_ambiguity_candidate(tmp_path):
    map_a = {
        "doc_id": "doc-0001",
        "summary": "Barack Obama spoke at the White House.",
        "keywords": ["obama", "white house"],
        "entities": [
            {
                "name": "Barack Obama",
                "entity_type": "PERSON",
                "aliases": ["President Obama"],
                "confidence": 0.92,
                "evidence": ["Barack Obama spoke at the White House."],
            },
            {
                "name": "White House",
                "entity_type": "ORG",
                "aliases": ["the White House"],
                "confidence": 0.88,
                "evidence": ["Barack Obama spoke at the White House."],
            },
        ],
        "relations": [],
    }
    map_b = {
        "doc_id": "doc-0002",
        "summary": "Obama met reporters outside the White House.",
        "keywords": ["obama", "reporters"],
        "entities": [
            {
                "name": "Obama",
                "entity_type": "PERSON",
                "aliases": ["Barack Obama"],
                "confidence": 0.84,
                "evidence": ["Obama met reporters outside the White House."],
            }
        ],
        "relations": [],
    }
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "map-a.json").write_text(json.dumps(map_a), encoding="utf-8")
    (tmp_path / "phase1" / "map-b.json").write_text(json.dumps(map_b), encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_entity_candidates",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/map-a.json", "phase1/map-b.json"],
            "max_ambiguities": 10,
        },
        task_root=tmp_path,
    )
    assert result["output_exists"] is True

    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    ambiguity_pairs = [tuple(item["cluster_keys"]) for item in payload["ambiguity_candidates"]]
    assert ("barack obama", "obama") in ambiguity_pairs


def test_merge_entity_candidates_tolerates_control_characters(tmp_path):
    malformed = """{
  "doc_id": "doc-0001",
  "summary": "test",
  "keywords": [],
  "entities": [
    {
      "name": "Anthony Weiner",
      "entity_type": "PERSON",
      "aliases": ["Weiner"],
      "confidence": 0.9,
      "evidence": ["Anthony Weiner\u0019s statement"]
    }
  ],
  "relations": []
}"""
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "bad.json").write_text(malformed, encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_entity_candidates",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/bad.json"],
            "max_ambiguities": 5,
        },
        task_root=tmp_path,
    )

    assert result["output_exists"] is True
    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    assert payload["entity_candidates"][0]["representative_name"] == "Anthony Weiner"


def test_merge_entity_candidates_repairs_embedded_quotes(tmp_path):
    malformed = """{
  "doc_id": "doc-0001",
  "summary": "test",
  "keywords": [],
  "entities": [
    {
      "name": "Knopf Doubleday",
      "entity_type": "ORG",
      "aliases": ["Knopf Doubleday"],
      "confidence": 0.9,
      "evidence": [
        "We are seeing historic, record-breaking sales ... for 'The Lost Symbol," said Sonny Mehta."
      ]
    }
  ],
  "relations": []
}"""
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "bad.json").write_text(malformed, encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_entity_candidates",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/bad.json"],
            "max_ambiguities": 5,
        },
        task_root=tmp_path,
    )

    assert result["output_exists"] is True
    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    assert payload["entity_candidates"][0]["representative_name"] == "Knopf Doubleday"


def test_python_handler_executor_bundles_outputs(tmp_path):
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.work_root = tmp_path
    (tmp_path / "phase2").mkdir()
    (tmp_path / "phase2" / "a.json").write_text(json.dumps({"ambiguity_id": "amb-1"}), encoding="utf-8")
    (tmp_path / "phase2" / "b.json").write_text(json.dumps({"ambiguity_id": "amb-2"}), encoding="utf-8")

    result = WorkerDaemon._execute_python_handler_task(
        daemon,
        "task-1",
        {
            "benchmark_id": "bench",
            "shared_workspace": str(tmp_path),
            "handler": "examples.hf_entity_graph.local_tasks:run_task",
            "operation": "bundle_outputs",
            "output_ref": "phase2/bundle.json",
            "input_refs": ["phase2/a.json", "phase2/b.json"],
        },
    )
    assert result["output_exists"] is True
    bundled = json.loads((tmp_path / "phase2" / "bundle.json").read_text(encoding="utf-8"))
    assert bundled["item_count"] == 2


def test_finalize_entity_graph_applies_merge_decisions(tmp_path):
    phase1_output = {
        "kind": "entity_graph_aggregate",
        "document_summaries": [
            {"doc_id": "doc-0001", "summary": "Obama visited the White House.", "keywords": ["obama"]},
        ],
        "entity_candidates": [
            {
                "cluster_key": "barack obama",
                "representative_name": "Barack Obama",
                "aliases": ["Barack Obama", "President Obama"],
                "entity_types": {"PERSON": 2},
                "doc_ids": ["doc-0001"],
                "mention_count": 2,
                "max_confidence": 0.92,
                "evidence": [{"doc_id": "doc-0001", "text": "Barack Obama"}],
            },
            {
                "cluster_key": "obama",
                "representative_name": "Obama",
                "aliases": ["Obama"],
                "entity_types": {"PERSON": 1},
                "doc_ids": ["doc-0001"],
                "mention_count": 1,
                "max_confidence": 0.8,
                "evidence": [{"doc_id": "doc-0001", "text": "Obama"}],
            },
            {
                "cluster_key": "white house",
                "representative_name": "White House",
                "aliases": ["White House"],
                "entity_types": {"ORG": 1},
                "doc_ids": ["doc-0001"],
                "mention_count": 1,
                "max_confidence": 0.88,
                "evidence": [{"doc_id": "doc-0001", "text": "White House"}],
            },
        ],
        "relation_candidates": [
            {
                "source_key": "obama",
                "target_key": "white house",
                "relation": "visited",
                "source_name": "Obama",
                "target_name": "White House",
                "doc_ids": ["doc-0001"],
                "mention_count": 1,
                "max_confidence": 0.7,
                "evidence": [{"doc_id": "doc-0001", "text": "Obama visited the White House."}],
            }
        ],
        "ambiguity_candidates": [{"ambiguity_id": "amb-0001"}],
    }
    phase2_output = {
        "kind": "bundle_outputs",
        "outputs": [
            {
                "ambiguity_id": "amb-0001",
                "cluster_keys": ["barack obama", "obama"],
                "decision": "merge",
                "canonical_name": "Barack Obama",
                "entity_type": "PERSON",
                "retained_aliases": ["Obama"],
                "supporting_doc_ids": ["doc-0001"],
                "supporting_evidence": ["Obama visited the White House."],
                "confidence": 0.91,
                "rationale": "Same person.",
            }
        ],
    }

    phase1_workspace = tmp_path / "phase1_workspace"
    phase2_workspace = tmp_path / "phase2_workspace"
    phase1_workspace.mkdir()
    phase2_workspace.mkdir()
    phase1_file = phase1_workspace / "final.json"
    phase2_file = phase2_workspace / "bundle.json"
    phase1_file.write_text(json.dumps(phase1_output), encoding="utf-8")
    phase2_file.write_text(json.dumps(phase2_output), encoding="utf-8")

    summary = finalize_entity_graph(
        run_dir=tmp_path,
        phase1_summary={"final_output_path": str(phase1_file)},
        phase2_summary={"final_output_path": str(phase2_file), "final_output_exists": True},
    )
    assert summary["entity_count"] == 2
    graph = json.loads(Path(summary["entity_graph_path"]).read_text(encoding="utf-8"))
    assert any(entity["canonical_name"] == "Barack Obama" for entity in graph["entities"])
    assert len(graph["relations"]) == 1
