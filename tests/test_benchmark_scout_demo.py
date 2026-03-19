from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.benchmark_scout.local_tasks import run_task
from examples.benchmark_scout.workflow import (
    DemoConfig,
    _select_candidates,
    _validate_model_credentials,
    build_adjudication_manifest,
    build_extraction_manifest,
    finalize_benchmark_results,
    materialize_local_inputs,
)
from runtime.worker_daemon import WorkerDaemon


def _demo_config(corpus_root: Path) -> DemoConfig:
    return DemoConfig(
        corpus_root=str(corpus_root),
        keyword_profile="text_llm_eval",
        sample_size=3,
        worker_count=2,
        reduce_arity=4,
        shard_count=2,
        max_iterations=8,
        max_runtime_seconds=90,
        agent_model="openai/gpt-5.2",
        max_source_chars=4000,
        max_ambiguities=8,
        min_year=2022,
        candidate_pool_size=8,
        max_scan_files=4,
    )


def _write_test_parquet(root: Path) -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pyarrow.table(
        {
            "corpus_id": ["p-001", "p-002", "p-003", "p-004"],
            "parsed_title": [
                "Benchmarking GPT-4 on MMLU",
                "Prompting Large Language Models for GSM8K Evaluation",
                "A systems paper with no evals",
                "Instruction tuning for TruthfulQA benchmark analysis",
            ],
            "metadata_title": ["", "", "", ""],
            "abstract": [
                "We evaluate GPT-4 on the MMLU benchmark and compare zero-shot and few-shot settings.",
                "This paper studies large language model prompting and reports GSM8K evaluation results.",
                "This paper is about storage systems and does not discuss benchmark evaluation.",
                "We study instruction tuning and report TruthfulQA benchmark performance for LLMs.",
            ],
            "sections": [
                [
                    {"title": "Results", "content": "MMLU test accuracy for GPT-4 reaches 86.4 in zero-shot mode."}
                ],
                [
                    {
                        "title": "Evaluation",
                        "content": "We report GSM8K pass@1 and compare few-shot prompting against zero-shot.",
                    }
                ],
                [{"title": "Method", "content": "A storage engine."}],
                [
                    {
                        "title": "Experiments",
                        "content": "TruthfulQA benchmark accuracy improves under instruction tuning.",
                    }
                ],
            ],
            "text": ["", "", "", ""],
            "year": [2024, 2023, 2024, 2022],
            "venue": ["arXiv", "ACL", "SOSP", "NeurIPS"],
            "DOI": ["", "", "", ""],
            "ArxIv": ["2401.00001", "2301.00002", "", "2201.00003"],
            "url": ["", "", "", ""],
        }
    )
    path = root / "slice-0000.parquet"
    pq.write_table(table, path)
    return path


def test_build_demo_manifests_include_expected_executors(tmp_path):
    config = _demo_config(tmp_path)
    extraction_manifest = build_extraction_manifest(
        run_dir=tmp_path,
        papers=[{"doc_id": "paper-0001", "paper_id": "p-001", "input_ref": "/tmp/paper-0001.json"}],
        config=config,
    )
    extraction_payload = json.loads(extraction_manifest.read_text(encoding="utf-8"))
    assert extraction_payload["map_executor"] == "agent"
    assert extraction_payload["reduce_executor"] == "python_handler"
    assert extraction_payload["reduce_payload"]["operation"] == "merge_benchmark_records"

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
    with pytest.raises(RuntimeError):
        _validate_model_credentials("openai/gpt-5.2")


def test_select_candidates_random_is_seeded(tmp_path):
    _write_test_parquet(tmp_path)

    first = _select_candidates(
        corpus_root=tmp_path,
        sample_size=2,
        sample_mode="random",
        sample_seed=17,
        min_year=2022,
        keyword_profile="text_llm_eval",
        candidate_pool_size=4,
        max_scan_files=2,
    )
    second = _select_candidates(
        corpus_root=tmp_path,
        sample_size=2,
        sample_mode="random",
        sample_seed=17,
        min_year=2022,
        keyword_profile="text_llm_eval",
        candidate_pool_size=4,
        max_scan_files=2,
    )
    third = _select_candidates(
        corpus_root=tmp_path,
        sample_size=2,
        sample_mode="random",
        sample_seed=23,
        min_year=2022,
        keyword_profile="text_llm_eval",
        candidate_pool_size=4,
        max_scan_files=2,
    )

    assert [item["corpus_id"] for item in first] == [item["corpus_id"] for item in second]
    assert [item["corpus_id"] for item in first] != [item["corpus_id"] for item in third]


def test_materialize_local_inputs_writes_normalized_papers(tmp_path):
    _write_test_parquet(tmp_path)
    config = _demo_config(tmp_path)

    papers = materialize_local_inputs(
        run_dir=tmp_path / "run",
        config=config,
        sample_size=2,
        sample_mode="first",
        sample_seed=17,
    )

    assert len(papers) == 2
    first_input = Path(papers[0]["input_ref"])
    payload = json.loads(first_input.read_text(encoding="utf-8"))
    assert payload["doc_id"] == "paper-0001"
    assert payload["paper_id"]
    assert "Title:" in payload["source_text"]
    assert "Abstract:" in payload["source_text"]


def test_merge_benchmark_records_creates_ambiguity_candidate(tmp_path):
    map_a = {
        "paper_id": "p-001",
        "title": "Paper A",
        "year": 2024,
        "venue": "arXiv",
        "summary": "MMLU benchmark results for GPT-4.",
        "benchmark_records": [
            {
                "task": "Question Answering benchmark performance",
                "dataset": "MMLU benchmark",
                "metric": "acc",
                "model_name": "GPT-4",
                "score_text": "86.4 accuracy on MMLU test",
                "score_value": 86.4,
                "score_unit": "percent",
                "evaluation_mode": "zero-shot",
                "dataset_split": "test",
                "setup_notes": "standard prompt",
                "confidence": 0.94,
                "evidence": ["GPT-4 reaches 86.4 accuracy on MMLU."],
            }
        ],
    }
    map_b = {
        "paper_id": "p-002",
        "title": "Paper B",
        "year": 2023,
        "venue": "ACL",
        "summary": "MMLU few-shot result.",
        "benchmark_records": [
            {
                "task": "Question Answering",
                "dataset": "MMLU",
                "metric": "accuracy",
                "model_name": "GPT 4",
                "score_text": "88.1 accuracy in few-shot evaluation",
                "score_value": 88.1,
                "score_unit": "percent",
                "evaluation_mode": "few-shot",
                "dataset_split": "test",
                "setup_notes": "5-shot prompt",
                "confidence": 0.9,
                "evidence": ["We report 88.1 accuracy on MMLU with five-shot prompts."],
            }
        ],
    }
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "map-a.json").write_text(json.dumps(map_a), encoding="utf-8")
    (tmp_path / "phase1" / "map-b.json").write_text(json.dumps(map_b), encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_benchmark_records",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/map-a.json", "phase1/map-b.json"],
            "max_ambiguities": 10,
        },
        task_root=tmp_path,
    )
    assert result["output_exists"] is True

    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    assert len(payload["benchmark_records"]) == 2
    assert payload["ambiguity_candidates"][0]["left_record"]["dataset"] == "MMLU benchmark"
    assert "metric_match" in payload["ambiguity_candidates"][0]["reason"]
    assert payload["ambiguity_candidates"][0]["group_key"].startswith("fuzzy:")


def test_merge_benchmark_records_ignores_cross_dataset_metric_only_matches(tmp_path):
    map_a = {
        "paper_id": "p-001",
        "title": "Paper A",
        "year": 2024,
        "venue": "arXiv",
        "summary": "MATH benchmark result.",
        "benchmark_records": [
            {
                "task": "Mathematical reasoning",
                "dataset": "MATH",
                "metric": "accuracy",
                "model_name": "GPT-4",
                "score_text": "37.7 accuracy on MATH",
                "score_value": 37.7,
                "score_unit": "percent",
                "evaluation_mode": "fine-tuned",
                "dataset_split": "test",
                "setup_notes": "fine-tuned on synthetic math data",
                "confidence": 0.95,
                "evidence": ["GPT-4 reaches 37.7 accuracy on MATH."],
            }
        ],
    }
    map_b = {
        "paper_id": "p-002",
        "title": "Paper B",
        "year": 2024,
        "venue": "ACL",
        "summary": "HistBench result.",
        "benchmark_records": [
            {
                "task": "Historical reasoning question answering",
                "dataset": "HistBench",
                "metric": "accuracy",
                "model_name": "Claude 3",
                "score_text": "52.0 accuracy on HistBench",
                "score_value": 52.0,
                "score_unit": "percent",
                "evaluation_mode": "zero-shot",
                "dataset_split": "test",
                "setup_notes": "LLM judge with historian validation",
                "confidence": 0.92,
                "evidence": ["Claude 3 achieves 52.0 accuracy on HistBench."],
            }
        ],
    }
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "map-a.json").write_text(json.dumps(map_a), encoding="utf-8")
    (tmp_path / "phase1" / "map-b.json").write_text(json.dumps(map_b), encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_benchmark_records",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/map-a.json", "phase1/map-b.json"],
            "max_ambiguities": 10,
        },
        task_root=tmp_path,
    )
    assert result["output_exists"] is True

    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    assert payload["ambiguity_candidates"] == []


def test_merge_benchmark_records_tolerates_embedded_quotes(tmp_path):
    malformed = """{
  "paper_id": "p-001",
  "title": "Paper A",
  "year": 2024,
  "venue": "arXiv",
  "summary": "test",
  "benchmark_records": [
    {
      "task": "Question Answering",
      "dataset": "MMLU",
      "metric": "accuracy",
      "model_name": "GPT-4",
      "score_text": "The paper reports "state of the art" accuracy.",
      "score_value": 86.4,
      "score_unit": "percent",
      "evaluation_mode": "zero-shot",
      "dataset_split": "test",
      "setup_notes": "",
      "confidence": 0.9,
      "evidence": ["The paper reports "state of the art" accuracy."]
    }
  ]
}"""
    (tmp_path / "phase1").mkdir()
    (tmp_path / "phase1" / "bad.json").write_text(malformed, encoding="utf-8")

    result = run_task(
        task_id="reduce-1",
        task_payload={
            "operation": "merge_benchmark_records",
            "output_ref": "phase1/final.json",
            "input_refs": ["phase1/bad.json"],
            "max_ambiguities": 5,
        },
        task_root=tmp_path,
    )

    assert result["output_exists"] is True
    payload = json.loads((tmp_path / "phase1" / "final.json").read_text(encoding="utf-8"))
    assert payload["benchmark_records"][0]["model_name"] == "GPT-4"


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
            "handler": "examples.benchmark_scout.local_tasks:run_task",
            "operation": "bundle_outputs",
            "output_ref": "phase2/bundle.json",
            "input_refs": ["phase2/a.json", "phase2/b.json"],
        },
    )
    assert result["output_exists"] is True
    bundled = json.loads((tmp_path / "phase2" / "bundle.json").read_text(encoding="utf-8"))
    assert bundled["item_count"] == 2


def test_finalize_benchmark_results_applies_comparable_decisions(tmp_path):
    phase1_output = {
        "kind": "benchmark_aggregate",
        "paper_summaries": [
            {"paper_id": "p-001", "title": "Paper A", "year": 2024, "venue": "arXiv", "summary": "A", "record_count": 1},
            {"paper_id": "p-002", "title": "Paper B", "year": 2023, "venue": "ACL", "summary": "B", "record_count": 1},
        ],
        "benchmark_records": [
            {
                "record_id": "p-001-rec-001",
                "paper_id": "p-001",
                "paper_title": "Paper A",
                "paper_year": 2024,
                "paper_venue": "arXiv",
                "task": "Question Answering",
                "dataset": "MMLU",
                "metric": "accuracy",
                "model_name": "GPT-4",
                "score_text": "86.4",
                "score_value": 86.4,
                "score_unit": "percent",
                "evaluation_mode": "zero-shot",
                "dataset_split": "test",
                "setup_notes": "standard",
                "confidence": 0.94,
                "evidence": [],
                "group_key": "question answering|mmlu|accuracy",
            },
            {
                "record_id": "p-002-rec-001",
                "paper_id": "p-002",
                "paper_title": "Paper B",
                "paper_year": 2023,
                "paper_venue": "ACL",
                "task": "QA",
                "dataset": "MMLU benchmark",
                "metric": "acc",
                "model_name": "GPT 4",
                "score_text": "88.1",
                "score_value": 88.1,
                "score_unit": "percent",
                "evaluation_mode": "zero-shot",
                "dataset_split": "test",
                "setup_notes": "same setup",
                "confidence": 0.9,
                "evidence": [],
                "group_key": "qa|mmlu benchmark|acc",
            },
        ],
        "ambiguity_candidates": [{"ambiguity_id": "amb-0001"}],
    }
    phase2_output = {
        "kind": "bundle_outputs",
        "outputs": [
            {
                "ambiguity_id": "amb-0001",
                "record_ids": ["p-001-rec-001", "p-002-rec-001"],
                "decision": "comparable",
                "canonical_task": "Question Answering",
                "canonical_dataset": "MMLU",
                "canonical_metric": "accuracy",
                "confidence": 0.9,
                "supporting_evidence": ["Same benchmark setup."],
                "rationale": "Equivalent evaluation framing.",
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

    summary = finalize_benchmark_results(
        run_dir=tmp_path,
        phase1_summary={"final_output_path": str(phase1_file)},
        phase2_summary={"final_output_path": str(phase2_file), "final_output_exists": True},
        corpus_root="/tmp/corpus",
        keyword_profile="text_llm_eval",
    )

    assert summary["record_count"] == 2
    assert summary["comparable_group_count"] == 1
    benchmark_results = json.loads(Path(summary["benchmark_results_path"]).read_text(encoding="utf-8"))
    assert benchmark_results["comparable_groups"][0]["canonical_dataset"] == "MMLU"
    assert all(record["comparable_group_id"] == "cmp-0001" for record in benchmark_results["records"])
