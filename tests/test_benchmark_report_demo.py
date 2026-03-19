import json
from pathlib import Path

from examples.benchmark_report.workflow import summarize_semantic_run, write_benchmark_bundle


def test_summarize_semantic_run_counts_judged_and_grouped_records(tmp_path):
    benchmark_results = {
        "paper_count": 3,
        "record_count": 3,
        "comparable_group_count": 1,
        "records": [
            {"record_id": "r1", "comparable_group_id": "cmp-0001"},
            {"record_id": "r2", "comparable_group_id": "cmp-0001"},
            {"record_id": "r3", "comparable_group_id": ""},
        ],
        "comparable_groups": [{"group_id": "cmp-0001", "record_ids": ["r1", "r2"]}],
    }
    judgments = [
        {"ambiguity_id": "amb-0001", "record_ids": ["r1", "r2"], "decision": "comparable"},
        {"ambiguity_id": "amb-0002", "record_ids": ["r2", "r3"], "decision": "not_comparable"},
    ]
    final_dir = tmp_path / "final"
    final_dir.mkdir(parents=True)
    benchmark_results_path = final_dir / "benchmark_results.json"
    comparison_judgments_path = final_dir / "comparison_judgments.json"
    benchmark_results_path.write_text(json.dumps(benchmark_results), encoding="utf-8")
    comparison_judgments_path.write_text(json.dumps(judgments), encoding="utf-8")

    demo_summary = {
        "run_dir": str(tmp_path),
        "final_summary": {
            "benchmark_results_path": str(benchmark_results_path),
            "comparison_judgments_path": str(comparison_judgments_path),
            "paper_count": 3,
            "record_count": 3,
            "ambiguity_count": 2,
            "adjudication_count": 2,
            "comparable_group_count": 1,
            "comparable_count": 1,
            "not_comparable_count": 1,
            "uncertain_count": 0,
        },
    }
    summary_path = tmp_path / "demo_summary.json"
    summary_path.write_text(json.dumps(demo_summary), encoding="utf-8")

    row = summarize_semantic_run(summary_path=summary_path, mode="two_pass", runtime_seconds=12.5)

    assert row["judged_record_count"] == 3
    assert row["grouped_record_count"] == 2
    assert row["curation_intervention_rate"] == 1.0
    assert row["comparable_group_count"] == 1


def test_write_benchmark_bundle_writes_csv_json_report_and_charts(tmp_path):
    scale_rows = [
        {
            "topology": "work_queue",
            "worker_count": 2,
            "task_count": 120,
            "throughput_tasks_per_min": 100.0,
            "throughput_input_items_per_min": 100.0,
            "p50_latency_ms": 80,
            "p95_latency_ms": 120,
            "success_count": 120,
            "failure_count": 0,
            "missing_count": 0,
            "elapsed_seconds": 72.0,
            "estimated_cost_usd": 0.0,
            "report_path": "/tmp/work_queue-2/scale_report.json",
            "log_path": "/tmp/work_queue-2/benchmark.log",
        },
        {
            "topology": "map_reduce",
            "worker_count": 8,
            "task_count": 120,
            "throughput_tasks_per_min": 240.0,
            "throughput_input_items_per_min": 240.0,
            "p50_latency_ms": 55,
            "p95_latency_ms": 95,
            "success_count": 120,
            "failure_count": 0,
            "missing_count": 0,
            "elapsed_seconds": 30.0,
            "estimated_cost_usd": 0.0,
            "report_path": "/tmp/map_reduce-8/scale_report.json",
            "log_path": "/tmp/map_reduce-8/benchmark.log",
        },
    ]
    semantic_rows = [
        {
            "mode": "extraction_only",
            "paper_count": 40,
            "record_count": 90,
            "ambiguity_count": 0,
            "adjudication_count": 0,
            "judged_record_count": 0,
            "grouped_record_count": 0,
            "comparable_group_count": 0,
            "comparable_count": 0,
            "not_comparable_count": 0,
            "uncertain_count": 0,
            "runtime_seconds": 120.0,
            "curation_intervention_rate": 0.0,
            "selective_reasoning_ratio": 0.0,
            "run_dir": "/tmp/extraction_only",
            "summary_path": "/tmp/extraction_only/demo_summary.json",
        },
        {
            "mode": "two_pass",
            "paper_count": 40,
            "record_count": 92,
            "ambiguity_count": 12,
            "adjudication_count": 12,
            "judged_record_count": 20,
            "grouped_record_count": 4,
            "comparable_group_count": 2,
            "comparable_count": 2,
            "not_comparable_count": 10,
            "uncertain_count": 0,
            "runtime_seconds": 165.0,
            "curation_intervention_rate": 20 / 92,
            "selective_reasoning_ratio": 12 / 40,
            "run_dir": "/tmp/two_pass",
            "summary_path": "/tmp/two_pass/demo_summary.json",
        },
    ]

    artifacts = write_benchmark_bundle(
        output_dir=tmp_path,
        scale_rows=scale_rows,
        semantic_rows=semantic_rows,
        config={"scale_worker_counts": [2, 8]},
    )

    for key in (
        "bundle_path",
        "scale_metrics_path",
        "semantic_metrics_path",
        "throughput_chart_path",
        "funnel_chart_path",
        "one_pass_chart_path",
        "report_path",
    ):
        assert Path(artifacts[key]).exists()

    bundle = json.loads(Path(artifacts["bundle_path"]).read_text(encoding="utf-8"))
    assert len(bundle["scale_runs"]) == 2
    assert len(bundle["semantic_runs"]) == 2
    report_text = Path(artifacts["report_path"]).read_text(encoding="utf-8")
    assert "Fastest scale run" in report_text
    assert "Records touched by explicit judgments" in report_text
