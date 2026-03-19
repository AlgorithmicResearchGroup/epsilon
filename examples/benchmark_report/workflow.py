"""Standalone benchmark/reporting workflow for Epsilon demos."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkerProcess:
    process: subprocess.Popen[str]
    log_handle: Any


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _split_csv_ints(value: str) -> List[int]:
    values = [int(item) for item in _split_csv(value)]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _run_logged_subprocess(
    *,
    cmd: Sequence[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> float:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as sink:
        subprocess.run(
            list(cmd),
            cwd=str(cwd or PROJECT_ROOT),
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    return max(0.0, time.time() - started)


def _start_worker_pool(
    *,
    worker_count: int,
    case_dir: Path,
    broker_router: str,
    broker_sub: str,
    default_executor: str,
) -> List[WorkerProcess]:
    logs_dir = case_dir / "worker-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    workers: List[WorkerProcess] = []
    for index in range(worker_count):
        worker_id = f"{case_dir.name}-worker-{index + 1:03d}"
        log_handle = (logs_dir / f"{worker_id}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "runtime" / "worker_daemon.py"),
                "--worker-id",
                worker_id,
                "--broker-router",
                broker_router,
                "--broker-sub",
                broker_sub,
                "--default-executor",
                default_executor,
                "--work-root",
                str(case_dir / "worker-local"),
            ],
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers.append(WorkerProcess(process=process, log_handle=log_handle))
    time.sleep(1.0)
    return workers


def _stop_worker_pool(workers: Iterable[WorkerProcess]) -> None:
    items = list(workers)
    for item in items:
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.time() + 10.0
    for item in items:
        remaining = max(0.1, deadline - time.time())
        try:
            item.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            item.process.kill()
            item.process.wait(timeout=5.0)
        finally:
            item.log_handle.close()


def run_scale_case(
    *,
    topology: str,
    worker_count: int,
    task_count: int,
    shard_count: int,
    reduce_arity: int,
    output_root: Path,
    global_timeout_seconds: int,
) -> Dict[str, Any]:
    router_port = _pick_free_tcp_port()
    sub_port = _pick_free_tcp_port()
    broker_router = f"tcp://127.0.0.1:{router_port}"
    broker_sub = f"tcp://127.0.0.1:{sub_port}"

    case_dir = output_root / f"topology-{topology}-workers-{worker_count}"
    case_dir.mkdir(parents=True, exist_ok=True)
    workers = _start_worker_pool(
        worker_count=worker_count,
        case_dir=case_dir,
        broker_router=broker_router,
        broker_sub=broker_sub,
        default_executor="local_reduce",
    )

    benchmark_log = case_dir / "benchmark.log"
    try:
        _run_logged_subprocess(
            cmd=[
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_scale_benchmark.py"),
                "--benchmark",
                "local_reduce",
                "--topology",
                topology,
                "--task-count",
                str(task_count),
                "--start-broker",
                "--broker-router",
                broker_router,
                "--broker-sub",
                broker_sub,
                "--output-dir",
                str(case_dir),
                "--global-timeout-seconds",
                str(global_timeout_seconds),
                "--shard-count",
                str(shard_count),
                "--reduce-arity",
                str(reduce_arity),
            ],
            log_path=benchmark_log,
            cwd=PROJECT_ROOT,
        )
    finally:
        _stop_worker_pool(workers)

    report_path = next(case_dir.rglob("scale_report.json"))
    report = _read_json(report_path)
    return {
        "kind": "scale",
        "topology": topology,
        "worker_count": worker_count,
        "task_count": task_count,
        "throughput_tasks_per_min": float(report["throughput_tasks_per_min"]),
        "throughput_input_items_per_min": float(report["throughput_input_items_per_min"]),
        "p50_latency_ms": int(report["latency_ms"]["p50"]),
        "p95_latency_ms": int(report["latency_ms"]["p95"]),
        "success_count": int(report["success_count"]),
        "failure_count": int(report["failure_count"]),
        "missing_count": int(report["missing_count"]),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report.get("estimated_cost_usd", 0.0)),
        "report_path": str(report_path),
        "log_path": str(benchmark_log),
    }


def summarize_semantic_run(
    *,
    summary_path: Path,
    mode: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    summary = _read_json(summary_path)
    final = dict(summary["final_summary"])
    judgments_path = Path(final["comparison_judgments_path"])
    benchmark_results_path = Path(final["benchmark_results_path"])
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    if not isinstance(judgments, list):
        judgments = list(judgments.get("judgments", []))
    benchmark_results = _read_json(benchmark_results_path)
    judged_record_ids = sorted(
        {
            str(record_id).strip()
            for item in judgments
            for record_id in item.get("record_ids", [])
            if str(record_id).strip()
        }
    )
    grouped_record_count = sum(
        1 for record in benchmark_results.get("records", []) if str(record.get("comparable_group_id", "")).strip()
    )
    record_count = int(final["record_count"])
    paper_count = int(final["paper_count"])
    judged_record_count = len(judged_record_ids)
    return {
        "kind": "semantic",
        "mode": mode,
        "paper_count": paper_count,
        "record_count": record_count,
        "ambiguity_count": int(final["ambiguity_count"]),
        "adjudication_count": int(final["adjudication_count"]),
        "judged_record_count": judged_record_count,
        "grouped_record_count": grouped_record_count,
        "comparable_group_count": int(final["comparable_group_count"]),
        "comparable_count": int(final["comparable_count"]),
        "not_comparable_count": int(final["not_comparable_count"]),
        "uncertain_count": int(final["uncertain_count"]),
        "runtime_seconds": float(runtime_seconds),
        "curation_intervention_rate": (judged_record_count / record_count) if record_count else 0.0,
        "selective_reasoning_ratio": (int(final["adjudication_count"]) / paper_count) if paper_count else 0.0,
        "run_dir": str(summary["run_dir"]),
        "summary_path": str(summary_path),
        "benchmark_results_path": str(benchmark_results_path),
        "comparison_judgments_path": str(judgments_path),
    }


def run_semantic_case(
    *,
    label: str,
    corpus_root: str,
    sample_size: int,
    sample_mode: str,
    sample_seed: int,
    worker_count: int,
    keyword_profile: str,
    max_ambiguities: int,
    output_root: Path,
    agent_model: Optional[str],
    min_year: int,
    candidate_pool_size: int,
    max_scan_files: int,
) -> Dict[str, Any]:
    run_dir = output_root / label
    log_path = run_dir / "runner.log"
    env = dict(os.environ)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "benchmark_scout" / "run_demo.py"),
        "--corpus-root",
        corpus_root,
        "--sample-size",
        str(sample_size),
        "--sample-mode",
        sample_mode,
        "--sample-seed",
        str(sample_seed),
        "--worker-count",
        str(worker_count),
        "--keyword-profile",
        keyword_profile,
        "--max-ambiguities",
        str(max_ambiguities),
        "--run-dir",
        str(run_dir),
        "--min-year",
        str(min_year),
        "--candidate-pool-size",
        str(candidate_pool_size),
        "--max-scan-files",
        str(max_scan_files),
    ]
    if agent_model:
        cmd.extend(["--agent-model", agent_model])
    runtime_seconds = _run_logged_subprocess(cmd=cmd, log_path=log_path, env=env, cwd=PROJECT_ROOT)
    return summarize_semantic_run(summary_path=run_dir / "demo_summary.json", mode=label, runtime_seconds=runtime_seconds)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_svg(path: Path, *, width: int, height: int, elements: List[str]) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        + "".join(elements)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _svg_text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    escaped = html.escape(value)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#222">{escaped}</text>'
    )


def _svg_axes(*, width: int, height: int, left: int, right: int, top: int, bottom: int, title: str, y_label: str) -> List[str]:
    elements = [
        _svg_text(width / 2.0, 28, title, size=22, weight="bold"),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333" stroke-width="1.5"/>',
        _svg_text(24, height / 2.0, y_label, size=14, anchor="middle"),
    ]
    return elements


def _plot_throughput_by_workers(scale_rows: List[Dict[str, Any]], path: Path) -> None:
    width, height = 900, 550
    left, right, top, bottom = 80, 40, 60, 80
    elements = _svg_axes(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        title="Throughput vs Worker Count",
        y_label="Queue tasks / min",
    )
    topologies = sorted({row["topology"] for row in scale_rows})
    colors = {
        "map_reduce": "#1f77b4",
        "sharded_queue": "#ff7f0e",
        "work_queue": "#2ca02c",
    }
    worker_counts = sorted({int(row["worker_count"]) for row in scale_rows})
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_y = max(1.0, max(float(row["throughput_tasks_per_min"]) for row in scale_rows) * 1.15)
    x_positions = {
        value: left + (chart_width * index / max(1, len(worker_counts) - 1))
        for index, value in enumerate(worker_counts)
    }
    for tick in range(6):
        y_value = max_y * tick / 5.0
        y_pos = height - bottom - (chart_height * tick / 5.0)
        elements.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width - right}" y2="{y_pos:.1f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        elements.append(_svg_text(left - 10, y_pos + 5, f"{y_value:.0f}", size=12, anchor="end"))
    for worker_count in worker_counts:
        x_pos = x_positions[worker_count]
        elements.append(
            f'<line x1="{x_pos:.1f}" y1="{height - bottom}" x2="{x_pos:.1f}" y2="{height - bottom + 6}" stroke="#333" stroke-width="1"/>'
        )
        elements.append(_svg_text(x_pos, height - bottom + 24, str(worker_count), size=12))
    for topology in topologies:
        rows = sorted((row for row in scale_rows if row["topology"] == topology), key=lambda item: item["worker_count"])
        points: List[str] = []
        for row in rows:
            x_pos = x_positions[int(row["worker_count"])]
            y_pos = height - bottom - (chart_height * float(row["throughput_tasks_per_min"]) / max_y)
            points.append(f"{x_pos:.1f},{y_pos:.1f}")
        color = colors.get(topology, "#444")
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in rows:
            x_pos = x_positions[int(row["worker_count"])]
            y_pos = height - bottom - (chart_height * float(row["throughput_tasks_per_min"]) / max_y)
            elements.append(f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="4" fill="{color}"/>')
    legend_x = width - right - 180
    legend_y = top + 10
    for index, topology in enumerate(topologies):
        color = colors.get(topology, "#444")
        y_pos = legend_y + index * 22
        elements.append(f'<rect x="{legend_x}" y="{y_pos - 10}" width="12" height="12" fill="{color}"/>')
        elements.append(_svg_text(legend_x + 18, y_pos, topology, size=12, anchor="start"))
    elements.append(_svg_text(width / 2.0, height - 22, "Workers", size=14))
    _write_svg(path, width=width, height=height, elements=elements)


def _plot_selective_reasoning_funnel(two_pass: Dict[str, Any], path: Path) -> None:
    stages = [
        ("Papers", int(two_pass["paper_count"])),
        ("Records", int(two_pass["record_count"])),
        ("Ambiguities", int(two_pass["ambiguity_count"])),
        ("Adjudications", int(two_pass["adjudication_count"])),
        ("Comparable Groups", int(two_pass["comparable_group_count"])),
    ]
    labels = [item[0] for item in stages]
    values = [item[1] for item in stages]
    width, height = 900, 550
    left, right, top, bottom = 80, 40, 60, 80
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_y = max(1.0, max(values) * 1.15)
    bar_width = chart_width / max(1, len(values)) * 0.65
    spacing = chart_width / max(1, len(values))
    colors = ["#1f77b4", "#4c78a8", "#f58518", "#e45756", "#72b7b2"]
    elements = _svg_axes(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        title="Selective Reasoning Funnel",
        y_label="Count",
    )
    for tick in range(6):
        y_value = max_y * tick / 5.0
        y_pos = height - bottom - (chart_height * tick / 5.0)
        elements.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width - right}" y2="{y_pos:.1f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        elements.append(_svg_text(left - 10, y_pos + 5, f"{y_value:.0f}", size=12, anchor="end"))
    for index, (label, value) in enumerate(zip(labels, values)):
        x_pos = left + index * spacing + (spacing - bar_width) / 2.0
        bar_height = chart_height * value / max_y
        y_pos = height - bottom - bar_height
        elements.append(
            f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}"/>'
        )
        elements.append(_svg_text(x_pos + bar_width / 2.0, y_pos - 8, str(value), size=12))
        elements.append(_svg_text(x_pos + bar_width / 2.0, height - bottom + 24, label, size=12))
    _write_svg(path, width=width, height=height, elements=elements)


def _plot_one_pass_vs_two_pass(semantic_rows: List[Dict[str, Any]], path: Path) -> None:
    by_mode = {row["mode"]: row for row in semantic_rows}
    one_pass = by_mode["extraction_only"]
    two_pass = by_mode["two_pass"]
    metric_labels = ["Records Extracted", "Judged Records", "Comparable Groups"]
    one_values = [
        int(one_pass["record_count"]),
        int(one_pass["judged_record_count"]),
        int(one_pass["comparable_group_count"]),
    ]
    two_values = [
        int(two_pass["record_count"]),
        int(two_pass["judged_record_count"]),
        int(two_pass["comparable_group_count"]),
    ]
    width, height = 900, 550
    left, right, top, bottom = 80, 40, 60, 90
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_y = max(1.0, max(one_values + two_values) * 1.15)
    group_spacing = chart_width / max(1, len(metric_labels))
    bar_width = group_spacing * 0.26
    colors = {"Extraction only": "#9ecae1", "Two pass": "#3182bd"}
    elements = _svg_axes(
        width=width,
        height=height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        title="One-Pass vs Two-Pass Output Enrichment",
        y_label="Count",
    )
    for tick in range(6):
        y_value = max_y * tick / 5.0
        y_pos = height - bottom - (chart_height * tick / 5.0)
        elements.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{width - right}" y2="{y_pos:.1f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        elements.append(_svg_text(left - 10, y_pos + 5, f"{y_value:.0f}", size=12, anchor="end"))
    for index, label in enumerate(metric_labels):
        group_left = left + index * group_spacing + group_spacing * 0.2
        for offset, (series_name, values) in enumerate((("Extraction only", one_values), ("Two pass", two_values))):
            value = values[index]
            bar_height = chart_height * value / max_y
            x_pos = group_left + offset * (bar_width + 12)
            y_pos = height - bottom - bar_height
            elements.append(
                f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[series_name]}"/>'
            )
            elements.append(_svg_text(x_pos + bar_width / 2.0, y_pos - 8, str(value), size=12))
        elements.append(_svg_text(group_left + bar_width + 6, height - bottom + 24, label, size=12))
    legend_x = width - right - 190
    legend_y = top + 10
    for index, (series_name, color) in enumerate(colors.items()):
        y_pos = legend_y + index * 22
        elements.append(f'<rect x="{legend_x}" y="{y_pos - 10}" width="12" height="12" fill="{color}"/>')
        elements.append(_svg_text(legend_x + 18, y_pos, series_name, size=12, anchor="start"))
    _write_svg(path, width=width, height=height, elements=elements)


def _best_scale_run(scale_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not scale_rows:
        return None
    return max(scale_rows, key=lambda item: (float(item["throughput_tasks_per_min"]), -int(item["p95_latency_ms"])))


def _render_report(
    *,
    scale_rows: List[Dict[str, Any]],
    semantic_rows: List[Dict[str, Any]],
    charts: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append("# Epsilon Benchmark Report")
    lines.append("")
    if scale_rows and semantic_rows:
        lines.append("This standalone demo measures two things:")
        lines.append("")
        lines.append("- system-scale throughput with deterministic queue workloads")
        lines.append("- semantic curation behavior with Benchmark Scout")
        lines.append("")
    elif scale_rows:
        lines.append("This run measures deterministic system-scale throughput only.")
        lines.append("")
    elif semantic_rows:
        lines.append("This run measures Benchmark Scout curation behavior only.")
        lines.append("")

    best_scale = _best_scale_run(scale_rows)
    two_pass = next((row for row in semantic_rows if row["mode"] == "two_pass"), None)
    one_pass = next((row for row in semantic_rows if row["mode"] == "extraction_only"), None)

    lines.append("## Headline Numbers")
    lines.append("")
    if best_scale is not None:
        lines.append(
            f"- Fastest scale run: `{best_scale['topology']}` at `{best_scale['worker_count']}` workers "
            f"with `{best_scale['throughput_tasks_per_min']:.2f}` queue tasks/min"
        )
    if two_pass is not None:
        lines.append(
            f"- Two-pass semantic run: `{two_pass['paper_count']}` papers, `{two_pass['record_count']}` extracted records, "
            f"`{two_pass['adjudication_count']}` adjudications"
        )
        lines.append(
            f"- Records touched by explicit judgments: `{two_pass['judged_record_count']}` "
            f"({two_pass['curation_intervention_rate'] * 100:.1f}% of extracted records)"
        )
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    if "throughput_by_workers" in charts:
        lines.append(f"- Throughput by workers: `{charts['throughput_by_workers']}`")
    if "selective_reasoning_funnel" in charts:
        lines.append(f"- Selective reasoning funnel: `{charts['selective_reasoning_funnel']}`")
    if "one_pass_vs_two_pass" in charts:
        lines.append(f"- One-pass vs two-pass: `{charts['one_pass_vs_two_pass']}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if scale_rows:
        lines.append("- The scale chart measures orchestration overhead and queue throughput, not model quality.")
    if semantic_rows:
        lines.append("- The semantic charts measure how often the second wave activates and how much of the final dataset it explicitly curates.")
    if one_pass is not None and two_pass is not None:
        delta_judged = int(two_pass["judged_record_count"]) - int(one_pass["judged_record_count"])
        delta_groups = int(two_pass["comparable_group_count"]) - int(one_pass["comparable_group_count"])
        lines.append(
            f"- Compared to extraction-only, the two-pass run added `{delta_judged}` judged records and `{delta_groups}` comparable groups."
        )
    return "\n".join(lines) + "\n"


def write_benchmark_bundle(
    *,
    output_dir: Path,
    scale_rows: List[Dict[str, Any]],
    semantic_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_rows = sorted(scale_rows, key=lambda item: (int(item["worker_count"]), str(item["topology"])))
    semantic_rows = sorted(semantic_rows, key=lambda item: str(item["mode"]))

    charts: Dict[str, str] = {}
    if scale_rows:
        throughput_chart = output_dir / "throughput_by_workers.svg"
        _plot_throughput_by_workers(scale_rows, throughput_chart)
        charts["throughput_by_workers"] = str(throughput_chart)
    if semantic_rows:
        funnel_chart = output_dir / "selective_reasoning_funnel.svg"
        two_pass = next((row for row in semantic_rows if row["mode"] == "two_pass"), semantic_rows[0])
        _plot_selective_reasoning_funnel(two_pass, funnel_chart)
        charts["selective_reasoning_funnel"] = str(funnel_chart)
        if {"extraction_only", "two_pass"}.issubset({row["mode"] for row in semantic_rows}):
            one_pass_chart = output_dir / "one_pass_vs_two_pass.svg"
            _plot_one_pass_vs_two_pass(semantic_rows, one_pass_chart)
            charts["one_pass_vs_two_pass"] = str(one_pass_chart)

    _write_csv(
        output_dir / "scale_metrics.csv",
        scale_rows,
        [
            "topology",
            "worker_count",
            "task_count",
            "throughput_tasks_per_min",
            "throughput_input_items_per_min",
            "p50_latency_ms",
            "p95_latency_ms",
            "success_count",
            "failure_count",
            "missing_count",
            "elapsed_seconds",
            "estimated_cost_usd",
            "report_path",
            "log_path",
        ],
    )
    _write_csv(
        output_dir / "semantic_metrics.csv",
        semantic_rows,
        [
            "mode",
            "paper_count",
            "record_count",
            "ambiguity_count",
            "adjudication_count",
            "judged_record_count",
            "grouped_record_count",
            "comparable_group_count",
            "comparable_count",
            "not_comparable_count",
            "uncertain_count",
            "runtime_seconds",
            "curation_intervention_rate",
            "selective_reasoning_ratio",
            "run_dir",
            "summary_path",
        ],
    )

    bundle = {
        "generated_at_ms": _now_ms(),
        "config": config,
        "scale_runs": scale_rows,
        "semantic_runs": semantic_rows,
        "charts": charts,
    }
    bundle_path = output_dir / "benchmark_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    report_text = _render_report(scale_rows=scale_rows, semantic_rows=semantic_rows, charts=charts)
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    result = {
        "bundle_path": str(bundle_path),
        "scale_metrics_path": str(output_dir / "scale_metrics.csv"),
        "semantic_metrics_path": str(output_dir / "semantic_metrics.csv"),
        "report_path": str(report_path),
    }
    if "throughput_by_workers" in charts:
        result["throughput_chart_path"] = charts["throughput_by_workers"]
    if "selective_reasoning_funnel" in charts:
        result["funnel_chart_path"] = charts["selective_reasoning_funnel"]
    if "one_pass_vs_two_pass" in charts:
        result["one_pass_chart_path"] = charts["one_pass_vs_two_pass"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone Epsilon benchmark/report demo")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs" / f"benchmark-report-{int(time.time() * 1000)}"),
    )
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")

    parser.add_argument("--scale-worker-counts", default="2,8,24")
    parser.add_argument("--scale-topologies", default="work_queue,sharded_queue,map_reduce")
    parser.add_argument("--scale-task-count", type=int, default=480)
    parser.add_argument("--scale-global-timeout-seconds", type=int, default=1800)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--reduce-arity", type=int, default=8)

    parser.add_argument("--corpus-root", required=False, default="")
    parser.add_argument("--semantic-sample-size", type=int, default=40)
    parser.add_argument("--semantic-sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--semantic-sample-seed", type=int, default=17)
    parser.add_argument("--semantic-worker-count", type=int, default=8)
    parser.add_argument("--keyword-profile", default="text_llm_eval")
    parser.add_argument("--semantic-max-ambiguities", type=int, default=40)
    parser.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL"))
    parser.add_argument("--min-year", type=int, default=2022)
    parser.add_argument("--candidate-pool-size", type=int, default=1000)
    parser.add_argument("--max-scan-files", type=int, default=300)
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []

    if not args.skip_scale:
        scale_root = output_dir / "scale_runs"
        for worker_count in _split_csv_ints(args.scale_worker_counts):
            for topology in _split_csv(args.scale_topologies):
                scale_rows.append(
                    run_scale_case(
                        topology=topology,
                        worker_count=worker_count,
                        task_count=int(args.scale_task_count),
                        shard_count=int(args.shard_count),
                        reduce_arity=int(args.reduce_arity),
                        output_root=scale_root,
                        global_timeout_seconds=int(args.scale_global_timeout_seconds),
                    )
                )

    if not args.skip_semantic:
        if not str(args.corpus_root).strip():
            raise RuntimeError("--corpus-root is required unless --skip-semantic is set")
        semantic_root = output_dir / "semantic_runs"
        semantic_rows.append(
            run_semantic_case(
                label="extraction_only",
                corpus_root=str(args.corpus_root),
                sample_size=int(args.semantic_sample_size),
                sample_mode=str(args.semantic_sample_mode),
                sample_seed=int(args.semantic_sample_seed),
                worker_count=int(args.semantic_worker_count),
                keyword_profile=str(args.keyword_profile),
                max_ambiguities=0,
                output_root=semantic_root,
                agent_model=str(args.agent_model) if args.agent_model else None,
                min_year=int(args.min_year),
                candidate_pool_size=int(args.candidate_pool_size),
                max_scan_files=int(args.max_scan_files),
            )
        )
        semantic_rows.append(
            run_semantic_case(
                label="two_pass",
                corpus_root=str(args.corpus_root),
                sample_size=int(args.semantic_sample_size),
                sample_mode=str(args.semantic_sample_mode),
                sample_seed=int(args.semantic_sample_seed),
                worker_count=int(args.semantic_worker_count),
                keyword_profile=str(args.keyword_profile),
                max_ambiguities=int(args.semantic_max_ambiguities),
                output_root=semantic_root,
                agent_model=str(args.agent_model) if args.agent_model else None,
                min_year=int(args.min_year),
                candidate_pool_size=int(args.candidate_pool_size),
                max_scan_files=int(args.max_scan_files),
            )
        )

    if not scale_rows and not semantic_rows:
        raise RuntimeError("Nothing to do: both scale and semantic tracks are disabled")

    artifact_paths = write_benchmark_bundle(
        output_dir=output_dir,
        scale_rows=scale_rows,
        semantic_rows=semantic_rows,
        config={
            "scale_worker_counts": _split_csv_ints(args.scale_worker_counts),
            "scale_topologies": _split_csv(args.scale_topologies),
            "scale_task_count": int(args.scale_task_count),
            "semantic_sample_size": int(args.semantic_sample_size),
            "semantic_sample_mode": str(args.semantic_sample_mode),
            "semantic_sample_seed": int(args.semantic_sample_seed),
            "semantic_worker_count": int(args.semantic_worker_count),
            "keyword_profile": str(args.keyword_profile),
            "semantic_max_ambiguities": int(args.semantic_max_ambiguities),
            "agent_model": str(args.agent_model) if args.agent_model else "",
            "corpus_root": str(args.corpus_root),
        },
    )
    return {
        "output_dir": str(output_dir),
        "scale_run_count": len(scale_rows),
        "semantic_run_count": len(semantic_rows),
        "artifacts": artifact_paths,
    }
