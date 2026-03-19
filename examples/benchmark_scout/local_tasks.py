"""Deterministic reducers/finalizers for the benchmark scout example."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _resolve_ref_path(task_root: Path, ref: str) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    return task_root / candidate


def _extract_outer_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_control_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\r":
        return "\\r"
    if char == "\t":
        return "\\t"
    return f"\\u{ord(char):04x}"


def _is_string_terminator(text: str, index: int) -> bool:
    probe = index + 1
    while probe < len(text) and text[probe] in " \t\r\n":
        probe += 1
    if probe >= len(text):
        return True
    return text[probe] in ",]}:"


def _repair_json_text(text: str) -> str:
    raw = _extract_outer_object(text.strip())
    repaired: List[str] = []
    in_string = False
    escape = False
    for index, char in enumerate(raw):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            continue
        if escape:
            repaired.append(char)
            escape = False
            continue
        if char == "\\":
            repaired.append(char)
            escape = True
            continue
        if char == '"':
            if _is_string_terminator(raw, index):
                repaired.append(char)
                in_string = False
            else:
                repaired.append('\\"')
            continue
        if ord(char) < 32:
            repaired.append(_normalize_control_char(char))
            continue
        repaired.append(char)
    repaired_text = "".join(repaired)
    return re.sub(r",(\s*[}\]])", r"\1", repaired_text)


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return json.loads(_repair_json_text(text), strict=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_key(value: Any) -> str:
    text = _normalize_text(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_set(value: Any) -> set[str]:
    return {token for token in _normalize_key(value).split() if token}


GENERIC_TASK_TOKENS = {
    "across",
    "analysis",
    "benchmark",
    "benchmarks",
    "comparison",
    "comparisons",
    "evaluation",
    "evaluations",
    "general",
    "model",
    "models",
    "overall",
    "performance",
    "problem",
    "problems",
    "quality",
    "system",
    "systems",
    "task",
    "tasks",
}

GENERIC_DATASET_TOKENS = {
    "benchmark",
    "benchmarks",
    "bench",
    "corpus",
    "curated",
    "dataset",
    "datasets",
    "dev",
    "evaluation",
    "full",
    "mini",
    "overall",
    "question",
    "questions",
    "sample",
    "samples",
    "seen",
    "set",
    "subset",
    "suite",
    "task",
    "tasks",
    "test",
    "testbed",
    "train",
    "unseen",
    "validation",
    "verified",
}

GENERIC_MODEL_TOKENS = {
    "base",
    "instruct",
    "instruction",
    "llm",
    "llms",
    "mini",
    "model",
    "models",
}

METRIC_SYNONYMS = {
    "acc": "accuracy",
    "accuracies": "accuracy",
    "pass": "pass",
    "pass1": "pass@1",
    "pass10": "pass@10",
    "f1score": "f1",
    "f1scores": "f1",
}


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _append_limited(items: List[Any], additions: Iterable[Any], limit: int) -> List[Any]:
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items}
    for item in additions:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_record_group_key(record: Dict[str, Any]) -> str:
    return "|".join(
        [
            _normalize_key(record.get("task")),
            _normalize_key(record.get("dataset")),
            _normalize_key(record.get("metric")),
        ]
    )


def _semantic_tokens(value: Any, *, field: str) -> set[str]:
    normalized = _normalize_key(value)
    normalized = normalized.replace("pass 1", "pass1").replace("pass 10", "pass10")
    tokens = [METRIC_SYNONYMS.get(token, token) for token in normalized.split()]
    if field == "task":
        return {token for token in tokens if token and token not in GENERIC_TASK_TOKENS}
    if field == "dataset":
        return {token for token in tokens if token and token not in GENERIC_DATASET_TOKENS}
    if field == "model":
        return {token for token in tokens if token and token not in GENERIC_MODEL_TOKENS}
    return {token for token in tokens if token}


def _metric_key(value: Any) -> str:
    tokens = sorted(_semantic_tokens(value, field="metric"))
    return " ".join(tokens)


def _overlap_count(left: set[str], right: set[str]) -> int:
    return len(left & right)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dataset_signal(left: set[str], right: set[str]) -> tuple[bool, int, float]:
    overlap = _overlap_count(left, right)
    similarity = _jaccard(left, right)
    return overlap > 0 or similarity >= 0.5, overlap, similarity


def _build_fuzzy_group_key(left: Dict[str, Any], right: Dict[str, Any]) -> str:
    shared_dataset = sorted(set(left["dataset_tokens"]) & set(right["dataset_tokens"]))
    shared_task = sorted(set(left["task_tokens"]) & set(right["task_tokens"]))
    label = shared_dataset or shared_task or [left["metric_key"] or "metric"]
    return f"fuzzy:{left['metric_key'] or 'metric'}:{'-'.join(label[:3])}"


def _candidate_payload(
    *,
    left: Dict[str, Any],
    right: Dict[str, Any],
    group_key: str,
    reason_bits: List[str],
    score: float,
    ambiguity_id: str,
) -> Dict[str, Any]:
    return {
        "ambiguity_id": ambiguity_id,
        "group_key": group_key,
        "reason": ",".join(reason_bits),
        "score": round(min(score, 0.99), 3),
        "left_record": {
            key: left[key]
            for key in (
                "record_id",
                "paper_id",
                "paper_title",
                "paper_year",
                "paper_venue",
                "task",
                "dataset",
                "metric",
                "model_name",
                "score_text",
                "score_value",
                "score_unit",
                "evaluation_mode",
                "dataset_split",
                "setup_notes",
                "confidence",
                "evidence",
            )
        },
        "right_record": {
            key: right[key]
            for key in (
                "record_id",
                "paper_id",
                "paper_title",
                "paper_year",
                "paper_venue",
                "task",
                "dataset",
                "metric",
                "model_name",
                "score_text",
                "score_value",
                "score_unit",
                "evaluation_mode",
                "dataset_split",
                "setup_notes",
                "confidence",
                "evidence",
            )
        },
    }


def _benchmark_record(record: Dict[str, Any], *, paper: Dict[str, Any], ordinal: int) -> Dict[str, Any]:
    evidence = _dedupe_strings(record.get("evidence", []))[:3]
    score_value = _coerce_float(record.get("score_value"))
    normalized = {
        "record_id": f"{paper['paper_id']}-rec-{ordinal:03d}",
        "paper_id": paper["paper_id"],
        "paper_title": paper["title"],
        "paper_year": paper["year"],
        "paper_venue": paper["venue"],
        "task": _normalize_text(record.get("task")),
        "dataset": _normalize_text(record.get("dataset")),
        "metric": _normalize_text(record.get("metric")),
        "model_name": _normalize_text(record.get("model_name")),
        "score_text": _normalize_text(record.get("score_text")),
        "score_value": score_value,
        "score_unit": _normalize_text(record.get("score_unit")) or "unknown",
        "evaluation_mode": _normalize_text(record.get("evaluation_mode")) or "unknown",
        "dataset_split": _normalize_text(record.get("dataset_split")) or "unknown",
        "setup_notes": _normalize_text(record.get("setup_notes")),
        "confidence": float(record.get("confidence", 0.0) or 0.0),
        "evidence": [
            {
                "paper_id": paper["paper_id"],
                "paper_title": paper["title"],
                "text": snippet[:280],
            }
            for snippet in evidence
            if snippet
        ],
    }
    normalized["group_key"] = _build_record_group_key(normalized)
    normalized["model_key"] = _normalize_key(normalized["model_name"])
    normalized["task_tokens"] = sorted(_semantic_tokens(normalized["task"], field="task"))
    normalized["dataset_tokens"] = sorted(_semantic_tokens(normalized["dataset"], field="dataset"))
    normalized["metric_key"] = _metric_key(normalized["metric"])
    normalized["metric_tokens"] = sorted(_semantic_tokens(normalized["metric"], field="metric"))
    normalized["model_tokens"] = sorted(_semantic_tokens(normalized["model_name"], field="model"))
    return normalized


def _refresh_record_features(record: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(record)
    updated["group_key"] = _build_record_group_key(updated)
    updated["model_key"] = _normalize_key(updated.get("model_name"))
    updated["task_tokens"] = sorted(_semantic_tokens(updated.get("task"), field="task"))
    updated["dataset_tokens"] = sorted(_semantic_tokens(updated.get("dataset"), field="dataset"))
    updated["metric_key"] = _metric_key(updated.get("metric"))
    updated["metric_tokens"] = sorted(_semantic_tokens(updated.get("metric"), field="metric"))
    updated["model_tokens"] = sorted(_semantic_tokens(updated.get("model_name"), field="model"))
    return updated


def _aggregate_from_map_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    paper = {
        "paper_id": _normalize_text(payload.get("paper_id")),
        "title": _normalize_text(payload.get("title")),
        "year": int(payload.get("year", 0) or 0),
        "venue": _normalize_text(payload.get("venue")),
        "summary": _normalize_text(payload.get("summary")),
    }
    records: List[Dict[str, Any]] = []
    for index, record in enumerate(payload.get("benchmark_records", []), start=1):
        if not isinstance(record, dict):
            continue
        if not (_normalize_text(record.get("task")) and _normalize_text(record.get("dataset")) and _normalize_text(record.get("metric"))):
            continue
        records.append(_benchmark_record(record, paper=paper, ordinal=index))
    return {
        "kind": "benchmark_aggregate",
        "paper_summaries": [
            {
                **paper,
                "record_count": len(records),
            }
        ],
        "benchmark_records": records,
        "ambiguity_candidates": [],
    }


def _compute_ambiguity_candidates(records: List[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("group_key"):
            groups[str(record["group_key"])].append(record)

    candidate_payloads: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    for group_key, group_records in sorted(groups.items()):
        if len(group_records) < 2:
            continue
        ranked = sorted(
            group_records,
            key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("paper_year", 0) or 0), item["record_id"]),
        )[:6]
        for left_index, left in enumerate(ranked):
            for right in ranked[left_index + 1 :]:
                if left["paper_id"] == right["paper_id"]:
                    continue
                pair_key = tuple(sorted((left["record_id"], right["record_id"])))
                seen_pairs.add(pair_key)
                reason_bits: List[str] = []
                if left["evaluation_mode"] != right["evaluation_mode"]:
                    reason_bits.append("evaluation_mode_diff")
                if left["dataset_split"] != right["dataset_split"]:
                    reason_bits.append("dataset_split_diff")
                if left["model_key"] != right["model_key"]:
                    reason_bits.append("cross_model")
                if _normalize_key(left.get("setup_notes")) != _normalize_key(right.get("setup_notes")):
                    reason_bits.append("setup_note_diff")
                if not reason_bits:
                    reason_bits.append("same_group_key")
                model_overlap = len(_token_set(left["model_name"]) & _token_set(right["model_name"]))
                score = 0.5 + min(float(left["confidence"]), float(right["confidence"])) * 0.3
                score += 0.05 * len(reason_bits)
                if model_overlap:
                    score += 0.1
                candidate_payloads.append(
                    _candidate_payload(
                        left=left,
                        right=right,
                        group_key=group_key,
                        reason_bits=reason_bits,
                        score=score,
                        ambiguity_id="",
                    )
                )

    ranked_records = sorted(
        records,
        key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("paper_year", 0) or 0), item["record_id"]),
    )[: min(len(records), 80)]
    for left_index, left in enumerate(ranked_records):
        left_task = set(left.get("task_tokens", []))
        left_dataset = set(left.get("dataset_tokens", []))
        left_model = set(left.get("model_tokens", []))
        left_metric = set(left.get("metric_tokens", []))
        for right in ranked_records[left_index + 1 :]:
            if left["paper_id"] == right["paper_id"]:
                continue
            pair_key = tuple(sorted((left["record_id"], right["record_id"])))
            if pair_key in seen_pairs:
                continue

            right_task = set(right.get("task_tokens", []))
            right_dataset = set(right.get("dataset_tokens", []))
            right_model = set(right.get("model_tokens", []))
            right_metric = set(right.get("metric_tokens", []))

            task_overlap = _overlap_count(left_task, right_task)
            dataset_signal, dataset_overlap, dataset_similarity = _dataset_signal(left_dataset, right_dataset)
            model_overlap = _overlap_count(left_model, right_model)
            metric_overlap = _overlap_count(left_metric, right_metric)
            metric_match = bool(left.get("metric_key")) and left.get("metric_key") == right.get("metric_key")
            same_unit = left.get("score_unit") == right.get("score_unit")
            task_similarity = _jaccard(left_task, right_task)

            if not dataset_signal:
                continue
            metric_signal = metric_match or metric_overlap > 0
            if metric_match:
                qualifies = True
            elif metric_overlap > 0 and (task_overlap > 0 or task_similarity >= 0.25 or model_overlap > 0 or same_unit):
                qualifies = True
            elif dataset_overlap > 0 and same_unit and (task_overlap > 0 or task_similarity >= 0.5):
                qualifies = True
            elif dataset_overlap > 1 and model_overlap > 0 and (task_overlap > 0 or metric_signal):
                qualifies = True
            else:
                qualifies = False
            if not qualifies:
                continue

            reason_bits = []
            if dataset_overlap > 0:
                reason_bits.append("dataset_overlap")
            if task_overlap > 0:
                reason_bits.append("task_overlap")
            if model_overlap > 0:
                reason_bits.append("model_overlap")
            if metric_match:
                reason_bits.append("metric_match")
            elif metric_overlap > 0:
                reason_bits.append("metric_overlap")
            if same_unit:
                reason_bits.append("score_unit_match")
            if left["evaluation_mode"] != right["evaluation_mode"]:
                reason_bits.append("evaluation_mode_diff")
            if left["dataset_split"] != right["dataset_split"]:
                reason_bits.append("dataset_split_diff")
            if _normalize_key(left.get("setup_notes")) != _normalize_key(right.get("setup_notes")):
                reason_bits.append("setup_note_diff")

            score = 0.36 + min(float(left.get("confidence", 0.0)), float(right.get("confidence", 0.0))) * 0.22
            score += 0.2 * min(dataset_overlap, 2)
            score += 0.1 * min(task_overlap, 2)
            score += 0.06 * min(model_overlap, 2)
            score += 0.08 if metric_match else 0.04 * min(metric_overlap, 2)
            if same_unit:
                score += 0.04
            score += 0.03 * len(
                [bit for bit in reason_bits if bit in {"evaluation_mode_diff", "dataset_split_diff", "setup_note_diff"}]
            )

            candidate_payloads.append(
                _candidate_payload(
                    left=left,
                    right=right,
                    group_key=_build_fuzzy_group_key(left, right),
                    reason_bits=reason_bits,
                    score=score,
                    ambiguity_id="",
                )
            )
            seen_pairs.add(pair_key)

    candidate_payloads.sort(
        key=lambda item: (
            -float(item["score"]),
            item["left_record"]["record_id"],
            item["right_record"]["record_id"],
        )
    )
    trimmed = candidate_payloads[:max_candidates]
    for index, item in enumerate(trimmed, start=1):
        item["ambiguity_id"] = f"amb-{index:04d}"
    return trimmed


def _merge_aggregate_payloads(payloads: List[Dict[str, Any]], max_ambiguities: int) -> Dict[str, Any]:
    paper_index: Dict[str, Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    for payload in payloads:
        if payload.get("kind") != "benchmark_aggregate":
            payload = _aggregate_from_map_output(payload)
        for paper in payload.get("paper_summaries", []):
            if not isinstance(paper, dict):
                continue
            paper_id = _normalize_text(paper.get("paper_id"))
            if not paper_id:
                continue
            paper_index[paper_id] = {
                "paper_id": paper_id,
                "title": _normalize_text(paper.get("title")),
                "year": int(paper.get("year", 0) or 0),
                "venue": _normalize_text(paper.get("venue")),
                "summary": _normalize_text(paper.get("summary")),
                "record_count": int(paper.get("record_count", 0) or 0),
            }
        for record in payload.get("benchmark_records", []):
            if isinstance(record, dict):
                records.append(_refresh_record_features(record))
    ambiguity_candidates = _compute_ambiguity_candidates(records, max_candidates=max_ambiguities)
    return {
        "kind": "benchmark_aggregate",
        "paper_summaries": sorted(paper_index.values(), key=lambda item: item["paper_id"]),
        "benchmark_records": records,
        "ambiguity_candidates": ambiguity_candidates,
    }


def _flatten_bundle(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("kind") != "bundle_outputs":
        return [payload]
    flattened: List[Dict[str, Any]] = []
    for item in payload.get("outputs", []):
        if isinstance(item, dict):
            flattened.extend(_flatten_bundle(item))
    return flattened


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


def finalize_benchmark_results(
    *,
    run_dir: Path,
    phase1_summary: Dict[str, Any],
    phase2_summary: Dict[str, Any] | None,
    corpus_root: str,
    keyword_profile: str,
) -> Dict[str, Any]:
    aggregate = json.loads(Path(phase1_summary["final_output_path"]).read_text(encoding="utf-8"))
    judgments: List[Dict[str, Any]] = []
    if phase2_summary and phase2_summary.get("final_output_exists"):
        judgments = _flatten_bundle(json.loads(Path(phase2_summary["final_output_path"]).read_text(encoding="utf-8")))

    records = [dict(record) for record in aggregate.get("benchmark_records", []) if isinstance(record, dict)]
    paper_summaries = [dict(item) for item in aggregate.get("paper_summaries", []) if isinstance(item, dict)]
    record_ids = [record["record_id"] for record in records if record.get("record_id")]
    union_find = _UnionFind(record_ids)

    canonical_votes: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for judgment in judgments:
        if str(judgment.get("decision", "")).strip().lower() != "comparable":
            continue
        pair = [str(item).strip() for item in judgment.get("record_ids", []) if str(item).strip()]
        if len(pair) != 2 or pair[0] not in union_find.parent or pair[1] not in union_find.parent:
            continue
        union_find.union(pair[0], pair[1])

    for judgment in judgments:
        decision = str(judgment.get("decision", "")).strip().lower()
        pair = [str(item).strip() for item in judgment.get("record_ids", []) if str(item).strip()]
        if decision != "comparable" or len(pair) != 2 or pair[0] not in union_find.parent or pair[1] not in union_find.parent:
            continue
        root = union_find.find(pair[0])
        for field in ("canonical_task", "canonical_dataset", "canonical_metric"):
            value = _normalize_text(judgment.get(field))
            if not value:
                continue
            votes = canonical_votes[root][field]
            votes[value] = int(votes.get(value, 0)) + 1

    comparable_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            continue
        comparable_groups[union_find.find(record_id)].append(record)

    group_id_by_root: Dict[str, str] = {}
    comparable_group_payloads: List[Dict[str, Any]] = []
    for index, (root, group_records) in enumerate(sorted(comparable_groups.items()), start=1):
        if len(group_records) < 2:
            continue
        group_id = f"cmp-{index:04d}"
        group_id_by_root[root] = group_id
        anchor = sorted(group_records, key=lambda item: item["record_id"])[0]
        canonical_task = max(canonical_votes[root]["canonical_task"].items(), key=lambda item: (item[1], len(item[0])), default=(anchor["task"], 0))[0]
        canonical_dataset = max(canonical_votes[root]["canonical_dataset"].items(), key=lambda item: (item[1], len(item[0])), default=(anchor["dataset"], 0))[0]
        canonical_metric = max(canonical_votes[root]["canonical_metric"].items(), key=lambda item: (item[1], len(item[0])), default=(anchor["metric"], 0))[0]
        comparable_group_payloads.append(
            {
                "group_id": group_id,
                "group_key": anchor.get("group_key", ""),
                "canonical_task": canonical_task,
                "canonical_dataset": canonical_dataset,
                "canonical_metric": canonical_metric,
                "record_ids": sorted(record["record_id"] for record in group_records),
                "paper_ids": sorted({record["paper_id"] for record in group_records}),
                "record_count": len(group_records),
            }
        )

    for record in records:
        root = union_find.find(record["record_id"])
        record["comparable_group_id"] = group_id_by_root.get(root, "")

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = {
        "dataset": {
            "corpus_root": corpus_root,
            "keyword_profile": keyword_profile,
        },
        "paper_count": len(paper_summaries),
        "record_count": len(records),
        "comparable_group_count": len(comparable_group_payloads),
        "records": sorted(records, key=lambda item: (item["task"], item["dataset"], item["metric"], item["paper_id"], item["record_id"])),
        "comparable_groups": comparable_group_payloads,
    }
    benchmark_results_path = final_dir / "benchmark_results.json"
    benchmark_results_path.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")

    judgments_path = final_dir / "comparison_judgments.json"
    judgments_path.write_text(json.dumps(judgments, indent=2), encoding="utf-8")

    paper_index_path = final_dir / "paper_index.json"
    paper_index_path.write_text(
        json.dumps({paper["paper_id"]: paper for paper in paper_summaries}, indent=2),
        encoding="utf-8",
    )

    csv_path = final_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(
            sink,
            fieldnames=[
                "record_id",
                "paper_id",
                "paper_title",
                "paper_year",
                "paper_venue",
                "task",
                "dataset",
                "metric",
                "model_name",
                "score_text",
                "score_value",
                "score_unit",
                "evaluation_mode",
                "dataset_split",
                "setup_notes",
                "confidence",
                "comparable_group_id",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in writer.fieldnames})

    comparable_count = sum(1 for item in judgments if str(item.get("decision", "")).strip().lower() == "comparable")
    not_comparable_count = sum(1 for item in judgments if str(item.get("decision", "")).strip().lower() == "not_comparable")
    uncertain_count = sum(1 for item in judgments if str(item.get("decision", "")).strip().lower() == "uncertain")
    report = "\n".join(
        [
            "# S2ORC Benchmark Scout",
            "",
            f"- Papers processed: {len(paper_summaries)}",
            f"- Benchmark records extracted: {len(records)}",
            f"- Ambiguity candidates: {len(aggregate.get('ambiguity_candidates', []))}",
            f"- Adjudication tasks completed: {len(judgments)}",
            f"- Comparable decisions: {comparable_count}",
            f"- Not comparable decisions: {not_comparable_count}",
            f"- Uncertain decisions: {uncertain_count}",
            f"- Comparable groups: {len(comparable_group_payloads)}",
            "",
            "## Outputs",
            "",
            "- `final/benchmark_results.json`",
            "- `final/benchmark_results.csv`",
            "- `final/comparison_judgments.json`",
            "- `final/paper_index.json`",
        ]
    )
    run_report_path = final_dir / "run_report.md"
    run_report_path.write_text(report + "\n", encoding="utf-8")

    return {
        "benchmark_results_path": str(benchmark_results_path),
        "benchmark_results_csv_path": str(csv_path),
        "comparison_judgments_path": str(judgments_path),
        "paper_index_path": str(paper_index_path),
        "run_report_path": str(run_report_path),
        "paper_count": len(paper_summaries),
        "record_count": len(records),
        "ambiguity_count": len(aggregate.get("ambiguity_candidates", [])),
        "adjudication_count": len(judgments),
        "comparable_group_count": len(comparable_group_payloads),
        "comparable_count": comparable_count,
        "not_comparable_count": not_comparable_count,
        "uncertain_count": uncertain_count,
    }


def run_task(*, task_id: str, task_payload: Dict[str, Any], task_root: Path) -> Dict[str, Any]:
    operation = str(task_payload.get("operation", "") or "").strip()
    output_ref = str(task_payload.get("output_ref", "") or "").strip()
    if not output_ref:
        raise ValueError("python_handler task requires 'output_ref'")
    output_path = _resolve_ref_path(task_root, output_ref)
    input_refs = [str(ref) for ref in task_payload.get("input_refs", [])]

    if operation == "merge_benchmark_records":
        payloads = [_read_json(_resolve_ref_path(task_root, ref)) for ref in input_refs]
        result = _merge_aggregate_payloads(
            payloads,
            max_ambiguities=int(task_payload.get("max_ambiguities", 24)),
        )
    elif operation == "bundle_outputs":
        outputs: List[Dict[str, Any]] = []
        for ref in input_refs:
            child_payload = _read_json(_resolve_ref_path(task_root, ref))
            outputs.extend(_flatten_bundle(child_payload) if isinstance(child_payload, dict) else [])
        result = {
            "kind": "bundle_outputs",
            "bundle_label": str(task_payload.get("reduce_label", "")).strip(),
            "item_count": len(outputs),
            "outputs": outputs,
        }
    else:
        raise ValueError(f"Unsupported benchmark-scout local operation '{operation}'")

    _write_json(output_path, result)
    return {
        "returncode": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "output_path": str(output_path),
        "output_exists": True,
        "output_preview": json.dumps(result)[:2000],
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
