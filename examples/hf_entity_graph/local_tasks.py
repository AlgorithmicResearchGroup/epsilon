"""Deterministic local reducers/finalizers for the HF entity graph demo."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _resolve_ref_path(task_root: Path, ref: str) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    return task_root / candidate


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


def _normalize_name(value: str) -> str:
    lowered = str(value or "").strip().casefold()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _token_set(value: str) -> set[str]:
    return {token for token in _normalize_name(value).split() if token}


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
    repaired_text = re.sub(r",(\s*[}\]])", r"\1", repaired_text)
    return repaired_text


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
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


def _dominant_type(type_counts: Dict[str, int]) -> str:
    if not type_counts:
        return "OTHER"
    return sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _build_candidate_from_map_output(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    doc_summary = {
        "doc_id": str(payload.get("doc_id", "")).strip(),
        "summary": str(payload.get("summary", "")).strip(),
        "keywords": _dedupe_strings(payload.get("keywords", [])),
    }

    entity_map: Dict[str, Dict[str, Any]] = {}
    relation_map: Dict[str, Dict[str, Any]] = {}

    for raw_entity in payload.get("entities", []):
        if not isinstance(raw_entity, dict):
            continue
        name = str(raw_entity.get("name", "")).strip()
        if not name:
            continue
        key = _normalize_name(name)
        if not key:
            continue
        aliases = _dedupe_strings([name, *raw_entity.get("aliases", [])])
        entity_type = str(raw_entity.get("entity_type", "OTHER") or "OTHER").strip().upper()
        confidence = float(raw_entity.get("confidence", 0.0) or 0.0)
        evidence = _dedupe_strings(raw_entity.get("evidence", []))
        candidate = entity_map.setdefault(
            key,
            {
                "cluster_key": key,
                "representative_name": name,
                "aliases": [],
                "entity_types": {},
                "doc_ids": [],
                "mention_count": 0,
                "max_confidence": 0.0,
                "evidence": [],
            },
        )
        if len(name) > len(candidate["representative_name"]):
            candidate["representative_name"] = name
        candidate["aliases"] = _dedupe_strings([*candidate["aliases"], *aliases])[:12]
        candidate["entity_types"][entity_type] = int(candidate["entity_types"].get(entity_type, 0)) + 1
        candidate["doc_ids"] = _dedupe_strings([*candidate["doc_ids"], doc_summary["doc_id"]])
        candidate["mention_count"] += 1
        candidate["max_confidence"] = max(float(candidate["max_confidence"]), confidence)
        evidence_items = [
            {
                "doc_id": doc_summary["doc_id"],
                "text": snippet[:240],
                "name": name,
                "entity_type": entity_type,
                "confidence": confidence,
            }
            for snippet in evidence[:3]
            if snippet
        ]
        candidate["evidence"] = _append_limited(candidate["evidence"], evidence_items, limit=8)

    for raw_relation in payload.get("relations", []):
        if not isinstance(raw_relation, dict):
            continue
        source = str(raw_relation.get("source", "")).strip()
        target = str(raw_relation.get("target", "")).strip()
        relation = str(raw_relation.get("relation", "")).strip().lower()
        if not source or not target or not relation:
            continue
        source_key = _normalize_name(source)
        target_key = _normalize_name(target)
        if not source_key or not target_key:
            continue
        relation_key = f"{source_key}|{relation}|{target_key}"
        confidence = float(raw_relation.get("confidence", 0.0) or 0.0)
        candidate = relation_map.setdefault(
            relation_key,
            {
                "source_key": source_key,
                "target_key": target_key,
                "relation": relation,
                "source_name": source,
                "target_name": target,
                "doc_ids": [],
                "mention_count": 0,
                "max_confidence": 0.0,
                "evidence": [],
            },
        )
        candidate["doc_ids"] = _dedupe_strings([*candidate["doc_ids"], doc_summary["doc_id"]])
        candidate["mention_count"] += 1
        candidate["max_confidence"] = max(float(candidate["max_confidence"]), confidence)
        evidence_items = [
            {
                "doc_id": doc_summary["doc_id"],
                "text": str(snippet)[:240],
                "confidence": confidence,
            }
            for snippet in raw_relation.get("evidence", [])[:2]
        ]
        candidate["evidence"] = _append_limited(candidate["evidence"], evidence_items, limit=6)

    aggregate = {
        "kind": "entity_graph_aggregate",
        "document_summaries": [doc_summary],
        "entity_candidates": list(entity_map.values()),
        "relation_candidates": list(relation_map.values()),
        "ambiguity_candidates": [],
    }
    return doc_summary, aggregate, list(entity_map.values())


def _merge_aggregate_payloads(payloads: List[Dict[str, Any]], max_ambiguities: int) -> Dict[str, Any]:
    documents: Dict[str, Dict[str, Any]] = {}
    entity_index: Dict[str, Dict[str, Any]] = {}
    relation_index: Dict[str, Dict[str, Any]] = {}

    for payload in payloads:
        if payload.get("kind") != "entity_graph_aggregate":
            _, aggregate, _ = _build_candidate_from_map_output(payload)
            payload = aggregate

        for doc in payload.get("document_summaries", []):
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("doc_id", "")).strip()
            if not doc_id:
                continue
            documents[doc_id] = {
                "doc_id": doc_id,
                "summary": str(doc.get("summary", "")).strip(),
                "keywords": _dedupe_strings(doc.get("keywords", [])),
            }

        for raw_entity in payload.get("entity_candidates", []):
            if not isinstance(raw_entity, dict):
                continue
            key = str(raw_entity.get("cluster_key", "")).strip()
            if not key:
                continue
            candidate = entity_index.setdefault(
                key,
                {
                    "cluster_key": key,
                    "representative_name": str(raw_entity.get("representative_name", "")).strip() or key,
                    "aliases": [],
                    "entity_types": {},
                    "doc_ids": [],
                    "mention_count": 0,
                    "max_confidence": 0.0,
                    "evidence": [],
                },
            )
            representative = str(raw_entity.get("representative_name", "")).strip()
            if len(representative) > len(candidate["representative_name"]):
                candidate["representative_name"] = representative
            candidate["aliases"] = _dedupe_strings([*candidate["aliases"], *raw_entity.get("aliases", [])])[:16]
            for entity_type, count in raw_entity.get("entity_types", {}).items():
                candidate["entity_types"][entity_type] = int(candidate["entity_types"].get(entity_type, 0)) + int(count)
            candidate["doc_ids"] = _dedupe_strings([*candidate["doc_ids"], *raw_entity.get("doc_ids", [])])
            candidate["mention_count"] += int(raw_entity.get("mention_count", 0))
            candidate["max_confidence"] = max(float(candidate["max_confidence"]), float(raw_entity.get("max_confidence", 0.0) or 0.0))
            candidate["evidence"] = _append_limited(candidate["evidence"], raw_entity.get("evidence", []), limit=10)

        for raw_relation in payload.get("relation_candidates", []):
            if not isinstance(raw_relation, dict):
                continue
            source_key = str(raw_relation.get("source_key", "")).strip()
            target_key = str(raw_relation.get("target_key", "")).strip()
            relation = str(raw_relation.get("relation", "")).strip().lower()
            if not source_key or not target_key or not relation:
                continue
            key = f"{source_key}|{relation}|{target_key}"
            candidate = relation_index.setdefault(
                key,
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "relation": relation,
                    "source_name": str(raw_relation.get("source_name", "")).strip() or source_key,
                    "target_name": str(raw_relation.get("target_name", "")).strip() or target_key,
                    "doc_ids": [],
                    "mention_count": 0,
                    "max_confidence": 0.0,
                    "evidence": [],
                },
            )
            candidate["doc_ids"] = _dedupe_strings([*candidate["doc_ids"], *raw_relation.get("doc_ids", [])])
            candidate["mention_count"] += int(raw_relation.get("mention_count", 0))
            candidate["max_confidence"] = max(float(candidate["max_confidence"]), float(raw_relation.get("max_confidence", 0.0) or 0.0))
            candidate["evidence"] = _append_limited(candidate["evidence"], raw_relation.get("evidence", []), limit=8)

    entity_candidates = sorted(
        entity_index.values(),
        key=lambda item: (-int(item["mention_count"]), item["representative_name"]),
    )
    relation_candidates = sorted(
        relation_index.values(),
        key=lambda item: (-int(item["mention_count"]), item["relation"], item["source_key"], item["target_key"]),
    )
    ambiguity_candidates = _compute_ambiguity_candidates(entity_candidates, max_candidates=max_ambiguities)
    return {
        "kind": "entity_graph_aggregate",
        "document_summaries": sorted(documents.values(), key=lambda item: item["doc_id"]),
        "entity_candidates": entity_candidates,
        "relation_candidates": relation_candidates,
        "ambiguity_candidates": ambiguity_candidates,
    }


def _compute_ambiguity_candidates(entity_candidates: List[Dict[str, Any]], max_candidates: int) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for index, left in enumerate(entity_candidates):
        left_tokens = _token_set(left["cluster_key"])
        left_aliases = {_normalize_name(alias) for alias in left.get("aliases", []) if alias}
        left_type = _dominant_type(left.get("entity_types", {}))
        for right in entity_candidates[index + 1 :]:
            right_tokens = _token_set(right["cluster_key"])
            right_aliases = {_normalize_name(alias) for alias in right.get("aliases", []) if alias}
            right_type = _dominant_type(right.get("entity_types", {}))
            alias_overlap = left_aliases & right_aliases
            union = left_tokens | right_tokens
            jaccard = (len(left_tokens & right_tokens) / len(union)) if union else 0.0
            containment = 1.0 if left["cluster_key"] in right["cluster_key"] or right["cluster_key"] in left["cluster_key"] else 0.0
            score = max(1.0 if alias_overlap else 0.0, (0.65 * jaccard) + (0.35 * containment))
            if score < 0.72 and not alias_overlap:
                continue
            if left_type != right_type and score < 0.9:
                continue
            if int(left["mention_count"]) == 1 and int(right["mention_count"]) == 1 and score < 0.9:
                continue
            ambiguity_id = f"amb-{len(scored) + 1:04d}"
            scored.append(
                {
                    "ambiguity_id": ambiguity_id,
                    "reason": "alias_overlap" if alias_overlap else "name_similarity",
                    "score": round(score, 3),
                    "cluster_keys": [left["cluster_key"], right["cluster_key"]],
                    "entity_candidates": [
                        {
                            "cluster_key": left["cluster_key"],
                            "representative_name": left["representative_name"],
                            "aliases": left.get("aliases", [])[:8],
                            "dominant_type": left_type,
                            "doc_ids": left.get("doc_ids", [])[:12],
                            "evidence": left.get("evidence", [])[:3],
                        },
                        {
                            "cluster_key": right["cluster_key"],
                            "representative_name": right["representative_name"],
                            "aliases": right.get("aliases", [])[:8],
                            "dominant_type": right_type,
                            "doc_ids": right.get("doc_ids", [])[:12],
                            "evidence": right.get("evidence", [])[:3],
                        },
                    ],
                }
            )

    return sorted(scored, key=lambda item: (-float(item["score"]), item["ambiguity_id"]))[:max_candidates]


def _flatten_bundled_outputs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("kind") != "bundle_outputs":
        return [payload]
    flattened: List[Dict[str, Any]] = []
    for item in payload.get("outputs", []):
        if isinstance(item, dict):
            flattened.extend(_flatten_bundled_outputs(item))
    return flattened


def run_task(*, task_id: str, task_payload: Dict[str, Any], task_root: Path) -> Dict[str, Any]:
    operation = str(task_payload.get("operation", "") or "").strip()
    output_ref = str(task_payload.get("output_ref", "") or "").strip()
    if not output_ref:
        raise ValueError("python_handler task requires 'output_ref'")
    output_path = _resolve_ref_path(task_root, output_ref)
    input_refs = [str(ref) for ref in task_payload.get("input_refs", [])]

    if operation == "merge_entity_candidates":
        payloads = [_read_json(_resolve_ref_path(task_root, ref)) for ref in input_refs]
        result = _merge_aggregate_payloads(
            payloads,
            max_ambiguities=int(task_payload.get("max_ambiguities", 24)),
        )
    elif operation == "bundle_outputs":
        bundled: List[Dict[str, Any]] = []
        for ref in input_refs:
            child_payload = _read_json(_resolve_ref_path(task_root, ref))
            bundled.extend(_flatten_bundled_outputs(child_payload) if isinstance(child_payload, dict) else [])
        result = {
            "kind": "bundle_outputs",
            "bundle_label": str(task_payload.get("reduce_label", "")).strip(),
            "item_count": len(bundled),
            "outputs": bundled,
        }
    else:
        raise ValueError(f"Unsupported entity-graph local operation '{operation}'")

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
