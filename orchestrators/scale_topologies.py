"""Helpers for large-scale manifest-backed queue topologies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from orchestrators.queue_runtime import QueueNodeSpec


ALLOWED_EXECUTORS = {"agent", "direct_wiki", "local_reduce", "python_handler"}


class ManifestValidationError(ValueError):
    """Raised when a scale-topology manifest is malformed."""


@dataclass(frozen=True)
class ItemSpec:
    item_id: str
    input_ref: str
    payload: Dict[str, Any]
    input_text: str = ""
    output_ref: str = ""


@dataclass(frozen=True)
class ManifestSpec:
    pattern: str
    task_type: str
    output_root: str
    items: List[ItemSpec]
    map_task_template: str
    reduce_task_template: str
    shard_count: Optional[int]
    reduce_arity: int
    map_executor: str
    reduce_executor: str
    map_payload: Dict[str, Any] = field(default_factory=dict)
    reduce_payload: Dict[str, Any] = field(default_factory=dict)


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"Manifest field '{key}' must be a non-empty string.")
    return value.strip()


def _normalize_output_root(value: str) -> str:
    cleaned = value.strip().strip("/")
    if not cleaned:
        raise ManifestValidationError("Manifest field 'output_root' must not be empty.")
    return cleaned


def _normalize_executor(value: Any, field_name: str, *, default: str) -> str:
    if value is None:
        return default
    token = str(value).strip().lower()
    if token not in ALLOWED_EXECUTORS:
        valid = ", ".join(sorted(ALLOWED_EXECUTORS))
        raise ManifestValidationError(f"Manifest field '{field_name}' must be one of: {valid}")
    return token


def load_task_manifest(path: str, pattern: str) -> ManifestSpec:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestValidationError("Task manifest must be a JSON object.")

    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ManifestValidationError("Manifest field 'items' must be a non-empty array.")

    seen_ids = set()
    items: List[ItemSpec] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise ManifestValidationError(f"Manifest item #{index} must be an object.")
        item_id = _require_string(raw_item, "id")
        if item_id in seen_ids:
            raise ManifestValidationError(f"Duplicate item id '{item_id}' in manifest.")
        seen_ids.add(item_id)
        payload = raw_item.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ManifestValidationError(f"Manifest item '{item_id}' field 'payload' must be an object.")
        input_ref = str(raw_item.get("input_ref", "") or "").strip()
        input_text = str(raw_item.get("input_text", "") or "")
        output_ref = str(raw_item.get("output_ref", "") or "").strip()
        items.append(
            ItemSpec(
                item_id=item_id,
                input_ref=input_ref,
                input_text=input_text,
                payload=payload,
                output_ref=output_ref,
            )
        )

    output_root = _normalize_output_root(_require_string(data, "output_root"))
    map_task_template = _require_string(data, "map_task_template")
    reduce_task_template = str(data.get("reduce_task_template", "") or "").strip()

    if pattern == "map_reduce" and not reduce_task_template:
        raise ManifestValidationError("map_reduce manifests require 'reduce_task_template'.")

    shard_count_value = data.get("shard_count")
    if shard_count_value is None or shard_count_value == "":
        shard_count = None
    else:
        try:
            shard_count = int(shard_count_value)
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError("Manifest field 'shard_count' must be an integer.") from exc
        if shard_count < 1:
            raise ManifestValidationError("Manifest field 'shard_count' must be >= 1.")

    reduce_arity_value = data.get("reduce_arity", 8)
    try:
        reduce_arity = int(reduce_arity_value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError("Manifest field 'reduce_arity' must be an integer.") from exc
    if reduce_arity < 2:
        raise ManifestValidationError("Manifest field 'reduce_arity' must be >= 2.")

    map_payload = data.get("map_payload", {})
    if map_payload is None:
        map_payload = {}
    if not isinstance(map_payload, dict):
        raise ManifestValidationError("Manifest field 'map_payload' must be an object.")

    reduce_payload = data.get("reduce_payload", {})
    if reduce_payload is None:
        reduce_payload = {}
    if not isinstance(reduce_payload, dict):
        raise ManifestValidationError("Manifest field 'reduce_payload' must be an object.")

    return ManifestSpec(
        pattern=pattern,
        task_type=str(data.get("task_type", pattern) or pattern),
        output_root=output_root,
        items=items,
        map_task_template=map_task_template,
        reduce_task_template=reduce_task_template,
        shard_count=shard_count,
        reduce_arity=reduce_arity,
        map_executor=_normalize_executor(data.get("map_executor"), "map_executor", default="agent"),
        reduce_executor=_normalize_executor(data.get("reduce_executor"), "reduce_executor", default="agent"),
        map_payload=map_payload,
        reduce_payload=reduce_payload,
    )


def _template_fields(template: str) -> List[str]:
    return [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]


def render_task_template(template: str, context: Mapping[str, Any]) -> str:
    try:
        return template.format_map(_TemplateContext(context))
    except KeyError as exc:
        fields = ", ".join(sorted(_template_fields(template)))
        raise ManifestValidationError(
            f"Template references unknown field '{exc.args[0]}'. Available fields: {fields}"
        ) from exc


class _TemplateContext(dict):
    def __missing__(self, key: str) -> Any:
        raise KeyError(key)


def build_template_context(
    *,
    item: Optional[ItemSpec] = None,
    output_ref: str,
    shard_id: str = "",
    child_output_refs: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "output_ref": output_ref,
        "shard_id": shard_id,
        "child_output_refs_json": json.dumps(list(child_output_refs or [])),
        "child_output_refs_text": "\n".join(child_output_refs or []),
        "child_count": len(child_output_refs or []),
    }
    if item is not None:
        context.update(
            {
                "item_id": item.item_id,
                "input_ref": item.input_ref,
                "input_text": item.input_text,
                "payload_json": json.dumps(item.payload, sort_keys=True),
            }
        )
        for key, value in item.payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                context[f"payload_{key}"] = value
    if extra:
        for key, value in extra.items():
            context[key] = value
    return context


def assign_items_to_shards(items: Sequence[ItemSpec], shard_count: int) -> Dict[str, List[ItemSpec]]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    actual_shards = min(shard_count, max(1, len(items)))
    shards: Dict[str, List[ItemSpec]] = {f"shard-{idx + 1:03d}": [] for idx in range(actual_shards)}
    shard_ids = list(shards.keys())
    for index, item in enumerate(items):
        shard_id = shard_ids[index % actual_shards]
        shards[shard_id].append(item)
    return shards


def _default_map_output_ref(output_root: str, item: ItemSpec) -> str:
    if item.output_ref:
        return item.output_ref
    return f"{output_root}/maps/{item.item_id}.json"


def build_sharded_queue_nodes(
    manifest: ManifestSpec,
    *,
    shard_count: int,
    shared_workspace: str,
) -> Tuple[List[QueueNodeSpec], str, Dict[str, Any]]:
    shards = assign_items_to_shards(manifest.items, shard_count)
    nodes: List[QueueNodeSpec] = []
    shard_reduce_ids: List[str] = []

    for shard_id, items in shards.items():
        item_node_ids: List[str] = []
        item_output_refs: List[str] = []
        for item in items:
            output_ref = _default_map_output_ref(manifest.output_root, item)
            item_output_refs.append(output_ref)
            node_id = f"map-{item.item_id}"
            item_node_ids.append(node_id)
            payload = build_map_payload(
                manifest=manifest,
                item=item,
                output_ref=output_ref,
                shared_workspace=shared_workspace,
                shard_id=shard_id,
            )
            nodes.append(
                QueueNodeSpec(
                    node_id=node_id,
                    role=f"Map item {item.item_id}",
                    task_type="map",
                    payload=payload,
                    kind="map",
                )
            )

        reduce_node_id = f"reduce-{shard_id}"
        reduce_output_ref = f"{manifest.output_root}/shards/{shard_id}.json"
        shard_reduce_ids.append(reduce_node_id)
        nodes.append(
            QueueNodeSpec(
                node_id=reduce_node_id,
                role=f"Shard reducer for {shard_id}",
                task_type="reduce",
                payload=build_reduce_payload(
                    manifest=manifest,
                    output_ref=reduce_output_ref,
                    shared_workspace=shared_workspace,
                    child_output_refs=item_output_refs,
                    reduce_label=shard_id,
                ),
                depends_on=item_node_ids,
                kind="reduce",
            )
        )

    final_output_ref = f"{manifest.output_root}/final-summary.json"
    nodes.append(
        QueueNodeSpec(
            node_id="reduce-final",
            role="Final shard merge",
            task_type="reduce",
            payload=build_reduce_payload(
                manifest=manifest,
                output_ref=final_output_ref,
                shared_workspace=shared_workspace,
                child_output_refs=[f"{manifest.output_root}/shards/{shard_id}.json" for shard_id in shards],
                reduce_label="final",
            ),
            depends_on=shard_reduce_ids,
            kind="reduce",
        )
    )
    metadata = {
        "pattern": "sharded_queue",
        "shard_count": len(shards),
        "shards": {shard_id: [item.item_id for item in items] for shard_id, items in shards.items()},
    }
    return nodes, final_output_ref, metadata


def build_map_reduce_nodes(
    manifest: ManifestSpec,
    *,
    shared_workspace: str,
) -> Tuple[List[QueueNodeSpec], str, Dict[str, Any]]:
    nodes: List[QueueNodeSpec] = []
    current_level: List[Tuple[str, str]] = []

    for item in manifest.items:
        output_ref = _default_map_output_ref(manifest.output_root, item)
        node_id = f"map-{item.item_id}"
        current_level.append((node_id, output_ref))
        nodes.append(
            QueueNodeSpec(
                node_id=node_id,
                role=f"Map item {item.item_id}",
                task_type="map",
                payload=build_map_payload(
                    manifest=manifest,
                    item=item,
                    output_ref=output_ref,
                    shared_workspace=shared_workspace,
                ),
                kind="map",
            )
        )

    reduce_levels = 0
    while len(current_level) > 1:
        reduce_levels += 1
        next_level: List[Tuple[str, str]] = []
        for group_index in range(0, len(current_level), manifest.reduce_arity):
            group = current_level[group_index : group_index + manifest.reduce_arity]
            child_ids = [node_id for node_id, _ in group]
            child_output_refs = [output_ref for _, output_ref in group]
            reduce_node_id = f"reduce-l{reduce_levels:02d}-n{(group_index // manifest.reduce_arity) + 1:04d}"
            reduce_output_ref = f"{manifest.output_root}/reduce/level-{reduce_levels:02d}/{reduce_node_id}.json"
            next_level.append((reduce_node_id, reduce_output_ref))
            nodes.append(
                QueueNodeSpec(
                    node_id=reduce_node_id,
                    role=f"Reducer level {reduce_levels}",
                    task_type="reduce",
                    payload=build_reduce_payload(
                        manifest=manifest,
                        output_ref=reduce_output_ref,
                        shared_workspace=shared_workspace,
                        child_output_refs=child_output_refs,
                        reduce_label=reduce_node_id,
                    ),
                    depends_on=child_ids,
                    kind="reduce",
                )
            )
        current_level = next_level

    final_output_ref = current_level[0][1]
    metadata = {
        "pattern": "map_reduce",
        "reduce_arity": manifest.reduce_arity,
        "reduce_levels": reduce_levels,
        "map_count": len(manifest.items),
        "reduce_count": len([node for node in nodes if node.kind == "reduce"]),
    }
    return nodes, final_output_ref, metadata


def build_map_payload(
    *,
    manifest: ManifestSpec,
    item: ItemSpec,
    output_ref: str,
    shared_workspace: str,
    shard_id: str = "",
) -> Dict[str, Any]:
    if manifest.map_executor == "local_reduce":
        return build_local_reduce_payload(
            description=f"Local map for {item.item_id}",
            output_ref=output_ref,
            shared_workspace=shared_workspace,
            input_ref=item.input_ref,
            input_text=item.input_text,
            payload=item.payload,
            reduce_label=shard_id or item.item_id,
            operation="map",
        )

    context = build_template_context(
        item=item,
        output_ref=output_ref,
        shard_id=shard_id,
        extra={"task_type": manifest.task_type},
    )
    instructions = render_task_template(manifest.map_task_template, context)
    return {
        "description": f"Map item {item.item_id}",
        "executor": manifest.map_executor,
        "instructions": instructions,
        "output_ref": output_ref,
        "input_ref": item.input_ref,
        "shared_workspace": shared_workspace,
        "manifest_item_id": item.item_id,
        "shard_id": shard_id,
        **manifest.map_payload,
    }


def build_reduce_payload(
    *,
    manifest: ManifestSpec,
    output_ref: str,
    shared_workspace: str,
    child_output_refs: Sequence[str],
    reduce_label: str,
) -> Dict[str, Any]:
    if manifest.reduce_executor == "local_reduce":
        return build_local_reduce_payload(
            description=f"Local reduce for {reduce_label}",
            output_ref=output_ref,
            shared_workspace=shared_workspace,
            input_refs=list(child_output_refs),
            reduce_label=reduce_label,
            operation="reduce",
        )

    context = build_template_context(
        output_ref=output_ref,
        child_output_refs=child_output_refs,
        extra={"reduce_label": reduce_label, "task_type": manifest.task_type},
    )
    instructions = render_task_template(manifest.reduce_task_template, context)
    return {
        "description": f"Reduce group {reduce_label}",
        "executor": manifest.reduce_executor,
        "instructions": instructions,
        "output_ref": output_ref,
        "input_refs": list(child_output_refs),
        "shared_workspace": shared_workspace,
        "reduce_label": reduce_label,
        **manifest.reduce_payload,
    }


def build_local_reduce_payload(
    *,
    description: str,
    output_ref: str,
    shared_workspace: str,
    input_ref: str = "",
    input_text: str = "",
    payload: Optional[Dict[str, Any]] = None,
    input_refs: Optional[Sequence[str]] = None,
    reduce_label: str = "",
    operation: str = "reduce",
) -> Dict[str, Any]:
    return {
        "description": description,
        "executor": "local_reduce",
        "operation": operation,
        "output_ref": output_ref,
        "input_ref": input_ref,
        "input_text": input_text,
        "payload": payload or {},
        "input_refs": list(input_refs or []),
        "shared_workspace": shared_workspace,
        "reduce_label": reduce_label,
    }


def manifest_to_dict(spec: ManifestSpec) -> Dict[str, Any]:
    return {
        **asdict(spec),
        "items": [asdict(item) for item in spec.items],
    }


def build_local_reduce_items(task_count: int) -> List[ItemSpec]:
    items: List[ItemSpec] = []
    for index in range(task_count):
        item_id = f"item-{index + 1:05d}"
        items.append(
            ItemSpec(
                item_id=item_id,
                input_ref="",
                input_text=(
                    f"{item_id} is a deterministic local reduce benchmark sample. "
                    "It exists to exercise map and reduce queue topologies without external network calls."
                ),
                payload={"ordinal": index + 1},
            )
        )
    return items
