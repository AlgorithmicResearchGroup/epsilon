from __future__ import annotations

import json

import pytest

from orchestrators.patterns import resolve_pattern
from orchestrators.scale_topologies import (
    ManifestSpec,
    ManifestValidationError,
    assign_items_to_shards,
    build_local_reduce_items,
    build_map_reduce_nodes,
    load_task_manifest,
)
from runtime.worker_daemon import WorkerDaemon


def test_pattern_registry_exposes_scale_patterns():
    assert resolve_pattern("sharded_queue").pattern == "sharded_queue"
    assert resolve_pattern("map-reduce").pattern == "map_reduce"


def test_load_task_manifest_requires_reduce_template_for_map_reduce(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_root": "results",
                "map_task_template": "write {output_ref}",
                "items": [{"id": "one", "input_ref": "input.txt"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError):
        load_task_manifest(str(manifest_path), "map_reduce")


def test_assign_items_to_shards_balances_items():
    items = build_local_reduce_items(5)
    shards = assign_items_to_shards(items, 2)

    assert set(shards.keys()) == {"shard-001", "shard-002"}
    assert [item.item_id for item in shards["shard-001"]] == ["item-00001", "item-00003", "item-00005"]
    assert [item.item_id for item in shards["shard-002"]] == ["item-00002", "item-00004"]


def test_build_map_reduce_nodes_creates_hierarchical_reducers(tmp_path):
    manifest = ManifestSpec(
        pattern="map_reduce",
        task_type="local_reduce",
        output_root="results",
        items=build_local_reduce_items(10),
        map_task_template="unused",
        reduce_task_template="unused",
        shard_count=None,
        reduce_arity=3,
        map_executor="local_reduce",
        reduce_executor="local_reduce",
    )

    nodes, final_output_ref, metadata = build_map_reduce_nodes(
        manifest,
        shared_workspace=str(tmp_path),
    )

    map_nodes = [node for node in nodes if node.kind == "map"]
    reduce_nodes = [node for node in nodes if node.kind == "reduce"]

    assert len(map_nodes) == 10
    assert len(reduce_nodes) == 7
    assert metadata["reduce_levels"] == 3
    assert final_output_ref.endswith(".json")


def test_local_reduce_executor_maps_and_reduces(tmp_path):
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.work_root = tmp_path

    first = WorkerDaemon._execute_local_reduce_task(
        daemon,
        "task-1",
        {
            "benchmark_id": "bench",
            "operation": "map",
            "shared_workspace": str(tmp_path),
            "output_ref": "results/map-1.json",
            "input_text": "alpha beta gamma",
            "payload": {"ordinal": 1},
        },
    )
    second = WorkerDaemon._execute_local_reduce_task(
        daemon,
        "task-2",
        {
            "benchmark_id": "bench",
            "operation": "map",
            "shared_workspace": str(tmp_path),
            "output_ref": "results/map-2.json",
            "input_text": "delta epsilon",
            "payload": {"ordinal": 2},
        },
    )
    reduced = WorkerDaemon._execute_local_reduce_task(
        daemon,
        "task-3",
        {
            "benchmark_id": "bench",
            "operation": "reduce",
            "shared_workspace": str(tmp_path),
            "output_ref": "results/final.json",
            "input_refs": ["results/map-1.json", "results/map-2.json"],
            "reduce_label": "final",
        },
    )

    assert first["output_exists"] is True
    assert second["output_exists"] is True
    assert reduced["output_exists"] is True

    final_payload = json.loads((tmp_path / "results" / "final.json").read_text(encoding="utf-8"))
    assert final_payload["child_count"] == 2
    assert final_payload["total_word_count"] == 5


def test_agent_executor_requires_declared_output_file(tmp_path, monkeypatch):
    daemon = WorkerDaemon.__new__(WorkerDaemon)
    daemon.work_root = tmp_path
    daemon.worker_id = "worker-1"
    daemon.default_executor = "agent"
    daemon.default_max_iterations = 1
    daemon.default_max_runtime_seconds = 1
    daemon.default_agent_model = None
    daemon._completed_count = 0

    class _FakeAgent:
        def __init__(self) -> None:
            self.failed = None
            self.completed = None
            self.sent = None

        def fail_task(self, task_id, error, lease_id=None):
            self.failed = (task_id, error, lease_id)

        def complete_task(self, task_id, result, lease_id=None):
            self.completed = (task_id, result, lease_id)

        def send_data(self, payload, topic):
            self.sent = (payload, topic)

    daemon._agent = _FakeAgent()

    monkeypatch.setattr(
        WorkerDaemon,
        "_execute_agent_task",
        lambda self, task_id, task_payload: {
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_path": "",
            "output_exists": False,
            "output_preview": "",
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )

    daemon._process_assignment(
        {
            "task_id": "task-1",
            "payload": {
                "benchmark_id": "bench",
                "executor": "agent",
                "task_type": "map",
                "client_task_id": "client-1",
                "output_ref": "results/out.json",
            },
        }
    )

    assert daemon._agent.failed is not None
    sent_payload, _topic = daemon._agent.sent
    assert sent_payload["status"] == "failure"
