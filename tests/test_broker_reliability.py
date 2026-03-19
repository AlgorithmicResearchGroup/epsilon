from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent_protocol.broker import DefaultTopologyManager, LeaseCoordinationEngine
from agent_protocol.messages import Message, MessageType


def _message(agent_id: str, message_type: MessageType, payload: Any = None) -> Message:
    return Message(
        agent_id=agent_id,
        message_type=message_type,
        payload=payload if payload is not None else {},
        topic="tasks",
    )


def _direct_messages(actions: List[Tuple[str, Any]]) -> List[Message]:
    out: List[Message] = []
    for action, payload in actions:
        if action != "direct":
            continue
        _identity, message = payload
        out.append(message)
    return out


def _single_assign(actions: List[Tuple[str, Any]]) -> Dict[str, Any]:
    messages = _direct_messages(actions)
    assigns = [m for m in messages if m.message_type == MessageType.TASK_ASSIGN]
    assert len(assigns) == 1
    return assigns[0].payload


def test_lease_timeout_redelivery_is_bounded_and_dead_lettered(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr("agent_protocol.broker.time.time", lambda: now["value"])

    engine = LeaseCoordinationEngine(
        topology=DefaultTopologyManager(),
        enable_logging=False,
        lease_timeout_seconds=5.0,
        sweep_interval_seconds=0.2,
        max_redeliveries=2,
    )

    worker = b"worker-1"
    submitter = b"submitter"

    engine.on_message(worker, _message("worker-1", MessageType.REGISTER, {"subscribed_topics": ["tasks"]}))
    submit_actions = engine.on_message(
        submitter,
        _message("submitter", MessageType.TASK_SUBMIT, {"job": "compile-part"}),
    )
    task_id = _direct_messages(submit_actions)[0].payload["task_id"]

    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    lease_id = assignment["lease_id"]
    assert assignment["attempt"] == 1

    now["value"] += 6.0
    engine.tick()  # redelivery #1
    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    lease_id = assignment["lease_id"]
    assert assignment["attempt"] == 2
    assert assignment["redelivery_count"] == 1

    now["value"] += 6.0
    engine.tick()  # redelivery #2
    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    lease_id = assignment["lease_id"]
    assert assignment["attempt"] == 3
    assert assignment["redelivery_count"] == 2

    now["value"] += 6.0
    engine.tick()  # should dead-letter (redelivery would exceed max=2)

    task = engine.tasks_by_id[task_id]
    assert task["status"] == "failed"
    assert task["dead_letter_reason"] == "lease_timeout"
    assert task_id in engine.dead_letter_tasks
    assert engine.get_stats()["tasks_dead_lettered"] == 1

    no_task_assign = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    assert no_task_assign["status"] == "no_tasks"


def test_stale_completion_without_active_lease_is_rejected(monkeypatch):
    now = {"value": 2000.0}
    monkeypatch.setattr("agent_protocol.broker.time.time", lambda: now["value"])

    engine = LeaseCoordinationEngine(
        topology=DefaultTopologyManager(),
        enable_logging=False,
        lease_timeout_seconds=5.0,
        sweep_interval_seconds=0.2,
        max_redeliveries=3,
    )

    w1 = b"worker-1"
    w2 = b"worker-2"
    submitter = b"submitter"

    engine.on_message(w1, _message("worker-1", MessageType.REGISTER, {"subscribed_topics": ["tasks"]}))
    engine.on_message(w2, _message("worker-2", MessageType.REGISTER, {"subscribed_topics": ["tasks"]}))
    submit_actions = engine.on_message(submitter, _message("submitter", MessageType.TASK_SUBMIT, {"job": "wiki"}))
    task_id = _direct_messages(submit_actions)[0].payload["task_id"]

    assignment = _single_assign(engine.on_message(w1, _message("worker-1", MessageType.TASK_REQUEST, {})))
    stale_lease_id = assignment["lease_id"]

    now["value"] += 6.0
    engine.tick()  # lease expires, task requeued

    stale_complete_actions = engine.on_message(
        w1,
        _message(
            "worker-1",
            MessageType.TASK_COMPLETE,
            {"task_id": task_id, "lease_id": stale_lease_id, "result": {"ok": True}},
        ),
    )
    stale_ack = _direct_messages(stale_complete_actions)[0]
    assert stale_ack.payload["status"] == "lease_not_found"
    assert engine.tasks_by_id[task_id]["status"] == "pending"

    reassignment = _single_assign(engine.on_message(w2, _message("worker-2", MessageType.TASK_REQUEST, {})))
    complete_actions = engine.on_message(
        w2,
        _message(
            "worker-2",
            MessageType.TASK_COMPLETE,
            {"task_id": task_id, "lease_id": reassignment["lease_id"], "result": {"ok": True}},
        ),
    )
    complete_ack = _direct_messages(complete_actions)[0]
    assert complete_ack.payload["status"] == "task_completed"
    assert engine.tasks_by_id[task_id]["status"] == "completed"


def test_task_payload_can_override_max_redeliveries(monkeypatch):
    now = {"value": 3000.0}
    monkeypatch.setattr("agent_protocol.broker.time.time", lambda: now["value"])

    engine = LeaseCoordinationEngine(
        topology=DefaultTopologyManager(),
        enable_logging=False,
        lease_timeout_seconds=5.0,
        sweep_interval_seconds=0.2,
        max_redeliveries=0,
    )

    worker = b"worker-1"
    submitter = b"submitter"

    engine.on_message(worker, _message("worker-1", MessageType.REGISTER, {"subscribed_topics": ["tasks"]}))
    submit_actions = engine.on_message(
        submitter,
        _message("submitter", MessageType.TASK_SUBMIT, {"job": "x", "max_redeliveries": 1}),
    )
    task_id = _direct_messages(submit_actions)[0].payload["task_id"]

    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    assert assignment["attempt"] == 1

    now["value"] += 6.0
    engine.tick()  # allowed by task override
    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    assert assignment["attempt"] == 2
    assert assignment["redelivery_count"] == 1

    now["value"] += 6.0
    engine.tick()  # exceeds task override => dead-letter

    task = engine.tasks_by_id[task_id]
    assert task["status"] == "failed"
    assert task_id in engine.dead_letter_tasks


def test_task_fail_is_retried_then_marked_failed(monkeypatch):
    now = {"value": 4000.0}
    monkeypatch.setattr("agent_protocol.broker.time.time", lambda: now["value"])

    engine = LeaseCoordinationEngine(
        topology=DefaultTopologyManager(),
        enable_logging=False,
        lease_timeout_seconds=5.0,
        sweep_interval_seconds=0.2,
        max_redeliveries=5,
        max_fail_retries=1,
    )

    worker = b"worker-1"
    submitter = b"submitter"

    engine.on_message(worker, _message("worker-1", MessageType.REGISTER, {"subscribed_topics": ["tasks"]}))
    submit_actions = engine.on_message(
        submitter,
        _message("submitter", MessageType.TASK_SUBMIT, {"job": "transient"}),
    )
    task_id = _direct_messages(submit_actions)[0].payload["task_id"]

    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    fail_actions = engine.on_message(
        worker,
        _message(
            "worker-1",
            MessageType.TASK_FAIL,
            {"task_id": task_id, "lease_id": assignment["lease_id"], "error": {"msg": "transient"}},
        ),
    )
    fail_ack = _direct_messages(fail_actions)[0]
    assert fail_ack.payload["status"] == "task_requeued"
    assert engine.tasks_by_id[task_id]["status"] == "pending"

    assignment = _single_assign(engine.on_message(worker, _message("worker-1", MessageType.TASK_REQUEST, {})))
    fail_actions = engine.on_message(
        worker,
        _message(
            "worker-1",
            MessageType.TASK_FAIL,
            {"task_id": task_id, "lease_id": assignment["lease_id"], "error": {"msg": "still failing"}},
        ),
    )
    fail_ack = _direct_messages(fail_actions)[0]
    assert fail_ack.payload["status"] == "task_failed"
    assert engine.tasks_by_id[task_id]["status"] == "failed"
    stats = engine.get_stats()
    assert stats["tasks_retried_after_fail"] == 1
    assert stats["tasks_failed"] == 1
