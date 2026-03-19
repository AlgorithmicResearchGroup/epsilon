import io
import json
import sys

from runtime.byoa_sdk import AdapterSession, build_run_input, coerce_run_output


def test_from_stdio_reads_run_context(monkeypatch):
    payload = {
        "type": "run_context",
        "task": "test task",
        "agent_id": "a1",
        "workspace": "/tmp/work",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
    session = AdapterSession.from_stdio()
    assert session.context["task"] == "test task"
    assert session.context["agent_id"] == "a1"


def test_check_messages_round_trip(monkeypatch):
    context = {"type": "run_context", "task": "x", "agent_id": "a2", "workspace": "/tmp/work"}
    reply = {
        "type": "messages",
        "messages": [{"agent_id": "peer", "payload": "hello"}],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps(context) + "\n" + json.dumps(reply) + "\n"),
    )
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    session = AdapterSession.from_stdio()
    messages = session.check_messages(limit=3)

    assert messages == [{"agent_id": "peer", "payload": "hello"}]
    first_line = out.getvalue().splitlines()[0]
    request = json.loads(first_line)
    assert request["action"] == "check_messages"
    assert request["limit"] == 3


def test_emit_helpers(monkeypatch):
    context = {"type": "run_context", "task": "x", "agent_id": "a3", "workspace": "/tmp/work"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(context) + "\n"))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    session = AdapterSession.from_stdio()
    session.log("hello")
    session.send_message("ready")
    session.done("ok")

    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0] == {"action": "log", "message": "hello"}
    assert lines[1]["action"] == "send_message"
    assert lines[1]["content"] == "ready"
    assert lines[2] == {"action": "done", "summary": "ok"}


def test_build_run_input_shapes_payload():
    payload = build_run_input(
        {
            "task": "do x",
            "agent_id": "a1",
            "workspace": "/tmp/w",
            "topics": ["general", "dev"],
            "broker_router": "tcp://localhost:5555",
            "broker_sub": "tcp://localhost:5556",
            "pattern": "dag",
            "task_type": "build",
            "role": "builder",
        }
    )
    assert payload["task"] == "do x"
    assert payload["broker"]["router"] == "tcp://localhost:5555"
    assert payload["meta"]["task_type"] == "build"


def test_coerce_run_output():
    assert coerce_run_output("ok") == "ok"
    assert coerce_run_output({"a": 1}) == '{"a": 1}'
    assert coerce_run_output(None) == ""
