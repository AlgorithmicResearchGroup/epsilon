import io
import json
import sys
from pathlib import Path

from runtime import byoa_function_runner


def _context(workspace: Path) -> str:
    payload = {
        "type": "run_context",
        "task": "build hello",
        "agent_id": "a1",
        "workspace": str(workspace),
        "broker_router": "tcp://localhost:5555",
        "broker_sub": "tcp://localhost:5556",
        "topics": ["general"],
        "pattern": "dag",
        "task_type": "build",
        "role": "builder",
    }
    return json.dumps(payload) + "\n"


def _lines(output: str):
    return [json.loads(line) for line in output.splitlines() if line.strip().startswith("{")]


def test_function_runner_with_string_result(monkeypatch, tmp_path):
    mod = tmp_path / "agent_ok.py"
    mod.write_text(
        "def run(input, **kwargs):\n"
        "    return f\"done:{input['task']}\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_ENTRY", f"{mod}:run")
    monkeypatch.setattr(sys, "stdin", io.StringIO(_context(tmp_path)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    code = byoa_function_runner.run_once()
    assert code == 0

    actions = _lines(out.getvalue())
    assert actions[-1]["action"] == "done"
    assert actions[-1]["summary"] == "done:build hello"


def test_function_runner_with_dict_result(monkeypatch, tmp_path):
    mod = tmp_path / "agent_dict.py"
    mod.write_text(
        "def run(input, **kwargs):\n"
        "    return {'ok': True, 'workspace': input['workspace']}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_ENTRY", f"{mod}:run")
    monkeypatch.setattr(sys, "stdin", io.StringIO(_context(tmp_path)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    code = byoa_function_runner.run_once()
    assert code == 0
    actions = _lines(out.getvalue())
    assert actions[-1]["action"] == "done"
    summary = json.loads(actions[-1]["summary"])
    assert summary["ok"] is True


def test_function_runner_failure(monkeypatch, tmp_path):
    mod = tmp_path / "agent_fail.py"
    mod.write_text(
        "def run(input, **kwargs):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COLLAB_AGENT_ADAPTER_ENTRY", f"{mod}:run")
    monkeypatch.setattr(sys, "stdin", io.StringIO(_context(tmp_path)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    code = byoa_function_runner.run_once()
    assert code == 1
    actions = _lines(out.getvalue())
    assert actions[-1]["action"] == "fail"
    assert "boom" in actions[-1]["error"]
