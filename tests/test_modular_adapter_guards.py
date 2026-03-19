from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "examples" / "byoa" / "modular_public_adapter.py"
MODULAR_REPO = ROOT / "examples" / "modular-public"


def _load_adapter_module():
    spec = importlib.util.spec_from_file_location("modular_public_adapter_test", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_required_outputs_parses_file_paths():
    mod = _load_adapter_module()
    task = """
REQUIRED OUTPUTS:
- SPEC.md
- tests/test_cli.py
- (derive explicit outputs from task)

ACCEPTANCE CHECKS:
- done
"""
    outputs = mod._extract_required_outputs(task)
    assert outputs == ["SPEC.md", "tests/test_cli.py"]


def test_missing_outputs_checks_workspace(tmp_path: Path):
    mod = _load_adapter_module()
    (tmp_path / "SPEC.md").write_text("ok", encoding="utf-8")
    missing = mod._missing_outputs(str(tmp_path), ["SPEC.md", "README.md"])
    assert missing == ["README.md"]


def test_pyhooks_submit_raises_submission_complete():
    sys.path.insert(0, str(MODULAR_REPO))
    import pyhooks  # type: ignore

    pyhooks.configure(task="x", workspace=str(MODULAR_REPO), model="openai/gpt-5.2")
    with pytest.raises(pyhooks.SubmissionComplete):
        asyncio.run(pyhooks.hooks.submit("done"))


def test_run_entrypoint_forwards_task_and_workspace(tmp_path: Path, monkeypatch):
    mod = _load_adapter_module()
    captured = {}

    async def _fake_run_modular(*, task: str, workspace: str, session=None):
        captured["task"] = task
        captured["workspace"] = workspace
        captured["session"] = session
        return "ok"

    monkeypatch.setattr(mod, "_run_modular", _fake_run_modular)
    result = mod.run({"task": "ship it", "workspace": str(tmp_path)})
    assert result == "ok"
    assert captured["task"] == "ship it"
    assert captured["workspace"] == str(tmp_path)
    assert captured["session"] is None
