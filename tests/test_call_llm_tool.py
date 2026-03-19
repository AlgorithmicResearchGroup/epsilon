from __future__ import annotations

import copy

import agent.tools.llm.llm_tool as llm_tool


def _base_cfg() -> dict:
    return {
        "enabled": True,
        "default_model": "openai/gpt-5.2",
        "allowed_models": ["openai/gpt-5.2", "anthropic/claude-opus-4-6"],
        "limits": {
            "max_tokens_default": 512,
            "max_tokens_min": 32,
            "max_tokens_max": 1024,
            "temperature_default": 0.0,
            "temperature_min": 0.0,
            "temperature_max": 0.7,
            "timeout_seconds_default": 30,
            "timeout_seconds_min": 5,
            "timeout_seconds_max": 60,
            "prompt_max_chars": 12000,
            "response_max_chars": 12000,
        },
    }


def test_call_llm_success_uses_default_model(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))

    captured = {}

    def _fake_chat_text(**kwargs):
        captured.update(kwargs)
        return {"text": "delegate answer", "prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(llm_tool, "chat_text", _fake_chat_text)

    out = llm_tool.call_llm({"prompt": "summarize this"})
    assert out["status"] == "success"
    assert out["stdout"] == "delegate answer"
    assert captured["model"] == "openai/gpt-5.2"


def test_call_llm_success_explicit_allowed_model(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(
        llm_tool,
        "chat_text",
        lambda **kwargs: {"text": "ok", "prompt_tokens": 1, "completion_tokens": 1},
    )

    out = llm_tool.call_llm(
        {"prompt": "analyze", "model": "anthropic/claude-opus-4-6"}
    )
    assert out["status"] == "success"
    assert out["stdout"] == "ok"


def test_call_llm_rejects_disallowed_model(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))

    out = llm_tool.call_llm({"prompt": "x", "model": "openai/gpt-4o-mini"})
    assert out["status"] == "error"
    assert "allowlist" in out["stderr"]


def test_call_llm_rejects_prompt_over_limit(monkeypatch):
    cfg = _base_cfg()
    cfg["limits"]["prompt_max_chars"] = 200
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(
        llm_tool,
        "chat_text",
        lambda **kwargs: {"text": "should_not_run", "prompt_tokens": 0, "completion_tokens": 0},
    )

    out = llm_tool.call_llm({"prompt": "x" * 201})
    assert out["status"] == "error"
    assert "max length" in out["stderr"]


def test_call_llm_clamps_limits(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))

    captured = {}

    def _fake_chat_text(**kwargs):
        captured.update(kwargs)
        return {"text": "ok", "prompt_tokens": 2, "completion_tokens": 3}

    monkeypatch.setattr(llm_tool, "chat_text", _fake_chat_text)

    out = llm_tool.call_llm(
        {
            "prompt": "do a thing",
            "max_tokens": 99_999,
            "temperature": 5.0,
            "timeout_seconds": 0,
        }
    )

    assert out["status"] == "success"
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.7
    assert captured["timeout_seconds"] == 5


def test_call_llm_returns_error_on_delegate_exception(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))

    def _raise(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_tool, "chat_text", _raise)
    out = llm_tool.call_llm({"prompt": "hello"})
    assert out["status"] == "error"
    assert "boom" in out["stderr"]


def test_call_llm_returns_error_on_empty_text(monkeypatch):
    cfg = _base_cfg()
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(
        llm_tool,
        "chat_text",
        lambda **kwargs: {"text": "", "prompt_tokens": 0, "completion_tokens": 0},
    )

    out = llm_tool.call_llm({"prompt": "hello"})
    assert out["status"] == "error"
    assert "empty text" in out["stderr"]


def test_call_llm_respects_disabled_flag(monkeypatch):
    cfg = _base_cfg()
    cfg["enabled"] = False
    monkeypatch.setattr(llm_tool, "_load_delegate_config", lambda: copy.deepcopy(cfg))

    out = llm_tool.call_llm({"prompt": "hello"})
    assert out["status"] == "error"
    assert "disabled" in out["stderr"]
