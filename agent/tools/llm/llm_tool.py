from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.models.litellm_client import chat_text


llm_tool_definitions = [
    {
        "name": "call_llm",
        "description": "Call an approved delegate LLM (via LiteLLM) in single-shot mode and return text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt sent to the delegate model."},
                "model": {"type": "string", "description": "Optional model override. Must be in delegate allowlist."},
                "system_prompt": {"type": "string", "description": "Optional system instruction for the delegate call."},
                "max_tokens": {"type": "integer", "description": "Optional response token cap (clamped by configured limits)."},
                "temperature": {"type": "number", "description": "Optional temperature (clamped by configured limits)."},
                "timeout_seconds": {"type": "integer", "description": "Optional request timeout in seconds (clamped by configured limits)."},
            },
            "required": ["prompt"],
        },
    }
]


_DEFAULT_DELEGATE_CONFIG = {
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


def _delegate_error(message: str):
    return {
        "tool": "call_llm",
        "status": "error",
        "attempt": "Delegate call failed",
        "stdout": "",
        "stderr": message,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_delegate_config():
    merged = {
        "enabled": _DEFAULT_DELEGATE_CONFIG["enabled"],
        "default_model": _DEFAULT_DELEGATE_CONFIG["default_model"],
        "allowed_models": list(_DEFAULT_DELEGATE_CONFIG["allowed_models"]),
        "limits": dict(_DEFAULT_DELEGATE_CONFIG["limits"]),
    }

    manifest_path = _project_root() / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return merged

    delegate = manifest.get("delegate_llm")
    if not isinstance(delegate, dict):
        return merged

    if "enabled" in delegate:
        merged["enabled"] = bool(delegate["enabled"])

    default_model = delegate.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        merged["default_model"] = default_model.strip()

    allowed_models = delegate.get("allowed_models")
    if isinstance(allowed_models, list):
        cleaned = [m.strip() for m in allowed_models if isinstance(m, str) and m.strip()]
        if cleaned:
            merged["allowed_models"] = cleaned

    limits = delegate.get("limits")
    if isinstance(limits, dict):
        for key in merged["limits"]:
            if key in limits:
                merged["limits"][key] = limits[key]

    if merged["default_model"] not in merged["allowed_models"] and merged["allowed_models"]:
        merged["default_model"] = merged["allowed_models"][0]

    return merged


def _clamp_int(value, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(fallback)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _clamp_float(value, minimum: float, maximum: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def call_llm(arguments, work_dir=None):
    cfg = _load_delegate_config()
    if not cfg.get("enabled", True):
        return _delegate_error("Delegate LLM tool is disabled in manifest delegate_llm.enabled.")

    limits = cfg["limits"]
    prompt = str(arguments.get("prompt", ""))
    if not prompt.strip():
        return _delegate_error("Argument 'prompt' is required.")

    prompt_cap = _clamp_int(
        limits.get("prompt_max_chars"),
        128,
        1_000_000,
        _DEFAULT_DELEGATE_CONFIG["limits"]["prompt_max_chars"],
    )
    if len(prompt) > prompt_cap:
        return _delegate_error(f"Prompt exceeds max length ({len(prompt)} > {prompt_cap} chars).")

    model = str(arguments.get("model") or cfg.get("default_model") or "").strip()
    if not model:
        return _delegate_error("No delegate model resolved.")

    allowed_models = cfg.get("allowed_models", [])
    if model not in allowed_models:
        allowed_str = ", ".join(allowed_models) if allowed_models else "(none configured)"
        return _delegate_error(f"Model '{model}' is not in delegate allowlist: {allowed_str}")

    max_tokens = _clamp_int(
        arguments.get("max_tokens", limits.get("max_tokens_default")),
        _clamp_int(limits.get("max_tokens_min"), 1, 1_000_000, _DEFAULT_DELEGATE_CONFIG["limits"]["max_tokens_min"]),
        _clamp_int(limits.get("max_tokens_max"), 1, 1_000_000, _DEFAULT_DELEGATE_CONFIG["limits"]["max_tokens_max"]),
        _clamp_int(limits.get("max_tokens_default"), 1, 1_000_000, _DEFAULT_DELEGATE_CONFIG["limits"]["max_tokens_default"]),
    )
    temperature = _clamp_float(
        arguments.get("temperature", limits.get("temperature_default")),
        _clamp_float(limits.get("temperature_min"), 0.0, 2.0, _DEFAULT_DELEGATE_CONFIG["limits"]["temperature_min"]),
        _clamp_float(limits.get("temperature_max"), 0.0, 2.0, _DEFAULT_DELEGATE_CONFIG["limits"]["temperature_max"]),
        _clamp_float(limits.get("temperature_default"), 0.0, 2.0, _DEFAULT_DELEGATE_CONFIG["limits"]["temperature_default"]),
    )
    timeout_seconds = _clamp_int(
        arguments.get("timeout_seconds", limits.get("timeout_seconds_default")),
        _clamp_int(limits.get("timeout_seconds_min"), 1, 86_400, _DEFAULT_DELEGATE_CONFIG["limits"]["timeout_seconds_min"]),
        _clamp_int(limits.get("timeout_seconds_max"), 1, 86_400, _DEFAULT_DELEGATE_CONFIG["limits"]["timeout_seconds_max"]),
        _clamp_int(limits.get("timeout_seconds_default"), 1, 86_400, _DEFAULT_DELEGATE_CONFIG["limits"]["timeout_seconds_default"]),
    )

    system_prompt = arguments.get("system_prompt")
    if system_prompt is not None:
        system_prompt = str(system_prompt)

    try:
        result = chat_text(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return _delegate_error(f"Delegate model call failed: {type(exc).__name__}: {exc}")

    text = str(result.get("text", "")).strip()
    if not text:
        return _delegate_error("Delegate model returned empty text.")

    response_cap = _clamp_int(
        limits.get("response_max_chars"),
        128,
        1_000_000,
        _DEFAULT_DELEGATE_CONFIG["limits"]["response_max_chars"],
    )
    if len(text) > response_cap:
        text = text[:response_cap]

    prompt_tokens = int(result.get("prompt_tokens", 0))
    completion_tokens = int(result.get("completion_tokens", 0))
    return {
        "tool": "call_llm",
        "status": "success",
        "attempt": (
            f"Delegate call succeeded using {model} "
            f"({prompt_tokens} in / {completion_tokens} out tokens)."
        ),
        "stdout": text,
        "stderr": "",
    }
