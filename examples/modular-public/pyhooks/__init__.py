from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litellm import completion

from .types import (
    MiddlemanOutput,
    MiddlemanResult,
    MiddlemanSettings,
    RatedOption,
    RatingOption,
    ScoreLogEntry,
    ScoreResult,
    Task,
    UsageInfo,
    UsageLimits,
    UsageSnapshot,
)

os.environ.setdefault("LITELLM_LOG", "ERROR")


@dataclass
class _Runtime:
    task: str = ""
    workspace: str = ""
    model: str = "openai/gpt-5.2"
    token_limit: int = 500000
    time_limit: int = 7200
    started_at: float = field(default_factory=time.time)
    token_usage: int = 0
    submitted: bool = False
    submission: str = ""
    _adapter_session: Any = None
    score_history: List[ScoreLogEntry] = field(default_factory=list)
    tool_errors: List[str] = field(default_factory=list)


_runtime = _Runtime()


class SubmissionComplete(Exception):
    def __init__(self, submission: str) -> None:
        super().__init__("submission completed")
        self.submission = submission


def configure(*, task: str, workspace: str, model: str, adapter_session: Any = None) -> None:
    _runtime.task = task
    _runtime.workspace = workspace
    _runtime.model = model or _runtime.model
    _runtime._adapter_session = adapter_session
    _runtime.started_at = time.time()
    _runtime.token_usage = 0
    _runtime.submitted = False
    _runtime.submission = ""
    _runtime.score_history = []
    _runtime.tool_errors = []


def is_submitted() -> bool:
    return _runtime.submitted


def get_submission() -> str:
    return _runtime.submission


def had_tool_errors() -> bool:
    return len(_runtime.tool_errors) > 0


def tool_error_summary(max_items: int = 5) -> str:
    if not _runtime.tool_errors:
        return ""
    head = _runtime.tool_errors[:max(1, int(max_items))]
    extra = len(_runtime.tool_errors) - len(head)
    summary = "; ".join(head)
    if extra > 0:
        summary += f"; ... (+{extra} more)"
    return summary


def _record_tool_error(message: str) -> None:
    _runtime.tool_errors.append(str(message))


def _resolve_model(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return _runtime.model
    if "/" in model:
        return model
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return f"openai/{model}"
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    return _runtime.model


def _resolve_api_key(model: str) -> Optional[str]:
    explicit = os.environ.get("LLM_API_KEY", "").strip()
    if explicit:
        return explicit

    provider = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC")
    return None


def _usage_from_response(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage", {})
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get("total_tokens") or usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    return int(getattr(usage, "total_tokens", 0) or getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))


def _log_line(message: str) -> None:
    text = str(message)
    if _runtime._adapter_session is not None:
        _runtime._adapter_session.log(text)
    else:
        print(text, flush=True)


class Actions:
    async def run_bash(self, command: str, timeout: int = 120) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=_runtime.workspace or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=max(1, int(timeout)))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout_b, stderr_b = b"", b"Command timed out"
            status = 124
        else:
            status = int(proc.returncode or 0)

        result = {
            "stdout": stdout_b.decode("utf-8", errors="replace"),
            "stderr": stderr_b.decode("utf-8", errors="replace"),
            "status": status,
        }
        if status != 0:
            _record_tool_error(f"bash exit {status}: {command[:200]}")
        return json.dumps(result)

    async def run_python(self, code: str, timeout: int = 120) -> str:
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-c",
            code,
            cwd=_runtime.workspace or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=max(1, int(timeout)))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            _record_tool_error("python timeout")
            return "Python execution timed out"
        if int(proc.returncode or 0) != 0:
            _record_tool_error(f"python exit {proc.returncode}")
        return (stdout_b + stderr_b).decode("utf-8", errors="replace")


class Hooks:
    def log(self, *args: Any) -> None:
        _log_line(" ".join(str(a) for a in args))

    def log_with_attributes(self, _attributes: Dict[str, Any], message: str) -> None:
        _log_line(message)

    async def getTask(self) -> Task:
        return Task(instructions=_runtime.task)

    async def get_usage(self) -> UsageInfo:
        elapsed = int(time.time() - _runtime.started_at)
        return UsageInfo(
            usage=UsageSnapshot(tokens=_runtime.token_usage, total_seconds=elapsed),
            usageLimits=UsageLimits(tokens=_runtime.token_limit, total_seconds=_runtime.time_limit),
        )

    async def action(self, payload: Dict[str, Any]) -> None:
        action_type = payload.get("type", "action")
        _log_line(f"[modular action] {action_type}")

    async def generate(
        self,
        *,
        messages: List[Any],
        settings: MiddlemanSettings,
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> MiddlemanResult:
        model = _resolve_model(settings.model)
        litellm_messages = []
        for m in messages:
            if hasattr(m, "model_dump"):
                litellm_messages.append(m.model_dump())
            elif isinstance(m, dict):
                litellm_messages.append(m)
            else:
                litellm_messages.append(dict(m))

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": litellm_messages,
            "max_tokens": int(settings.max_tokens),
            "temperature": float(settings.temp),
            "n": max(1, int(settings.n)),
            "drop_params": True,
            "timeout": int(os.environ.get("LLM_TIMEOUT_SECONDS", "120")),
            "num_retries": int(os.environ.get("LLM_MAX_RETRIES", "2")),
        }
        if settings.stop:
            kwargs["stop"] = settings.stop
        api_base = os.environ.get("LLM_API_BASE", "").strip()
        if api_base:
            kwargs["api_base"] = api_base
        api_key = _resolve_api_key(model)
        if api_key:
            kwargs["api_key"] = api_key

        if functions:
            kwargs["tools"] = [{"type": "function", "function": fn} for fn in functions]
            kwargs["tool_choice"] = "auto"

        response = completion(**kwargs)
        _runtime.token_usage += _usage_from_response(response)

        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        outputs: List[MiddlemanOutput] = []
        for ch in choices or []:
            message = getattr(ch, "message", None)
            if message is None and isinstance(ch, dict):
                message = ch.get("message", {})
            content = getattr(message, "content", "") if not isinstance(message, dict) else message.get("content", "")
            tool_calls = getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls")
            function_call = None
            if tool_calls:
                first = tool_calls[0]
                fn = getattr(first, "function", None) if not isinstance(first, dict) else first.get("function", {})
                name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
                arguments = getattr(fn, "arguments", "{}") if not isinstance(fn, dict) else fn.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                function_call = {"name": name, "arguments": arguments, "type": "function"}
            outputs.append(MiddlemanOutput(completion=str(content or ""), function_call=function_call))

        return MiddlemanResult(
            outputs=outputs,
            model=model,
            usage={"total_tokens": _usage_from_response(response)},
        )

    async def generate_one(self, settings: MiddlemanSettings, messages: List[Any]) -> str:
        result = await self.generate(messages=messages, settings=settings)
        if not result.outputs:
            return ""
        return result.outputs[0].completion

    async def rate_options(
        self,
        *,
        rating_template: str,
        transcript: str,
        options: List[RatingOption],
        rating_model: str,
    ) -> RatedOption:
        del rating_template, transcript, rating_model
        if not options:
            return RatedOption(action="", rating=0.0)
        best = max(options, key=lambda o: float(o.fixedRating if o.fixedRating is not None else 0.0))
        return RatedOption(action=best.action, fixedRating=best.fixedRating, rating=best.fixedRating)

    async def score(self) -> ScoreResult:
        result = ScoreResult(status="noScore", score=None)
        self.log("Score unavailable in RESI shim runtime.")
        return result

    async def scoreLog(self) -> List[ScoreLogEntry]:
        return list(_runtime.score_history)

    async def submit(self, submission: Any) -> None:
        text = submission if isinstance(submission, str) else json.dumps(submission)
        _runtime.submitted = True
        _runtime.submission = text
        if _runtime._adapter_session is not None:
            _runtime._adapter_session.send_message(
                content=f"modular-public submitted: {text[:2000]}",
                topic="general",
            )
        self.log(f"[submit] {text[:500]}")
        raise SubmissionComplete(text)

    def save_state(self, state: Dict[str, Any]) -> None:
        if not _runtime.workspace:
            return
        path = os.path.join(_runtime.workspace, "modular_state.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            self.log(f"save_state failed: {exc}")

    def main(self, fn):
        asyncio.run(fn())


hooks = Hooks()
actions = Actions()
