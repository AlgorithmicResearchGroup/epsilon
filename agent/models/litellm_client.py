import json
import os
import re
from typing import Any, Dict, List, Optional

os.environ.setdefault("LITELLM_LOG", "ERROR")
from litellm import completion


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    chunks.append(text)
                continue
            text = str(_get(item, "text", "")).strip()
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    return str(content or "")


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = _get(response, "usage", {})
    prompt_tokens = int(_get(usage, "prompt_tokens", 0) or _get(usage, "input_tokens", 0) or 0)
    completion_tokens = int(_get(usage, "completion_tokens", 0) or _get(usage, "output_tokens", 0) or 0)
    total_tokens = int(_get(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _extract_first_tool_call(response: Any) -> Dict[str, Any]:
    choices = _get(response, "choices", []) or []
    if not choices:
        return {"tool_name": None, "tool_args": None, "tool_call_id": None, "assistant_text": ""}

    message = _get(choices[0], "message", {})
    tool_calls = _get(message, "tool_calls", []) or []
    assistant_text = _message_content_to_text(_get(message, "content", ""))

    if not tool_calls:
        return {
            "tool_name": None,
            "tool_args": None,
            "tool_call_id": None,
            "assistant_text": assistant_text,
        }

    tool_call = tool_calls[0]
    fn = _get(tool_call, "function", {})
    tool_name = _get(fn, "name")
    raw_args = _get(fn, "arguments", "{}")

    if isinstance(raw_args, dict):
        parsed_args = raw_args
    else:
        try:
            parsed_args = json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Unable to parse tool args JSON for '{tool_name}': {exc}") from exc

    return {
        "tool_name": tool_name,
        "tool_args": parsed_args,
        "tool_call_id": _get(tool_call, "id"),
        "assistant_text": assistant_text,
    }


def _extract_assistant_text(response: Any) -> str:
    choices = _get(response, "choices", []) or []
    if not choices:
        return ""
    message = _get(choices[0], "message", {})
    return _message_content_to_text(_get(message, "content", ""))


def _looks_like_tool_json_parse_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if not text:
        return False
    markers = (
        "failed to parse tool call arguments as json",
        "tool_use_failed",
        "invalid tool call arguments",
        "tool call arguments",
    )
    return any(marker in text for marker in markers) and "json" in text


def _looks_like_output_parse_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    if not text:
        return False
    markers = (
        "output_parse_failed",
        "parsing failed",
        "could not be parsed",
        "failed_generation",
    )
    return "parse" in text and any(marker in text for marker in markers)


def _extract_failed_generation(exc: Exception) -> str:
    text = str(exc or "")
    if not text:
        return ""
    match = re.search(r'"failed_generation"\s*:\s*"((?:\\.|[^"])*)"', text, flags=re.DOTALL)
    if not match:
        return ""
    raw_value = match.group(1)
    try:
        return json.loads(f"\"{raw_value}\"")
    except Exception:
        return raw_value


def _append_retry_guidance(messages: List[Dict[str, Any]], guidance: str) -> List[Dict[str, Any]]:
    retry_messages = list(messages)
    if retry_messages and _get(retry_messages[-1], "role") == "user":
        last_content = str(_get(retry_messages[-1], "content", ""))
        if guidance in last_content:
            return retry_messages
    retry_messages.append({"role": "user", "content": guidance})
    return retry_messages


def _build_tool_json_retry_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    guidance = (
        "Your previous tool call failed because tool arguments were not valid JSON. "
        "Retry now by calling exactly one tool with strictly valid JSON arguments. "
        "For run_bash, use a compact single-line command and do not use heredoc syntax "
        "(<<EOF, <<'PY'). If multi-line code is needed, write a file first, then execute it."
    )
    return _append_retry_guidance(messages, guidance)


def _build_output_parse_retry_messages(
    messages: List[Dict[str, Any]], tool_choice: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    forced_tool_name = None
    if isinstance(tool_choice, dict):
        fn = _get(tool_choice, "function", {})
        forced_tool_name = _get(fn, "name")
    guidance = (
        "Your previous response could not be parsed by the provider. "
        "Reply with exactly one tool call and no prose."
    )
    if forced_tool_name:
        guidance += f" Call only `{forced_tool_name}`."
    return _append_retry_guidance(messages, guidance)


def _synthetic_text_response(text: str) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": text or "", "tool_calls": []}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _resolve_api_key(model: str) -> Optional[str]:
    # Explicit override for custom/proxy setups.
    llm_key = os.environ.get("LLM_API_KEY", "").strip()
    if llm_key:
        return llm_key

    provider = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC")
    if provider == "groq":
        return os.environ.get("GROQ_API_KEY")

    if provider:
        provider_token = "".join(ch if ch.isalnum() else "_" for ch in provider).strip("_").upper()
        for candidate in (
            f"{provider_token}_API_KEY",
            f"{provider_token}_KEY",
            f"{provider_token}_TOKEN",
        ):
            value = os.environ.get(candidate, "").strip()
            if value:
                return value

    openai_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI") or "").strip()
    if openai_key:
        return openai_key
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC") or "").strip()
    if anthropic_key:
        return anthropic_key
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        return groq_key
    return None


def chat_with_tools(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tool_choice: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: bool = False,
) -> Dict[str, Any]:
    timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "2"))

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout_seconds,
        "num_retries": max_retries,
        "drop_params": True,
    }

    api_base = os.environ.get("LLM_API_BASE", "").strip()
    if api_base:
        kwargs["api_base"] = api_base

    api_key = _resolve_api_key(model)
    if api_key:
        kwargs["api_key"] = api_key

    if tools:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    try:
        response = completion(**kwargs)
    except Exception as exc:
        if not tools:
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs["num_retries"] = 0

        if _looks_like_tool_json_parse_error(exc):
            retry_kwargs["messages"] = _build_tool_json_retry_messages(messages)
        elif _looks_like_output_parse_error(exc):
            retry_kwargs["messages"] = _build_output_parse_retry_messages(messages, tool_choice=tool_choice)
            if tool_choice is None:
                retry_kwargs["tool_choice"] = "required"
        else:
            raise

        try:
            response = completion(**retry_kwargs)
        except Exception as retry_exc:
            if _looks_like_tool_json_parse_error(retry_exc) or _looks_like_output_parse_error(retry_exc):
                failed_generation = _extract_failed_generation(retry_exc) or _extract_failed_generation(exc)
                fallback_text = failed_generation or str(retry_exc)
                response = _synthetic_text_response(fallback_text)
            else:
                raise

    extracted = _extract_first_tool_call(response)
    usage = _extract_usage(response)

    return {
        "response": response,
        **extracted,
        **usage,
    }


def chat_text(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Dict[str, Any]:
    resolved_timeout = int(timeout_seconds) if timeout_seconds is not None else int(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
    resolved_retries = int(max_retries) if max_retries is not None else int(os.environ.get("LLM_MAX_RETRIES", "2"))

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "timeout": resolved_timeout,
        "num_retries": resolved_retries,
        "drop_params": True,
    }

    api_base = os.environ.get("LLM_API_BASE", "").strip()
    if api_base:
        kwargs["api_base"] = api_base

    api_key = _resolve_api_key(model)
    if api_key:
        kwargs["api_key"] = api_key

    response = completion(**kwargs)
    usage = _extract_usage(response)
    text = _extract_assistant_text(response)
    return {
        "response": response,
        "text": text,
        **usage,
    }
