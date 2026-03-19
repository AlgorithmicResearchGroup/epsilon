import pytest

from agent.models import litellm_client


def _mock_tool_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_bash",
                                "arguments": "{\"script\":\"ls -la\"}",
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_chat_with_tools_retries_on_tool_json_parse_error(monkeypatch):
    calls = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "litellm.BadRequestError: GroqException - "
                "{\"error\":{\"message\":\"Failed to parse tool call arguments as JSON\","
                "\"code\":\"tool_use_failed\"}}"
            )
        return _mock_tool_response()

    monkeypatch.setattr(litellm_client, "completion", _fake_completion)

    result = litellm_client.chat_with_tools(
        model="groq/llama-3.1-8b-instant",
        messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "List files"}],
        tools=[{"type": "function", "function": {"name": "run_bash", "parameters": {"type": "object"}}}],
    )

    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "user"
    assert "tool arguments were not valid JSON" in calls[1]["messages"][-1]["content"]
    assert result["tool_name"] == "run_bash"
    assert result["tool_args"] == {"script": "ls -la"}


def test_chat_with_tools_does_not_retry_without_tools(monkeypatch):
    calls = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Failed to parse tool call arguments as JSON")

    monkeypatch.setattr(litellm_client, "completion", _fake_completion)

    with pytest.raises(RuntimeError):
        litellm_client.chat_with_tools(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
        )

    assert len(calls) == 1


def test_chat_with_tools_does_not_retry_on_unrelated_error(monkeypatch):
    calls = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("upstream connection reset")

    monkeypatch.setattr(litellm_client, "completion", _fake_completion)

    with pytest.raises(RuntimeError):
        litellm_client.chat_with_tools(
            model="groq/llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "run_bash", "parameters": {"type": "object"}}}],
        )

    assert len(calls) == 1


def test_chat_with_tools_retries_on_output_parse_failed(monkeypatch):
    calls = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "litellm.BadRequestError: GroqException - "
                "{\"error\":{\"message\":\"Parsing failed. The model generated output that could not be parsed.\","
                "\"code\":\"output_parse_failed\","
                "\"failed_generation\":\"Now need to call done with summary.\"}}"
            )
        return _mock_tool_response()

    monkeypatch.setattr(litellm_client, "completion", _fake_completion)

    result = litellm_client.chat_with_tools(
        model="groq/llama-3.1-8b-instant",
        messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "finish"}],
        tools=[{"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}],
    )

    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "user"
    assert "exactly one tool call and no prose" in calls[1]["messages"][-1]["content"]
    assert calls[1]["tool_choice"] == "required"
    assert result["tool_name"] == "run_bash"


def test_chat_with_tools_falls_back_to_text_after_repeated_parse_failures(monkeypatch):
    calls = []

    def _fake_completion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            "litellm.BadRequestError: GroqException - "
            "{\"error\":{\"message\":\"Parsing failed.\","
            "\"code\":\"output_parse_failed\","
            "\"failed_generation\":\"Now need to call done with summary.\"}}"
        )

    monkeypatch.setattr(litellm_client, "completion", _fake_completion)

    result = litellm_client.chat_with_tools(
        model="groq/llama-3.1-8b-instant",
        messages=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "finish"}],
        tools=[{"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}],
    )

    assert len(calls) == 2
    assert result["tool_name"] is None
    assert "Now need to call done with summary." in result["assistant_text"]
