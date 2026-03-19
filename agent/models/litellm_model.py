import json
import tiktoken

from agent.models.litellm_client import chat_with_tools
from agent.utils import tool_schema_to_openai


class LiteLLMModel:
    def __init__(self, system_prompt, all_tools, model_name="openai/gpt-5.2", max_tokens=4096):
        self.model_name = model_name
        self.response_max_tokens = max_tokens
        self.context_window = 200000
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.oai_tools = [tool_schema_to_openai(tool) for tool in all_tools]
        self.last_tool_call_id = None

        self.messages = [{"role": "system", "content": system_prompt}]

    def initial_request(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        return self._send_and_process()

    def send_tool_result(self, content):
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": self.last_tool_call_id,
                "content": content,
            }
        )
        return self._send_and_process()

    def send_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        return self._send_and_process()

    def _send_and_process(self):
        self._truncate_if_needed()

        result = chat_with_tools(
            model=self.model_name,
            messages=self.messages,
            tools=self.oai_tools,
            max_tokens=self.response_max_tokens,
            temperature=0,
            parallel_tool_calls=False,
        )

        tool_name = result["tool_name"]
        tool_args = result["tool_args"]
        tool_call_id = result["tool_call_id"]
        assistant_text = result["assistant_text"]

        if tool_name is not None and not tool_call_id:
            tool_call_id = f"tool_{len(self.messages)}"

        assistant_message = {
            "role": "assistant",
            "content": assistant_text,
        }
        if tool_name is not None:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }
            ]

        self.messages.append(assistant_message)
        self.last_tool_call_id = tool_call_id

        return (
            tool_name,
            tool_args if tool_name is not None else assistant_text,
            result["total_tokens"],
            result["prompt_tokens"],
            result["completion_tokens"],
        )

    def _estimate_tokens(self):
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            elif content is None:
                content = ""
            total += len(self.encoding.encode(str(content), disallowed_special=()))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += len(self.encoding.encode(json.dumps(tool_calls), disallowed_special=()))
        return total

    def _truncate_if_needed(self):
        budget = self.context_window - self.response_max_tokens - 500
        total = self._estimate_tokens()

        if total <= budget:
            return

        preserve_start = 2
        min_tail = 8

        while total > budget and len(self.messages) > preserve_start + min_tail + 1:
            for _ in range(2):
                removed = self.messages.pop(preserve_start)
                content = removed.get("content", "") or ""
                if isinstance(content, list):
                    content = json.dumps(content)
                total -= len(self.encoding.encode(str(content), disallowed_special=()))
                tool_calls = removed.get("tool_calls")
                if tool_calls:
                    total -= len(self.encoding.encode(json.dumps(tool_calls), disallowed_special=()))
