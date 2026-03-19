#!/usr/bin/env python3
"""Small SDK for external BYOA agents using the RESI adapter protocol."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional


class AdapterSession:
    """Convenience wrapper around the runner's newline-delimited JSON protocol."""

    def __init__(self, context: Dict[str, Any]) -> None:
        self.context = context

    @classmethod
    def from_stdio(cls) -> "AdapterSession":
        line = sys.stdin.readline()
        if not line:
            raise RuntimeError("No run context received from adapter runner.")
        payload = json.loads(line)
        msg_type = payload.get("type")
        if msg_type != "run_context":
            raise RuntimeError(f"Expected run_context, got: {msg_type!r}")
        return cls(payload)

    def _emit(self, action: str, **kwargs: Any) -> None:
        payload = {"action": action, **kwargs}
        print(json.dumps(payload), flush=True)

    def log(self, message: str) -> None:
        self._emit("log", message=message)

    def send_message(
        self,
        content: str,
        topic: str = "general",
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            "send_message",
            content=content,
            topic=topic,
            target=target,
            metadata=metadata or {},
        )

    def check_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        self._emit("check_messages", limit=max(1, int(limit)))
        line = sys.stdin.readline()
        if not line:
            return []
        payload = json.loads(line)
        if payload.get("type") != "messages":
            return []
        messages = payload.get("messages")
        return messages if isinstance(messages, list) else []

    def done(self, summary: str = "") -> None:
        self._emit("done", summary=summary)

    def fail(self, error: str) -> None:
        self._emit("fail", error=error)


def build_run_input(context: Dict[str, Any]) -> Dict[str, Any]:
    """Build a stable high-level input payload for function-style adapters."""
    topics = context.get("topics")
    if not isinstance(topics, list):
        topics = []

    return {
        "task": str(context.get("task", "")),
        "agent_id": str(context.get("agent_id", "")),
        "workspace": str(context.get("workspace", "")),
        "topics": [str(t) for t in topics],
        "broker": {
            "router": str(context.get("broker_router", "")),
            "sub": str(context.get("broker_sub", "")),
        },
        "meta": {
            "pattern": str(context.get("pattern") or ""),
            "task_type": str(context.get("task_type") or ""),
            "role": str(context.get("role") or ""),
        },
    }


def coerce_run_output(result: Any) -> str:
    """Normalize a run() return value into the adapter summary string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list, int, float, bool)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)
