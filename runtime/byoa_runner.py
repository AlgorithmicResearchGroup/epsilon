#!/usr/bin/env python3
"""Runtime bridge for Bring-Your-Own-Agent (BYOA) subprocesses.

External agent contract:
- Input: one JSON line on stdin with {"type":"run_context", ...}
- Output: newline-delimited JSON commands on stdout, e.g.
  {"action":"log","message":"starting"}
  {"action":"send_message","content":"ready","topic":"general"}
  {"action":"check_messages","limit":25}
  {"action":"done","summary":"completed"}
  {"action":"fail","error":"something broke"}

Plain non-JSON stdout lines are logged as-is.

Adapter selection:
- COLLAB_AGENT_ADAPTER_ENTRY="<module_or_file>:<run_function>" (preferred)
- COLLAB_AGENT_ADAPTER_CMD="python3 your_adapter.py" (advanced/low-level)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from agent_protocol.agent import Agent as ProtocolAgent
from agent_protocol.messages import Message

load_dotenv()


def _get_task_description() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return os.environ.get("TASK_DESCRIPTION", "").strip()


def _read_adapter_command() -> Tuple[List[str], str]:
    entry = os.environ.get("COLLAB_AGENT_ADAPTER_ENTRY", "").strip()
    raw = os.environ.get("COLLAB_AGENT_ADAPTER_CMD", "").strip()
    if entry:
        return [sys.executable, "-m", "runtime.byoa_function_runner"], f"function:{entry}"
    if not raw:
        raise RuntimeError(
            "Adapter mode requires either COLLAB_AGENT_ADAPTER_ENTRY "
            "('<module_or_file>:<run_function>') or COLLAB_AGENT_ADAPTER_CMD."
        )
    try:
        return shlex.split(raw), "command"
    except Exception as exc:
        raise RuntimeError(f"Failed to parse COLLAB_AGENT_ADAPTER_CMD: {exc}") from exc


def _message_to_dict(message: Message) -> Dict[str, Any]:
    return {
        "agent_id": message.agent_id,
        "topic": message.topic,
        "target": message.target,
        "message_type": str(message.message_type.value),
        "payload": message.payload,
        "metadata": message.metadata or {},
        "timestamp": message.timestamp,
    }


class AdapterBridge:
    def __init__(self) -> None:
        self.task = _get_task_description()
        if not self.task:
            raise RuntimeError(
                "Usage: python runtime/byoa_runner.py '<task description>' "
                "or set TASK_DESCRIPTION env var."
            )

        self.adapter_cmd, self.adapter_source = _read_adapter_command()
        self.agent_id = os.environ.get("AGENT_ID", f"adapter-{int(time.time() * 1000)}")
        self.workspace = os.environ.get("SHARED_WORKSPACE", "")
        self.topics = [t.strip() for t in os.environ.get("AGENT_TOPICS", "general").split(",") if t.strip()]

        self.broker_router = os.environ.get("BROKER_ROUTER", os.environ.get("BROKER_PUSH", "tcp://localhost:5555"))
        self.broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
        self.heartbeat_interval = float(os.environ.get("PROTOCOL_HEARTBEAT_INTERVAL_SECONDS", "5"))

        self._message_buffer: Deque[Message] = deque()
        self._message_lock = threading.Lock()
        self._done = False
        self._failed = False
        self._failure_reason = ""

        self.protocol_agent = ProtocolAgent(
            agent_id=self.agent_id,
            broker_router=self.broker_router,
            broker_sub=self.broker_sub,
            topics=self.topics,
            message_handler=self._on_message,
            enable_logging=False,
            heartbeat_interval_seconds=self.heartbeat_interval,
            heartbeat_enabled=True,
        )

    def _on_message(self, message: Message) -> None:
        with self._message_lock:
            self._message_buffer.append(message)

    def _drain_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        drained: List[Dict[str, Any]] = []
        with self._message_lock:
            while self._message_buffer and len(drained) < max(1, limit):
                drained.append(_message_to_dict(self._message_buffer.popleft()))
        return drained

    def _write_child_line(self, proc: subprocess.Popen[str], payload: Dict[str, Any]) -> None:
        if not proc.stdin:
            return
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return

    def _handle_action(self, proc: subprocess.Popen[str], action_payload: Dict[str, Any]) -> None:
        action = str(action_payload.get("action", "")).strip().lower()
        if not action:
            return

        if action == "log":
            message = str(action_payload.get("message", "")).strip()
            if message:
                print(f"[ADAPTER:{self.agent_id}] {message}", flush=True)
            return

        if action == "send_message":
            content = action_payload.get("content", action_payload.get("message", ""))
            topic = str(action_payload.get("topic", "general")).strip() or "general"
            target = action_payload.get("target")
            metadata = action_payload.get("metadata")
            if isinstance(metadata, dict):
                metadata_dict: Optional[Dict[str, Any]] = metadata
            else:
                metadata_dict = None
            self.protocol_agent.send_message(
                payload=content,
                topic=topic,
                target=str(target).strip() if target else None,
                metadata=metadata_dict,
            )
            return

        if action == "check_messages":
            try:
                limit = int(action_payload.get("limit", 50))
            except Exception:
                limit = 50
            messages = self._drain_messages(limit=limit)
            self._write_child_line(proc, {"type": "messages", "messages": messages})
            return

        if action == "submit_task":
            self.protocol_agent.submit_task(action_payload.get("payload", {}))
            return

        if action == "request_task":
            self.protocol_agent.request_task()
            return

        if action == "renew_task":
            task_id = str(action_payload.get("task_id", "")).strip()
            lease_id = str(action_payload.get("lease_id", "")).strip()
            if task_id and lease_id:
                self.protocol_agent.renew_task(task_id, lease_id)
            return

        if action == "complete_task":
            task_id = str(action_payload.get("task_id", "")).strip()
            if task_id:
                self.protocol_agent.complete_task(
                    task_id,
                    result=action_payload.get("result"),
                    lease_id=action_payload.get("lease_id"),
                )
            return

        if action == "fail_task":
            task_id = str(action_payload.get("task_id", "")).strip()
            if task_id:
                self.protocol_agent.fail_task(
                    task_id,
                    error=action_payload.get("error"),
                    lease_id=action_payload.get("lease_id"),
                )
            return

        if action == "done":
            summary = str(action_payload.get("summary", "")).strip()
            if summary:
                print(f"[ADAPTER:{self.agent_id}] done: {summary}", flush=True)
            self._done = True
            return

        if action == "fail":
            self._failed = True
            self._failure_reason = str(action_payload.get("error", "Adapter reported failure")).strip()
            print(f"[ADAPTER:{self.agent_id}] failure: {self._failure_reason}", flush=True)
            return

        print(
            f"[ADAPTER:{self.agent_id}] ignored unknown action: {action}",
            flush=True,
        )

    def run(self) -> int:
        print(f"[BYOA] Agent: {self.agent_id}", flush=True)
        print(f"[BYOA] Adapter source: {self.adapter_source}", flush=True)
        print(f"[BYOA] Adapter command: {' '.join(self.adapter_cmd)}", flush=True)
        if self.workspace:
            print(f"[BYOA] Workspace: {self.workspace}", flush=True)

        self.protocol_agent.start()
        time.sleep(0.2)

        adapter_env = os.environ.copy()
        adapter_env["RESI_AGENT_ID"] = self.agent_id
        adapter_env["RESI_TASK"] = self.task
        adapter_env["RESI_WORKSPACE"] = self.workspace
        adapter_env["RESI_BROKER_ROUTER"] = self.broker_router
        adapter_env["RESI_BROKER_SUB"] = self.broker_sub
        adapter_env["RESI_TOPICS"] = ",".join(self.topics)
        existing_pythonpath = adapter_env.get("PYTHONPATH", "").strip()
        adapter_env["PYTHONPATH"] = (
            f"{PROJECT_ROOT}:{existing_pythonpath}" if existing_pythonpath else PROJECT_ROOT
        )

        proc = subprocess.Popen(
            self.adapter_cmd,
            cwd=PROJECT_ROOT,
            env=adapter_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        self._write_child_line(
            proc,
            {
                "type": "run_context",
                "task": self.task,
                "agent_id": self.agent_id,
                "workspace": self.workspace,
                "broker_router": self.broker_router,
                "broker_sub": self.broker_sub,
                "topics": self.topics,
                "pattern": os.environ.get("COLLAB_PATTERN", ""),
                "task_type": os.environ.get("AGENT_TASK_TYPE", ""),
                "role": os.environ.get("AGENT_ROLE", ""),
            },
        )

        try:
            if not proc.stdout:
                raise RuntimeError("Adapter subprocess has no stdout stream.")

            for line in proc.stdout:
                text = line.rstrip("\n")
                if not text:
                    continue

                parsed: Optional[Dict[str, Any]] = None
                if text.startswith("{") and text.endswith("}"):
                    try:
                        candidate = json.loads(text)
                        if isinstance(candidate, dict):
                            parsed = candidate
                    except json.JSONDecodeError:
                        parsed = None

                if parsed is None:
                    print(f"[ADAPTER:{self.agent_id}] {text}", flush=True)
                    continue

                self._handle_action(proc, parsed)
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            self.protocol_agent.stop()

        if self._failed:
            print(f"[BYOA] Adapter failed: {self._failure_reason}", flush=True)
            return 1

        if proc.returncode != 0:
            print(f"[BYOA] Adapter exited with code {proc.returncode}", flush=True)
            return int(proc.returncode)

        if self._done:
            print("[BYOA] Adapter marked task done.", flush=True)
        else:
            print("[BYOA] Adapter exited cleanly without explicit done.", flush=True)
        return 0


def main() -> None:
    try:
        bridge = AdapterBridge()
        code = bridge.run()
    except Exception as exc:
        print(f"[BYOA] Fatal: {exc}", flush=True)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
