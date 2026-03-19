#!/usr/bin/env python3
"""Example BYOA adapter agent.

Run with:
  COLLAB_AGENT_MODE=adapter \
  COLLAB_AGENT_ADAPTER_CMD="python3 examples/byoa/while_loop_agent.py" \
  python3 orchestrate.py --pattern dag "Write hello.txt with hello world"
"""

from __future__ import annotations

import os
import time

from runtime.byoa_sdk import AdapterSession


def main() -> None:
    session = AdapterSession.from_stdio()
    task = session.context.get("task", "")
    workspace = session.context.get("workspace", os.getcwd())

    session.log(f"while-loop adapter started | workspace={workspace}")

    output_path = os.path.join(workspace, "byoa_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task}\n")
        f.write("hello world\n")

    session.send_message(
        content="Created byoa_output.txt with task echo + hello world.",
        topic="general",
    )

    # Optional: poll for inbound protocol messages.
    for _ in range(2):
        messages = session.check_messages(limit=10)
        if messages:
            session.log(f"received {len(messages)} message(s)")
        time.sleep(0.25)

    session.done("wrote byoa_output.txt")


if __name__ == "__main__":
    main()
