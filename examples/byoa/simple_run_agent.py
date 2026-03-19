#!/usr/bin/env python3
"""Simple function-style BYOA adapter.

Run with:
  COLLAB_AGENT_MODE=adapter \
  COLLAB_AGENT_ADAPTER_ENTRY="examples/byoa/simple_run_agent.py:run" \
  python3 orchestrate.py --pattern dag "Write hello.txt with hello world"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run(input: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
    task = str(input.get("task", ""))
    workspace = Path(str(input.get("workspace", ".")))
    workspace.mkdir(parents=True, exist_ok=True)

    output_path = workspace / "byoa_output.txt"
    output_path.write_text(f"Task: {task}\nhello world\n", encoding="utf-8")

    return {
        "status": "ok",
        "artifact": output_path.name,
        "message": "wrote byoa_output.txt",
    }
