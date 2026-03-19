#!/usr/bin/env python3
"""RESI BYOA adapter for the poking-agents modular-public agent.

Preferred:
  COLLAB_AGENT_MODE=adapter \
  COLLAB_AGENT_ADAPTER_ENTRY="examples/byoa/modular_public_adapter.py:run" \
  python3 orchestrate.py --pattern dag "create hello.txt with hello world"

Legacy command mode (still supported):
  COLLAB_AGENT_MODE=adapter \
  COLLAB_AGENT_ADAPTER_CMD="python3 examples/byoa/modular_public_adapter.py" \
  python3 orchestrate.py --pattern dag "create hello.txt with hello world"
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.byoa_sdk import AdapterSession


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "modular-public"


FILE_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".sql", ".go", ".rs",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".dart",
    ".ipynb", ".lock",
}


def _extract_required_outputs(task_text: str) -> list[str]:
    marker = "REQUIRED OUTPUTS:"
    start = task_text.find(marker)
    if start < 0:
        return []
    start += len(marker)
    end = task_text.find("ACCEPTANCE CHECKS:", start)
    section = task_text[start:end if end >= 0 else len(task_text)]

    candidates: list[str] = []
    seen: set[str] = set()
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        item = line.lstrip("-").strip()
        if not item or item.lower().startswith("(derive explicit outputs"):
            continue

        fragments = re.findall(r"`([^`]+)`", item) or [item]
        for frag in fragments:
            for token in re.findall(r"[A-Za-z0-9_./-]+", frag):
                t = token.strip().strip(".,;:")
                if not t or "*" in t:
                    continue
                looks_like_file = "/" in t or Path(t).suffix.lower() in FILE_EXTENSIONS
                if not looks_like_file:
                    continue
                if t not in seen:
                    seen.add(t)
                    candidates.append(t)
    return candidates


def _missing_outputs(workspace: str, required_outputs: list[str]) -> list[str]:
    missing: list[str] = []
    root = Path(workspace)
    for rel in required_outputs:
        path = root / rel
        if rel.endswith("/"):
            if not path.is_dir():
                missing.append(rel)
        else:
            if not path.exists():
                missing.append(rel)
    return missing


async def _run_modular(
    *,
    task: str,
    workspace: str,
    session: Optional[AdapterSession] = None,
) -> str:
    repo_root = _repo_root()
    if not repo_root.exists():
        raise RuntimeError(f"modular-public not found at: {repo_root}")

    # Ensure local shim package and modular modules resolve first.
    sys.path.insert(0, str(repo_root))

    import pyhooks  # type: ignore
    from base import Agent, Settings, State, hooks  # type: ignore
    from modules import actors, discriminators, generators, prompters, tools  # type: ignore

    task = str(task or "").strip()
    workspace = str(workspace or os.getcwd()).strip()
    model = os.environ.get("AGENT_MODEL", "openai/gpt-5.2")
    max_steps = int(os.environ.get("MODULAR_MAX_STEPS", "24"))
    required_outputs = _extract_required_outputs(task)

    os.makedirs(workspace, exist_ok=True)
    os.chdir(workspace)

    pyhooks.configure(
        task=task,
        workspace=workspace,
        model=model,
        adapter_session=session,
    )

    settings = Settings(
        toolkit=os.environ.get("MODULAR_TOOLKIT", "_basic"),
        prompter=os.environ.get("MODULAR_PROMPTER", "_basic"),
        generator=os.environ.get("MODULAR_GENERATOR", "_gpt_basic_1x4o"),
        discriminator=os.environ.get("MODULAR_DISCRIMINATOR", "_basic"),
        actor=os.environ.get("MODULAR_ACTOR", "_basic"),
        autosubmit=False,
    )

    state = State(
        task_string=task,
        next_step={"module_type": "prompter", "args": {}},
        token_limit=500000,
        timeout=600,
    )
    agent = Agent(state=state, settings=settings, toolkit_dict={})

    hooks.log(f"modular-public adapter starting | max_steps={max_steps}")
    if required_outputs:
        hooks.log(f"required outputs: {', '.join(required_outputs)}")
    for step in range(max_steps):
        toolkit_dict = getattr(tools, agent.settings.toolkit)
        agent.set_toolkit_dict(toolkit_dict)

        try:
            module_type = agent.state.next_step["module_type"]
            if module_type == "prompter":
                await getattr(prompters, agent.settings.prompter)(agent)
            elif module_type == "generator":
                await getattr(generators, agent.settings.generator)(agent)
            elif module_type == "discriminator":
                await getattr(discriminators, agent.settings.discriminator)(agent)
            elif module_type == "actor":
                await getattr(actors, agent.settings.actor)(agent)
            else:
                raise RuntimeError(f"Invalid module type: {module_type}")
        except pyhooks.SubmissionComplete:
            if required_outputs:
                missing = _missing_outputs(workspace, required_outputs)
                if missing:
                    raise RuntimeError(
                        "submit called but required outputs are missing: "
                        + ", ".join(missing)
                    )
            return pyhooks.get_submission() or "submitted"

        usage_info = await hooks.get_usage()
        agent.state.token_usage = usage_info.usage.tokens
        agent.state.token_limit = usage_info.usageLimits.tokens
        agent.state.time_usage = usage_info.usage.total_seconds
        agent.state.time_limit = usage_info.usageLimits.total_seconds
        hooks.save_state({"state": agent.state.model_dump(), "settings": agent.settings.model_dump()})

        if pyhooks.is_submitted():
            if required_outputs:
                missing = _missing_outputs(workspace, required_outputs)
                if missing:
                    raise RuntimeError(
                        "submit called but required outputs are missing: "
                        + ", ".join(missing)
                    )
            return pyhooks.get_submission() or "submitted"

    if pyhooks.had_tool_errors():
        raise RuntimeError(
            "tool command failures without explicit submit: "
            + pyhooks.tool_error_summary()
        )

    if required_outputs:
        missing = _missing_outputs(workspace, required_outputs)
        if missing:
            raise RuntimeError(
                "required outputs missing without explicit submit: "
                + ", ".join(missing)
            )

    # Fallback if no explicit submit call happened.
    if agent.state.nodes:
        last = agent.state.nodes[-1].message
        text = (last.content or "").strip()
        if text:
            return text[:4000]
    return "completed without explicit submit"


def run(
    input: Dict[str, Any],
    *,
    session: Optional[AdapterSession] = None,
    **_kwargs: Any,
) -> str:
    """HAL-style entrypoint: run(input) -> output."""
    task = str(input.get("task", "")).strip()
    workspace = str(input.get("workspace", os.getcwd())).strip()
    return asyncio.run(_run_modular(task=task, workspace=workspace, session=session))


def main() -> None:
    session = AdapterSession.from_stdio()
    try:
        summary = run(session.context, session=session)
    except Exception as exc:
        session.fail(f"modular-public adapter failed: {exc}")
        return
    session.done(summary)


if __name__ == "__main__":
    main()
