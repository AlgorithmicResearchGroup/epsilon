#!/usr/bin/env python3
"""High-level BYOA adapter wrapper for HAL-style run(input) callables."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import os
import traceback
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict

from runtime.byoa_sdk import AdapterSession, build_run_input, coerce_run_output


def _read_entrypoint() -> str:
    entry = os.environ.get("COLLAB_AGENT_ADAPTER_ENTRY", "").strip()
    if not entry:
        raise RuntimeError(
            "COLLAB_AGENT_ADAPTER_ENTRY is required for function adapter mode "
            "(expected '<module_or_file>:<run_function>')."
        )
    if ":" not in entry:
        raise RuntimeError(
            "Invalid COLLAB_AGENT_ADAPTER_ENTRY format. "
            "Expected '<module_or_file>:<run_function>'."
        )
    return entry


def _load_module(module_ref: str) -> ModuleType:
    if (
        module_ref.endswith(".py")
        or os.path.sep in module_ref
        or (os.path.altsep and os.path.altsep in module_ref)
        or module_ref.startswith(".")
    ):
        file_path = Path(module_ref).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        if not file_path.exists():
            raise RuntimeError(f"Adapter entry file not found: {file_path}")
        module_name = f"resi_byoa_{file_path.stem}_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Unable to load adapter module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    module_ref, function_name = entrypoint.rsplit(":", 1)
    module_ref = module_ref.strip()
    function_name = function_name.strip()
    if not module_ref or not function_name:
        raise RuntimeError(
            "Invalid COLLAB_AGENT_ADAPTER_ENTRY format. "
            "Expected '<module_or_file>:<run_function>'."
        )

    module = _load_module(module_ref)
    run_fn = getattr(module, function_name, None)
    if run_fn is None:
        raise RuntimeError(
            f"Adapter function '{function_name}' not found in '{module_ref}'."
        )
    if not callable(run_fn):
        raise RuntimeError(
            f"Adapter symbol '{function_name}' from '{module_ref}' is not callable."
        )
    return run_fn


def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def invoke_run_function(
    run_fn: Callable[..., Any],
    run_input: Dict[str, Any],
    session: AdapterSession,
) -> Any:
    sig = inspect.signature(run_fn)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: Dict[str, Any] = {}
    if "session" in params or accepts_kwargs:
        kwargs["session"] = session
    if "context" in params or accepts_kwargs:
        kwargs["context"] = session.context

    positional_params = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional_params and not any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        raise RuntimeError("Adapter run function must accept the run input as the first argument.")

    result = run_fn(run_input, **kwargs)
    return _maybe_await(result)


def run_once() -> int:
    try:
        session = AdapterSession.from_stdio()
    except Exception as exc:
        print(f"function adapter failed to read run context: {exc}", flush=True)
        return 1

    try:
        entrypoint = _read_entrypoint()
        run_fn = load_entrypoint(entrypoint)
        run_input = build_run_input(session.context)
        result = invoke_run_function(run_fn, run_input, session)
        session.done(coerce_run_output(result))
        return 0
    except Exception as exc:
        session.fail(f"function adapter failed: {exc}")
        print(traceback.format_exc(), flush=True)
        return 1


def main() -> None:
    raise SystemExit(run_once())


if __name__ == "__main__":
    main()
