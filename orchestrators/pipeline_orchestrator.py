#!/usr/bin/env python3
"""
Pipeline orchestrator: decomposes a task into ordered stages, runs each stage
sequentially, then applies QA/fix waves on the integrated output.

Usage:
  python orchestrators/pipeline_orchestrator.py "Build a URL shortener microservice"
  python orchestrators/pipeline_orchestrator.py --prompts challenge_prompts.json --prompt 2
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from agent_protocol.broker import MessageBroker
from agent.models.litellm_client import chat_with_tools
from agent.utils import tool_schema_to_openai
from orchestrators.patterns import resolve_pattern
from orchestrators.dag_orchestrator import (
    _broker_port_from_endpoint,
    _cleanup_docker_runtime,
    _ensure_docker_available,
    _ensure_docker_image,
    _setup_docker_runtime,
    _truthy,
    build_agent_task,
    build_fix_defs,
    build_qa_task,
    call_assign_fixes,
    read_qa_report,
    run_wave,
)


PIPELINE_DECOMPOSE_TOOL = {
    "name": "decompose_pipeline",
    "description": "Decompose a user task into an ordered pipeline of executable stages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short lowercase stage id (e.g. contract, backend, integration).",
                        },
                        "role": {
                            "type": "string",
                            "description": "One-line role summary for the stage.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Detailed stage instructions with exact outputs and file targets.",
                        },
                        "inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Inputs expected at stage start.",
                        },
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete artifacts produced by this stage.",
                        },
                        "acceptance": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Objective checks for stage completion.",
                        },
                    },
                    "required": ["id", "role", "task"],
                },
            },
        },
        "required": ["stages"],
    },
}

PIPELINE_SYSTEM_PROMPT = """\
You are a technical project planner building a linear delivery pipeline.

Create an ordered sequence of 3-10 stages. Stages execute strictly in order.
Each stage must leave the workspace in a usable state for the next stage.

Rules:
1. Return only stages that are necessary for delivery.
2. Use unique lowercase stage IDs.
3. Provide concrete tasks with exact files, interfaces, and validations.
4. Include explicit outputs and acceptance checks per stage.
5. Keep early stages focused on contracts/foundations, middle on implementation,
   and final stages on integration/verification.
6. Do not create a dedicated QA stage; QA is handled by the orchestrator loop.

Call the decompose_pipeline tool."""


def call_pipeline(model_name: str, task: str) -> Dict[str, Any]:
    result = chat_with_tools(
        model=model_name,
        messages=[
            {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this task into pipeline stages:\n\n{task}"},
        ],
        tools=[tool_schema_to_openai(PIPELINE_DECOMPOSE_TOOL)],
        tool_choice={"type": "function", "function": {"name": "decompose_pipeline"}},
        max_tokens=4096,
        temperature=0,
    )
    if result["tool_name"] != "decompose_pipeline" or not isinstance(result["tool_args"], dict):
        raise RuntimeError(
            f"Pipeline decomposition call returned no valid decompose_pipeline tool call. "
            f"tool={result['tool_name']} text={result['assistant_text']}"
        )
    return result["tool_args"]


def _to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        items = []
        for item in value:
            text = str(item or "").strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _stage_task_type(index: int, total: int, stage_id: str, role: str, task: str) -> str:
    haystack = f"{stage_id} {role} {task}".lower()
    if index == 0 or any(k in haystack for k in ("contract", "schema", "interface", "spec")):
        return "contract"
    if index == total - 1 or any(k in haystack for k in ("verify", "integration", "validate", "test")):
        return "verify"
    return "build"


def normalize_pipeline(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = result.get("stages") if isinstance(result, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ValueError("Pipeline decomposition produced no stages.")

    normalized: List[Dict[str, Any]] = []
    used_ids = set()
    previous_id = None
    total = len(raw)

    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Stage at index {idx} is not an object.")

        base_id = str(item.get("id", f"stage_{idx}")).strip().lower() or f"stage_{idx}"
        safe_id = re.sub(r"[^a-z0-9_]+", "_", base_id).strip("_") or f"stage_{idx}"
        stage_id = safe_id
        suffix = 2
        while stage_id in used_ids:
            stage_id = f"{safe_id}_{suffix}"
            suffix += 1
        used_ids.add(stage_id)

        role = str(item.get("role", "Pipeline contributor")).strip() or "Pipeline contributor"
        task = str(item.get("task", "")).strip()
        if not task:
            raise ValueError(f"Stage '{stage_id}' has no task instructions.")

        depends_on = [previous_id] if previous_id else []
        previous_id = stage_id

        normalized.append(
            {
                "id": stage_id,
                "task_type": _stage_task_type(idx - 1, total, stage_id, role, task),
                "role": role,
                "task": task,
                "inputs": _to_string_list(item.get("inputs")),
                "outputs": _to_string_list(item.get("outputs")),
                "acceptance": _to_string_list(item.get("acceptance")),
                "depends_on": depends_on,
            }
        )

    return normalized


def _print_workspace(shared_dir: str) -> None:
    print(f"\n  Shared workspace contents:")
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            print(f"    {rel} ({size} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline multi-agent orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all agents/broker")
    parser.add_argument("--docker-image", help="Docker image to use when --executor=docker")
    parser.add_argument("--pattern", help="Optional pattern label; must resolve to PIPELINE for this entrypoint")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
        idx = (args.prompt or 1) - 1
        entry = prompts[idx]
        print(f"Loaded prompt #{idx + 1}: {entry['name']}")
        args.task = entry["prompt"]

    if not args.task:
        parser.error("Provide a task string or use --prompts/--prompt.")

    if args.pattern:
        resolved = resolve_pattern(args.pattern).pattern
        if resolved != "pipeline":
            parser.error(
                f"pipeline_orchestrator.py only supports PIPELINE pattern, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved

    return args


def main():
    args = parse_args()
    task = args.task
    shared_dir_arg = args.shared_dir

    manifest_path = os.path.join(PROJECT_ROOT, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]
    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL", settings["model"])
    agent_model = os.environ.get("AGENT_MODEL", settings["model"])
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "40"))
    max_runtime = os.environ.get("MAX_RUNTIME_SECONDS", "300")
    max_waves = int(os.environ.get("MAX_WAVES", "3"))
    run_indefinitely = _truthy(os.environ.get("RESI_RUN_INDEFINITELY"))
    qa_enabled = run_indefinitely or max_waves > 0
    qa_iterations = int(os.environ.get("QA_ITERATIONS", "30"))
    fix_iterations = int(os.environ.get("FIX_ITERATIONS", "15"))
    fix_runtime = os.environ.get("FIX_RUNTIME_SECONDS", "120")
    executor = (args.executor or os.environ.get("COLLAB_EXECUTOR", "host")).strip().lower()
    docker_image = args.docker_image or os.environ.get("COLLAB_DOCKER_IMAGE", "epsilon")

    print("=" * 70)
    print("  PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Pattern: {os.environ.get('COLLAB_PATTERN', 'pipeline')}")
    print(f"Orchestrator model: {orchestrator_model}")
    print(f"Agent model: {agent_model}")
    print(f"Executor: {executor}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    if run_indefinitely:
        print("Mode: unlimited iterations/runtime/QA-fix waves (until completion or manual stop)")
    print(f"QA waves: {'unlimited' if run_indefinitely else max_waves} | QA iters: {qa_iterations} | Fix iters: {fix_iterations}")
    print()

    print("[ORCHESTRATOR] Decomposing task into pipeline stages...\n")
    result = call_pipeline(orchestrator_model, task)

    try:
        stages = normalize_pipeline(result)
    except Exception as exc:
        print(f"[ORCHESTRATOR] Invalid pipeline output: {exc}")
        print(f"[ORCHESTRATOR] Raw output: {json.dumps(result, indent=2)[:4000]}")
        raise

    print(f"[ORCHESTRATOR] Created {len(stages)} pipeline stage(s):")
    for stage in stages:
        dep_text = f" (depends on: {', '.join(stage['depends_on'])})" if stage["depends_on"] else ""
        print(f"  - {stage['id']} [{stage['task_type']}]: {stage['role']}{dep_text}")
    print()

    stage_defs = []
    for stage in stages:
        full_task = build_agent_task(stage, stages, branch=None)
        node_iterations = max_iterations + 5 if stage.get("task_type") in {"reduce", "verify"} else max_iterations
        stage_defs.append(
            {
                "id": stage["id"],
                "task": full_task,
                "task_type": stage.get("task_type", "build"),
                "max_iterations": node_iterations,
            }
        )

    if shared_dir_arg:
        shared_dir = shared_dir_arg
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"shared-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[SHARED] Workspace: {shared_dir}")

    broker = None
    docker_runtime = None
    runtime = {
        "executor": "host",
        "broker_router": os.environ.get("BROKER_ROUTER", "tcp://localhost:5555"),
        "broker_sub": os.environ.get("BROKER_SUB", "tcp://localhost:5556"),
        "shared_workspace_path": shared_dir,
    }
    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")

    if executor == "docker":
        if broker_mode != "host":
            raise RuntimeError("COLLAB_EXECUTOR=docker requires BROKER_MODE_ORCHESTRATOR=host.")
        _ensure_docker_available()
        _ensure_docker_image(docker_image, auto_build=_truthy(os.environ.get("COLLAB_DOCKER_AUTO_BUILD")))
        docker_runtime = _setup_docker_runtime(shared_dir, docker_image)
        runtime.update(docker_runtime)
        print(f"[DOCKER] Network: {docker_runtime['docker_network']}")
        print(f"[BROKER] Container: {docker_runtime['docker_broker_container']}")
    elif broker_mode == "host":
        router_port = _broker_port_from_endpoint(runtime.get("broker_router", ""), 5555)
        sub_port = _broker_port_from_endpoint(runtime.get("broker_sub", ""), 5556)
        broker = MessageBroker(
            router_port=router_port,
            pub_port=sub_port,
            enable_logging=False,
        )
        broker.start()
        time.sleep(0.5)
        print(f"[BROKER] Started on :{router_port}/:{sub_port}")
    else:
        print("[BROKER] Connecting to existing broker")

    build_failures = []
    qa_passed = False
    qa_waves_run = 0
    try:
        for idx, stage_def in enumerate(stage_defs, start=1):
            print(f"[ORCHESTRATOR] Running stage {idx}/{len(stage_defs)}: {stage_def['id']}")
            wave_results = run_wave(
                [stage_def],
                f"STAGE-{idx}",
                agent_model,
                max_iterations,
                max_runtime,
                shared_dir,
                runtime,
            )
            failed = [name for name, code in wave_results.items() if code != 0]
            build_failures.extend(failed)
            if failed:
                print(f"[ORCHESTRATOR] Stage failed: {failed}")
                break

        if qa_enabled:
            wave_num = 0
            while run_indefinitely or wave_num < max_waves:
                qa_waves_run += 1
                report_path = os.path.join(shared_dir, "qa_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)

                qa_task = build_qa_task(task, stages, shared_dir)
                run_wave(
                    [{"id": "qa", "task": qa_task}],
                    f"QA-{wave_num + 1}",
                    agent_model,
                    qa_iterations,
                    fix_runtime,
                    shared_dir,
                    runtime,
                )

                report = read_qa_report(shared_dir)
                print(f"\n[QA-{wave_num + 1}] Status: {report['status']}")
                print(f"[QA-{wave_num + 1}] Summary: {report['summary']}")

                if report["status"] == "pass":
                    qa_passed = True
                    print(f"\n{'=' * 70}")
                    print(f"  QA PASSED on wave {wave_num + 1}!")
                    print(f"{'=' * 70}")
                    break

                for err in report.get("errors", []):
                    print(f"  [{err['id']}] {err['severity']} — {err['file']}: {err['description']}")

                if not run_indefinitely and wave_num == max_waves - 1:
                    print(f"\n[ORCHESTRATOR] QA FAILED after {max_waves} wave(s). Errors remain.")
                    break

                print(f"\n[ORCHESTRATOR] Assigning {len(report['errors'])} error(s) to pipeline stages...")
                assignments = call_assign_fixes(orchestrator_model, report, stages)

                fix_defs = build_fix_defs(assignments, stages, report)
                run_wave(
                    fix_defs,
                    f"FIX-{wave_num + 1}",
                    agent_model,
                    fix_iterations,
                    fix_runtime,
                    shared_dir,
                    runtime,
                )
                wave_num += 1
    finally:
        if broker:
            broker.stop()
        if docker_runtime:
            _cleanup_docker_runtime(docker_runtime)

    build_passed = len(build_failures) == 0
    final_passed = build_passed and (qa_passed if qa_enabled else True)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Build result: {'PASSED' if build_passed else 'FAILED'}")
    print(f"  QA result: {'PASSED' if qa_passed else ('SKIPPED' if not qa_enabled else 'FAILED')}")

    _print_workspace(shared_dir)

    workspace_files = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            workspace_files.append(
                {"path": os.path.relpath(path, shared_dir), "size_bytes": os.path.getsize(path)}
            )

    last_qa_report = None
    report_path = os.path.join(shared_dir, "qa_report.json")
    if qa_enabled and os.path.exists(report_path):
        with open(report_path) as f:
            last_qa_report = json.load(f)

    summary = {
        "status": "pass" if final_passed else "fail",
        "pattern": "pipeline",
        "workspace": shared_dir,
        "task": task,
        "stages": stages,
        "build_passed": build_passed,
        "build_failures": build_failures,
        "qa_waves_run": qa_waves_run,
        "qa_passed": qa_passed,
        "qa_skipped": not qa_enabled,
        "run_indefinitely": run_indefinitely,
        "qa_report": last_qa_report,
        "files": workspace_files,
    }
    summary_path = os.path.join(shared_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Run summary written to: {summary_path}")

    sys.exit(0 if final_passed else 1)


if __name__ == "__main__":
    main()
