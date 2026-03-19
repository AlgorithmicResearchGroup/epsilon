#!/usr/bin/env python3
"""
Supervisor orchestrator: runs a DAG with per-stage supervision decisions.

The supervisor evaluates each completed stage and can decide to:
- pass: accept stage output
- retry: rerun stage with focused feedback
- split: inject follow-up stages before downstream dependencies proceed
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from agent.models.litellm_client import chat_with_tools
from agent.utils import tool_schema_to_openai
from agent_protocol.broker import MessageBroker
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
    call_orchestrator,
    normalize_decomposition_result,
    read_qa_report,
    run_wave,
)
from orchestrators.patterns import resolve_pattern


SUPERVISOR_REVIEW_TOOL = {
    "name": "review_stage",
    "description": "Review a stage result and decide pass, retry, split, or reassign to a new stage owner.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "description": "One of: pass, retry, split, reassign.",
            },
            "reason": {
                "type": "string",
                "description": "Concise rationale grounded in observed workspace state.",
            },
            "follow_ups": {
                "type": "array",
                "description": "Required when decision=split. Follow-up tasks to execute before dependents.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "role": {"type": "string"},
                        "task": {"type": "string"},
                        "outputs": {"type": "array", "items": {"type": "string"}},
                        "acceptance": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "role", "task"],
                },
            },
            "reassignment": {
                "type": "object",
                "description": "Required when decision=reassign. Defines a replacement stage for the same scope.",
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string"},
                    "task": {"type": "string"},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "acceptance": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role"],
            },
        },
        "required": ["decision", "reason"],
    },
}

SUPERVISOR_REVIEW_SYSTEM_PROMPT = """\
You are a strict technical supervisor for a staged build.

Given the task, stage definition, and current workspace files, decide:
- pass: stage is complete enough for downstream work.
- retry: stage must rerun with focused guidance.
- split: stage partially succeeded but needs one or more focused follow-up stages.
- reassign: stage should be replaced by a new owner/recovery stage.

Rules:
1. Be conservative: only pass when outputs/acceptance are clearly satisfied.
2. Use retry for small corrections; split for substantial additional work.
3. Use reassign when prior retries are not converging or a different role/expertise is needed.
4. Keep follow_ups concrete and bounded (1-3 follow-up stages).
5. Follow-ups and reassigned stages must be executable by autonomous coding agents.

Call the review_stage tool."""


def _workspace_snapshot(shared_dir: str, max_files: int = 120, preview_bytes: int = 160) -> str:
    lines = []
    count = 0
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            count += 1
            if count > max_files:
                lines.append("- ... (truncated)")
                return "\n".join(lines)
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            line = f"- {rel} ({size} bytes)"
            if size <= 4096:
                try:
                    raw = open(path, "rb").read(preview_bytes)
                    line += f" | bytes_preview={raw!r}"
                except OSError:
                    pass
            lines.append(line)
    return "\n".join(lines) if lines else "- (empty)"


def _build_review_prompt(
    original_task: str,
    node: Dict[str, Any],
    stage_attempt: int,
    max_stage_retries: int,
    shared_dir: str,
    stage_status: str = "success",
    exit_code: Optional[int] = None,
) -> str:
    outputs = "\n".join(f"- {item}" for item in node.get("outputs", [])) or "- (none declared)"
    acceptance = "\n".join(f"- {item}" for item in node.get("acceptance", [])) or "- (none declared)"
    files = _workspace_snapshot(shared_dir)
    status_block = f"STAGE EXECUTION STATUS: {stage_status}"
    if exit_code is not None:
        status_block += f"\nSTAGE EXIT CODE: {exit_code}"

    return (
        f"ORIGINAL TASK:\n{original_task}\n\n"
        f"STAGE ID: {node['id']}\n"
        f"STAGE ROLE: {node['role']}\n"
        f"ATTEMPT: {stage_attempt}/{max_stage_retries + 1}\n\n"
        f"{status_block}\n\n"
        f"STAGE TASK:\n{node['task']}\n\n"
        f"REQUIRED OUTPUTS:\n{outputs}\n\n"
        f"ACCEPTANCE CHECKS:\n{acceptance}\n\n"
        f"WORKSPACE SNAPSHOT (includes byte previews for small files):\n{files}\n\n"
        "If required artifact files exist with matching byte previews, prefer 'pass'. "
        "Do not request retry unless there is clear evidence of mismatch or missing outputs. "
        "If this is a repeated failed attempt that is not converging, prefer 'reassign' over another retry.\n\n"
        "Decide pass, retry, split, or reassign."
    )


def _review_stage(
    model_name: str,
    original_task: str,
    node: Dict[str, Any],
    stage_attempt: int,
    max_stage_retries: int,
    shared_dir: str,
    stage_status: str = "success",
    exit_code: Optional[int] = None,
) -> Dict[str, Any]:
    prompt = _build_review_prompt(
        original_task,
        node,
        stage_attempt,
        max_stage_retries,
        shared_dir,
        stage_status=stage_status,
        exit_code=exit_code,
    )
    result = chat_with_tools(
        model=model_name,
        messages=[
            {"role": "system", "content": SUPERVISOR_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=[tool_schema_to_openai(SUPERVISOR_REVIEW_TOOL)],
        tool_choice={"type": "function", "function": {"name": "review_stage"}},
        max_tokens=4096,
        temperature=0,
    )
    if result["tool_name"] != "review_stage" or not isinstance(result["tool_args"], dict):
        raise RuntimeError(
            f"Stage-review call returned no valid review_stage tool call. "
            f"tool={result['tool_name']} text={result['assistant_text']}"
        )
    return result["tool_args"]


def _sanitize_stage_id(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", (value or "").strip().lower()).strip("_")
    return safe or f"stage_{int(time.time() * 1000)}"


def _to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        out = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_follow_ups(
    follow_ups: Any,
    parent_id: str,
    used_ids: Set[str],
    lineage: str,
    reassign_depth: int,
) -> List[Dict[str, Any]]:
    if not isinstance(follow_ups, list):
        return []
    normalized = []
    prev = parent_id
    for idx, raw in enumerate(follow_ups, start=1):
        if not isinstance(raw, dict):
            continue
        task = str(raw.get("task", "")).strip()
        role = str(raw.get("role", "Follow-up stage")).strip() or "Follow-up stage"
        if not task:
            continue
        base = _sanitize_stage_id(str(raw.get("id", f"{parent_id}_followup_{idx}")))
        stage_id = base
        suffix = 2
        while stage_id in used_ids:
            stage_id = f"{base}_{suffix}"
            suffix += 1
        used_ids.add(stage_id)
        node = {
            "id": stage_id,
            "task_type": "build",
            "role": role,
            "task": task,
            "inputs": [],
            "outputs": _to_string_list(raw.get("outputs")),
            "acceptance": _to_string_list(raw.get("acceptance")),
            "depends_on": [prev],
            "lineage": lineage,
            "reassign_depth": reassign_depth,
        }
        prev = stage_id
        normalized.append(node)
    return normalized


def _declared_output_paths(node: Dict[str, Any], shared_dir: str) -> List[str]:
    paths = []
    pattern = re.compile(r"\b(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+\b")
    for output in node.get("outputs", []):
        if not isinstance(output, str):
            continue
        for match in pattern.findall(output):
            rel = match.strip().strip("`\"'")
            if not rel or rel.startswith(("http://", "https://")):
                continue
            full = os.path.join(shared_dir, rel)
            if os.path.commonpath([os.path.abspath(full), os.path.abspath(shared_dir)]) != os.path.abspath(shared_dir):
                continue
            paths.append(rel)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for item in paths:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _build_reassigned_node(
    node: Dict[str, Any],
    review: Dict[str, Any],
    used_ids: Set[str],
) -> Dict[str, Any]:
    payload = review.get("reassignment")
    if not isinstance(payload, dict):
        payload = {}
    base = _sanitize_stage_id(str(payload.get("id", f"{node['id']}_reassign")))
    stage_id = base
    suffix = 2
    while stage_id in used_ids:
        stage_id = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(stage_id)

    role = str(payload.get("role", "")).strip() or f"{node.get('role', 'Stage')} recovery"
    task = str(payload.get("task", "")).strip() or str(node.get("task", "")).strip()
    reason = str(review.get("reason", "")).strip()
    if reason:
        task = (
            f"{task}\n\nSUPERVISOR REASSIGNMENT CONTEXT:\n{reason}\n"
            "Recover this stage and satisfy outputs/acceptance."
        ).strip()

    return {
        "id": stage_id,
        "task_type": node.get("task_type", "build"),
        "role": role,
        "task": task,
        "inputs": list(node.get("inputs", [])),
        "outputs": _to_string_list(payload.get("outputs")) or list(node.get("outputs", [])),
        "acceptance": _to_string_list(payload.get("acceptance")) or list(node.get("acceptance", [])),
        "depends_on": list(node.get("depends_on", [])),
        "lineage": str(node.get("lineage", node.get("id", stage_id))),
        "reassign_depth": int(node.get("reassign_depth", 0)) + 1,
    }


def _ready_stage_ids(nodes: Dict[str, Dict[str, Any]], pending: Set[str], completed: Set[str]) -> List[str]:
    ready = []
    for node_id in sorted(pending):
        deps = set(nodes[node_id].get("depends_on", []))
        if deps.issubset(completed):
            ready.append(node_id)
    return ready


def parse_args():
    parser = argparse.ArgumentParser(description="Supervisor multi-agent orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--branch", help="Git branch to work on")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all agents/broker")
    parser.add_argument("--docker-image", help="Docker image to use when --executor=docker")
    parser.add_argument("--pattern", help="Optional pattern label; must resolve to SUPERVISOR for this entrypoint")
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
        if resolved != "supervisor":
            parser.error(
                f"supervisor_orchestrator.py only supports SUPERVISOR pattern, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved
    return args


def main():
    args = parse_args()
    task = args.task
    branch = args.branch
    shared_dir_arg = args.shared_dir

    manifest_path = os.path.join(PROJECT_ROOT, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]
    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL", settings["model"])
    supervisor_model = os.environ.get("SUPERVISOR_MODEL", orchestrator_model)
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
    max_stage_retries = int(os.environ.get("SUPERVISOR_MAX_STAGE_RETRIES", "2"))
    max_reassign_depth = int(os.environ.get("SUPERVISOR_MAX_REASSIGN_DEPTH", "2"))

    print("=" * 70)
    print("  SUPERVISOR ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Pattern: {os.environ.get('COLLAB_PATTERN', 'supervisor')}")
    print(f"Orchestrator model: {orchestrator_model}")
    print(f"Supervisor model: {supervisor_model}")
    print(f"Agent model: {agent_model}")
    print(f"Executor: {executor}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    if run_indefinitely:
        print("Mode: unlimited iterations/runtime/QA-fix waves (until completion or manual stop)")
    print(f"Stage retries: {max_stage_retries} | QA waves: {'unlimited' if run_indefinitely else max_waves}")
    print(f"Max reassign depth: {max_reassign_depth}")
    if branch:
        print(f"Branch: {branch}")
    print()

    print("[SUPERVISOR] Decomposing task...\n")
    result = call_orchestrator(orchestrator_model, task)

    try:
        initial_nodes = normalize_decomposition_result(result)
    except Exception as exc:
        print(f"[SUPERVISOR] Invalid decomposition output: {exc}")
        print(f"[SUPERVISOR] Raw output: {json.dumps(result, indent=2)[:4000]}")
        raise

    for node in initial_nodes:
        node.setdefault("lineage", node["id"])
        node.setdefault("reassign_depth", 0)

    nodes: Dict[str, Dict[str, Any]] = {node["id"]: node for node in initial_nodes}
    used_ids: Set[str] = set(nodes.keys())
    pending: Set[str] = set(nodes.keys())
    completed: Set[str] = set()
    node_attempts: Dict[str, int] = {node_id: 0 for node_id in nodes}
    stage_history: List[Dict[str, Any]] = []

    print(f"[SUPERVISOR] Created {len(nodes)} initial stage(s):")
    for node in initial_nodes:
        deps = f" (depends on: {', '.join(node['depends_on'])})" if node["depends_on"] else ""
        print(f"  - {node['id']} [{node.get('task_type', 'build')}]: {node['role']}{deps}")
    print()

    if shared_dir_arg:
        shared_dir = shared_dir_arg
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"shared-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[SHARED] Workspace: {shared_dir}")

    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")
    broker = None
    docker_runtime = None
    runtime = {
        "executor": "host",
        "broker_router": os.environ.get("BROKER_ROUTER", "tcp://localhost:5555"),
        "broker_sub": os.environ.get("BROKER_SUB", "tcp://localhost:5556"),
        "shared_workspace_path": shared_dir,
    }

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

    build_failures: List[str] = []
    qa_passed = False
    qa_waves_run = 0

    try:
        while pending:
            ready = _ready_stage_ids(nodes, pending, completed)
            if not ready:
                unresolved = {node_id: nodes[node_id].get("depends_on", []) for node_id in sorted(pending)}
                build_failures.append(f"dependency_blocked:{json.dumps(unresolved)}")
                print(f"[SUPERVISOR] No ready stages. Unresolved dependencies: {unresolved}")
                break

            node_id = ready[0]
            node = nodes[node_id]
            node_attempts[node_id] = node_attempts.get(node_id, 0) + 1
            attempt = node_attempts[node_id]

            print(f"[SUPERVISOR] Running stage '{node_id}' attempt {attempt}/{max_stage_retries + 1}")
            wrapped = build_agent_task(node, list(nodes.values()), branch=branch)
            node_iterations = max_iterations + 5 if node.get("task_type") in {"reduce", "verify"} else max_iterations
            wave_results = run_wave(
                [{"id": node_id, "task": wrapped, "max_iterations": node_iterations}],
                f"SUPERVISE-{node_id}-A{attempt}",
                agent_model,
                max_iterations,
                max_runtime,
                shared_dir,
                runtime,
            )
            exit_code = next(iter(wave_results.values()))
            if exit_code != 0:
                try:
                    review = _review_stage(
                        supervisor_model,
                        task,
                        node,
                        attempt,
                        max_stage_retries,
                        shared_dir,
                        stage_status="failed",
                        exit_code=exit_code,
                    )
                except Exception as exc:
                    review = {"decision": "retry", "reason": f"review_error_fallback: {exc}"}

                decision = str(review.get("decision", "retry")).strip().lower()
                reason = str(review.get("reason", "")).strip() or f"exit_code={exit_code}"
                if decision not in {"pass", "retry", "split", "reassign"}:
                    decision = "retry"

                stage_history.append({"stage_id": node_id, "attempt": attempt, "decision": decision, "reason": reason})
                print(f"[SUPERVISOR] Failure review for '{node_id}': {decision} ({reason})")

                if decision == "pass":
                    pending.remove(node_id)
                    completed.add(node_id)
                    continue

                if decision == "retry":
                    if attempt <= max_stage_retries:
                        node["task"] = (
                            f"{node['task']}\n\nSUPERVISOR FEEDBACK (attempt {attempt}):\n{reason}\n"
                            "Address this feedback directly before calling done."
                        )
                        continue
                    build_failures.append(f"{node_id}:exit_{exit_code}:retry_limit")
                    print(f"[SUPERVISOR] Retry limit reached for '{node_id}'.")
                    break

                if decision == "reassign":
                    current_depth = int(node.get("reassign_depth", 0))
                    if current_depth >= max_reassign_depth:
                        build_failures.append(f"{node_id}:reassign_depth_limit")
                        print(
                            f"[SUPERVISOR] Reassign depth limit reached for '{node_id}' "
                            f"(depth={current_depth}, limit={max_reassign_depth})."
                        )
                        break
                    reassigned = _build_reassigned_node(node, review, used_ids)
                    for other_id, other_node in nodes.items():
                        if other_id == node_id:
                            continue
                        deps = list(other_node.get("depends_on", []))
                        if node_id in deps:
                            deps = [dep for dep in deps if dep != node_id]
                            if reassigned["id"] not in deps:
                                deps.append(reassigned["id"])
                            other_node["depends_on"] = deps
                    pending.remove(node_id)
                    nodes[reassigned["id"]] = reassigned
                    node_attempts[reassigned["id"]] = 0
                    pending.add(reassigned["id"])
                    print(
                        f"[SUPERVISOR] Reassigned stage '{node_id}' -> '{reassigned['id']}' ({reassigned['role']})."
                    )
                    continue

                follow_ups = _normalize_follow_ups(
                    review.get("follow_ups"),
                    node_id,
                    used_ids,
                    lineage=str(node.get("lineage", node_id)),
                    reassign_depth=int(node.get("reassign_depth", 0)),
                )
                if not follow_ups:
                    if attempt <= max_stage_retries:
                        print(
                            f"[SUPERVISOR] Split requested for '{node_id}' but no valid follow-ups provided; retrying."
                        )
                        continue
                    build_failures.append(f"{node_id}:exit_{exit_code}:invalid_split")
                    print(f"[SUPERVISOR] Invalid split for '{node_id}' at retry limit.")
                    break

                follow_up_ids = [f["id"] for f in follow_ups]
                for other_id, other_node in nodes.items():
                    if other_id == node_id:
                        continue
                    deps = list(other_node.get("depends_on", []))
                    if node_id in deps:
                        for follow_id in follow_up_ids:
                            if follow_id not in deps:
                                deps.append(follow_id)
                        other_node["depends_on"] = deps

                pending.remove(node_id)
                completed.add(node_id)
                for follow_node in follow_ups:
                    nodes[follow_node["id"]] = follow_node
                    node_attempts[follow_node["id"]] = 0
                    pending.add(follow_node["id"])
                    print(f"[SUPERVISOR] Injected follow-up stage: {follow_node['id']}")
                continue

            try:
                review = _review_stage(
                    supervisor_model, task, node, attempt, max_stage_retries, shared_dir
                )
            except Exception as exc:
                review = {"decision": "pass", "reason": f"review_error_fallback: {exc}"}

            decision = str(review.get("decision", "pass")).strip().lower()
            reason = str(review.get("reason", "")).strip()
            if decision not in {"pass", "retry", "split", "reassign"}:
                decision = "pass"

            stage_history.append({"stage_id": node_id, "attempt": attempt, "decision": decision, "reason": reason})
            print(f"[SUPERVISOR] Review decision for '{node_id}': {decision} ({reason})")

            if decision == "pass":
                pending.remove(node_id)
                completed.add(node_id)
                continue

            if decision == "retry":
                declared_paths = _declared_output_paths(node, shared_dir)
                if declared_paths and all(os.path.exists(os.path.join(shared_dir, p)) for p in declared_paths):
                    print(
                        f"[SUPERVISOR] Overriding retry for '{node_id}': declared outputs exist ({', '.join(declared_paths)})."
                    )
                    stage_history.append(
                        {
                            "stage_id": node_id,
                            "attempt": attempt,
                            "decision": "override_pass",
                            "reason": "declared outputs present after successful execution",
                        }
                    )
                    pending.remove(node_id)
                    completed.add(node_id)
                    continue
                if attempt <= max_stage_retries:
                    feedback = reason or "Supervisor requested refinement."
                    node["task"] = (
                        f"{node['task']}\n\nSUPERVISOR FEEDBACK (attempt {attempt}):\n{feedback}\n"
                        "Address this feedback directly before calling done."
                    )
                    continue
                build_failures.append(f"{node_id}:retry_limit")
                print(f"[SUPERVISOR] Retry limit reached for '{node_id}'.")
                break

            if decision == "reassign":
                current_depth = int(node.get("reassign_depth", 0))
                if current_depth >= max_reassign_depth:
                    build_failures.append(f"{node_id}:reassign_depth_limit")
                    print(
                        f"[SUPERVISOR] Reassign depth limit reached for '{node_id}' "
                        f"(depth={current_depth}, limit={max_reassign_depth})."
                    )
                    break
                reassigned = _build_reassigned_node(node, review, used_ids)
                for other_id, other_node in nodes.items():
                    if other_id == node_id:
                        continue
                    deps = list(other_node.get("depends_on", []))
                    if node_id in deps:
                        deps = [dep for dep in deps if dep != node_id]
                        if reassigned["id"] not in deps:
                            deps.append(reassigned["id"])
                        other_node["depends_on"] = deps
                pending.remove(node_id)
                nodes[reassigned["id"]] = reassigned
                node_attempts[reassigned["id"]] = 0
                pending.add(reassigned["id"])
                print(f"[SUPERVISOR] Reassigned stage '{node_id}' -> '{reassigned['id']}' ({reassigned['role']}).")
                continue

            follow_ups = _normalize_follow_ups(
                review.get("follow_ups"),
                node_id,
                used_ids,
                lineage=str(node.get("lineage", node_id)),
                reassign_depth=int(node.get("reassign_depth", 0)),
            )
            if not follow_ups:
                pending.remove(node_id)
                completed.add(node_id)
                continue

            follow_up_ids = [f["id"] for f in follow_ups]
            for other_id, other_node in nodes.items():
                if other_id == node_id:
                    continue
                deps = list(other_node.get("depends_on", []))
                if node_id in deps:
                    for follow_id in follow_up_ids:
                        if follow_id not in deps:
                            deps.append(follow_id)
                    other_node["depends_on"] = deps

            pending.remove(node_id)
            completed.add(node_id)
            for follow_node in follow_ups:
                nodes[follow_node["id"]] = follow_node
                node_attempts[follow_node["id"]] = 0
                pending.add(follow_node["id"])
                print(f"[SUPERVISOR] Injected follow-up stage: {follow_node['id']}")

        build_passed = len(build_failures) == 0 and len(pending) == 0

        if qa_enabled:
            wave_num = 0
            while run_indefinitely or wave_num < max_waves:
                qa_waves_run += 1
                report_path = os.path.join(shared_dir, "qa_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)

                qa_task = build_qa_task(task, list(nodes.values()), shared_dir)
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
                    break

                if not run_indefinitely and wave_num == max_waves - 1:
                    break

                assignments = call_assign_fixes(orchestrator_model, report, list(nodes.values()))

                fix_defs = build_fix_defs(assignments, list(nodes.values()), report)
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

    final_passed = build_passed and (qa_passed if qa_enabled else True)

    print("\n" + "=" * 70)
    print("  SUPERVISOR COMPLETE")
    print("=" * 70)
    print(f"  Build result: {'PASSED' if build_passed else 'FAILED'}")
    if qa_enabled:
        print(f"  QA result: {'PASSED' if qa_passed else 'FAILED'}")
    else:
        print("  QA result: SKIPPED (MAX_WAVES=0)")

    workspace_files = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            workspace_files.append({"path": os.path.relpath(path, shared_dir), "size_bytes": os.path.getsize(path)})

    last_qa_report = None
    if qa_enabled:
        report_path = os.path.join(shared_dir, "qa_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                last_qa_report = json.load(f)

    summary = {
        "status": "pass" if final_passed else "fail",
        "pattern": "supervisor",
        "workspace": shared_dir,
        "task": task,
        "nodes": list(nodes.values()),
        "stage_history": stage_history,
        "max_stage_retries": max_stage_retries,
        "max_reassign_depth": max_reassign_depth,
        "completed_stage_ids": sorted(completed),
        "pending_stage_ids": sorted(pending),
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
