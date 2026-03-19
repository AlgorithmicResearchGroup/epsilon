#!/usr/bin/env python3
"""
Dynamic orchestrator: LLM decomposes a task into N agent assignments,
launches them, then runs QA/fix waves until the build passes.

Usage:
  python orchestrators/dag_orchestrator.py "Build a URL shortener microservice"
  python orchestrators/dag_orchestrator.py --prompts challenge_prompts.json --prompt 2
  python orchestrators/dag_orchestrator.py "Build backend API" --branch team/backend --shared-dir /path/to/repo

Env vars:
  MAX_WAVES          — QA/fix retry waves (default 3, 0 = skip QA)
  QA_ITERATIONS      — QA agent iteration budget (default 30)
  FIX_ITERATIONS     — fix agent iteration budget (default 15)
  FIX_RUNTIME_SECONDS — fix agent runtime budget (default 120)
  COLLAB_EXECUTOR    — host|docker (default host)
  COLLAB_DOCKER_IMAGE — Docker image used when COLLAB_EXECUTOR=docker (default epsilon)
  COLLAB_DOCKER_AUTO_BUILD — 1 to auto-build image if missing
  COLLAB_DOCKER_AGENT_MAIN_BINARY — docker entrypoint path for native agents (default /home/agent/bin/agent-main)
  COLLAB_DOCKER_BYOA_RUNNER_BINARY — docker entrypoint path for adapter agents (default /home/agent/bin/byoa-runner)
  COLLAB_AGENT_MODE  — native|adapter (default native)
  COLLAB_AGENT_ADAPTER_CMD — external agent command when COLLAB_AGENT_MODE=adapter
  COLLAB_AGENT_ADAPTER_ENTRY — Python function entrypoint when COLLAB_AGENT_MODE=adapter
"""
import argparse
import os
import sys
import json
import time
import subprocess
import threading
import re
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

TASK_TYPES = {"map", "contract", "build", "reduce", "verify"}


# ---------------------------------------------------------------------------
# Orchestrator tool schema
# ---------------------------------------------------------------------------

DECOMPOSE_TOOL = {
    "name": "decompose_task",
    "description": "Decompose a user task into a typed task graph for parallel collaborative execution.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short lowercase task identifier (e.g. 'contract_api', 'map_01'). Must be unique.",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Task graph node type: map|contract|build|reduce|verify.",
                        },
                        "role": {
                            "type": "string",
                            "description": "One-line role summary (e.g. 'API contract author', 'integration verifier').",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "Detailed task instructions. Be explicit about deliverables and constraints. "
                                "The agent will execute this verbatim."
                            ),
                        },
                        "inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Inputs consumed by this task (files/contracts/data).",
                        },
                        "outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete artifacts this task must produce.",
                        },
                        "acceptance": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Objective checks that define done for this task.",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Task IDs this task must wait for. Empty if no dependencies.",
                        },
                    },
                    "required": ["id", "role", "task", "depends_on"],
                },
            },
        },
        "required": ["agents"],
    },
}

ASSIGN_FIXES_TOOL = {
    "name": "assign_fixes",
    "description": "Assign QA errors to the original agents that should fix them.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "ID of the original agent that should fix these errors.",
                        },
                        "error_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Error IDs from the QA report (e.g. err_1, err_2).",
                        },
                        "fix_task": {
                            "type": "string",
                            "description": (
                                "Specific instructions for what this agent needs to fix. "
                                "Reference exact files, line numbers, and error messages."
                            ),
                        },
                    },
                    "required": ["agent_id", "error_ids", "fix_task"],
                },
            },
        },
        "required": ["assignments"],
    },
}

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a project manager decomposing a task into a typed task graph.

Each agent is an autonomous AI agent with access to: bash, file read/write, web search, URL fetching, and inter-agent messaging. \
Agents share a single working directory where all artifacts go. \
They communicate via a messaging protocol (send_message / check_messages tools).

Rules for decomposition:
1. Create a typed DAG with 2-30 nodes depending on task size. Use only as many nodes as needed.
2. Each node gets a unique lowercase ID.
3. Every node must include: id, task_type, role, task, inputs, outputs, acceptance, depends_on.
4. task_type must be one of: map, contract, build, reduce, verify.
5. Prefer broad parallelism: maximize independent map/build nodes where possible.
6. If the task requires shared semantics/interfaces, create an early contract wave before build nodes.
7. Be EXTREMELY specific in task descriptions. For code: name exact files, function signatures, data schemas. \
For research: name exact topics, source types, output file names, section structure. \
The agent has no context beyond what you write.
8. outputs and acceptance must be concrete and testable, not vague goals.
9. Every file goes in the working directory using relative paths. Do NOT use a shared/ subdirectory.
10. Do NOT create a dedicated QA/review agent — quality review is handled separately by the orchestrator.
11. Keep dependency chains shallow; avoid unnecessary serialization.

Call the decompose_task tool with your DAG node assignments."""

ASSIGN_FIXES_SYSTEM_PROMPT = """\
You are a technical project manager assigning bug fixes to the agents that originally built the code.

You will receive:
1. A list of agents with their original roles and task descriptions
2. A QA error report listing specific bugs found in the integrated build

Your job: assign each error to the agent best positioned to fix it, based on which agent \
created the files involved. Write a clear, specific fix_task for each agent describing \
exactly what they need to change."""


# ---------------------------------------------------------------------------
# QA agent task template
# ---------------------------------------------------------------------------

QA_AGENT_TASK_TEMPLATE = """\
You are the QA agent. Your job is to test the integrated result of a multi-agent build.

THE ORIGINAL TASK THE TEAM WAS WORKING ON:
{original_task}

AGENTS THAT BUILT THIS:
{agent_summaries}

FILES IN THE WORKSPACE:
{file_listing}

YOUR TESTING PROTOCOL:
1. Read the key files to understand what was produced.
2. Determine the task type from the original task description and files present:

   FOR CODE PROJECTS:
   - Install dependencies (requirements.txt, package.json, etc.)
   - Run test suites (pytest, npm test, etc.)
   - Actually run the application: start servers, curl endpoints, run CLI tools with sample input
   - Check HTML references point to real files/CDNs
   - Check for missing imports, undefined references, circular imports

   FOR RESEARCH/ANALYSIS/DOCUMENTS:
   - Verify all required topics/sections are present with substantive content
   - Check that sources/citations are included and URLs are valid (curl -I a sample)
   - Verify minimum depth (not just bullet points — real analysis)
   - Check for placeholder text, TODOs, or incomplete sections
   - Verify the final deliverable file exists and is assembled

   FOR ANY TASK:
   - Verify the deliverable matches what was requested in the original task
   - Check for completeness against every requirement listed

WRITE YOUR REPORT:
You MUST write a file called qa_report.json with this exact schema:
{{
  "status": "pass" or "fail",
  "summary": "one paragraph summary of what you found",
  "errors": [
    {{
      "id": "err_1",
      "severity": "critical" or "warning",
      "category": "import_error|missing_dependency|runtime_crash|test_failure|integration_bug|missing_section|shallow_content|broken_reference|placeholder_text|incomplete_coverage",
      "file": "the file that needs fixing",
      "description": "what is wrong",
      "evidence": "the exact error output or code snippet that shows the problem",
      "suggested_fix": "what should be changed to fix it"
    }}
  ],
  "files_tested": ["list of files you examined or tested"],
  "commands_run": [{{"cmd": "the command", "exit_code": 0}}]
}}

CRITICAL RULES:
- You MUST actually run/test the output. Do not just read it and guess.
- For code: if a server needs to run in the background, use: python app.py & then sleep 3, then curl, \
then kill %1. Or use: timeout 5 python app.py to check for startup errors.
- Write qa_report.json BEFORE calling done.
- Be thorough but focus on things that actually break. Style issues are "warning" severity.
- If everything works, status is "pass" and errors is an empty list.
- Only report real errors you observed, not hypothetical ones."""


# ---------------------------------------------------------------------------
# Fix agent task template
# ---------------------------------------------------------------------------

FIX_AGENT_TASK_TEMPLATE = """\
You are {agent_id} on a fix iteration. You previously built code as part of a team, \
and QA testing found errors in your work.

YOUR ORIGINAL ROLE: {original_role}

ERRORS YOU MUST FIX:
{error_details}

FIX INSTRUCTIONS:
{fix_task}

RULES:
- Read the relevant files first to understand the current state.
- Make surgical fixes. Do NOT rewrite files from scratch.
- After fixing, run a quick sanity check (e.g. python -c "import mymodule" or python -m pytest).
- Call done when your fixes are applied."""


# ---------------------------------------------------------------------------
# Orchestrator LLM calls (LiteLLM unified)
# ---------------------------------------------------------------------------

def call_orchestrator(model_name, task):
    result = chat_with_tools(
        model=model_name,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this task into agent assignments:\n\n{task}"},
        ],
        tools=[tool_schema_to_openai(DECOMPOSE_TOOL)],
        tool_choice={"type": "function", "function": {"name": "decompose_task"}},
        max_tokens=4096,
        temperature=0,
    )
    if result["tool_name"] != "decompose_task" or not isinstance(result["tool_args"], dict):
        raise RuntimeError(
            f"Decompose call returned no valid decompose_task tool call. tool={result['tool_name']} text={result['assistant_text']}"
        )
    return result["tool_args"]


# ---------------------------------------------------------------------------
# Decomposition normalization / validation
# ---------------------------------------------------------------------------

def _to_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _infer_task_type(node_id, role, task):
    haystack = f"{node_id} {role} {task}".lower()
    if any(k in haystack for k in ("contract", "schema", "interface", "spec")):
        return "contract"
    if any(k in haystack for k in ("verify", "validate", "test", "check")):
        return "verify"
    if any(k in haystack for k in ("integrate", "merge", "assemble", "aggregate", "finalize")):
        return "reduce"
    if any(k in haystack for k in ("map", "parallel", "independent", "fanout", "fan-out")):
        return "map"
    return "build"


def normalize_decomposition_result(result):
    """Normalize model decomposition output to a validated typed DAG node list."""
    if isinstance(result, dict):
        raw_nodes = result.get("agents") or result.get("subtasks") or result.get("tasks")
        if raw_nodes is None:
            raise ValueError("Decomposition result missing 'agents'/'subtasks'/'tasks'.")
    elif isinstance(result, list):
        raw_nodes = result
    else:
        raise ValueError(f"Unexpected decomposition result type: {type(result)}")

    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Decomposition produced no nodes.")

    normalized = []
    for idx, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"Node at index {idx} is not an object.")

        node_id = str(raw.get("id", f"node_{idx + 1}")).strip() or f"node_{idx + 1}"
        role = str(raw.get("role") or raw.get("owner") or "General contributor").strip()
        task = str(raw.get("task") or raw.get("instructions") or raw.get("description") or "").strip()
        if not task:
            raise ValueError(f"Node '{node_id}' is missing task instructions.")

        task_type = str(raw.get("task_type", "")).strip().lower()
        if task_type not in TASK_TYPES:
            task_type = _infer_task_type(node_id=node_id, role=role, task=task)

        depends_on = _to_string_list(raw.get("depends_on", raw.get("dependencies", [])))
        inputs = _to_string_list(raw.get("inputs"))
        outputs = _to_string_list(raw.get("outputs"))
        acceptance = _to_string_list(raw.get("acceptance"))

        normalized.append({
            "id": node_id,
            "task_type": task_type,
            "role": role,
            "task": task,
            "inputs": inputs,
            "outputs": outputs,
            "acceptance": acceptance,
            "depends_on": depends_on,
        })

    ids = [n["id"] for n in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate node IDs in decomposition: {ids}")

    id_set = set(ids)
    for node in normalized:
        deps = node["depends_on"]
        if node["id"] in deps:
            raise ValueError(f"Node '{node['id']}' cannot depend on itself.")
        unknown = [d for d in deps if d not in id_set]
        if unknown:
            raise ValueError(f"Node '{node['id']}' has unknown dependencies: {unknown}")

    return normalized


# ---------------------------------------------------------------------------
# Fix assignment LLM calls
# ---------------------------------------------------------------------------

def _build_assign_fixes_prompt(report, agents):
    agent_lines = []
    for a in agents:
        agent_lines.append(
            f"- {a['id']} (type: {a.get('task_type', 'build')}, role: {a['role']}): {a['task'][:200]}..."
        )
    agents_str = "\n".join(agent_lines)

    error_lines = []
    for err in report["errors"]:
        error_lines.append(
            f"- {err['id']} [{err['severity']}] in {err['file']}: {err['description']}\n"
            f"  Evidence: {err['evidence'][:200]}\n"
            f"  Suggested fix: {err.get('suggested_fix', 'N/A')}"
        )
    errors_str = "\n".join(error_lines)

    return (
        f"ORIGINAL AGENTS:\n{agents_str}\n\n"
        f"QA ERRORS TO ASSIGN:\n{errors_str}\n\n"
        f"Assign each error to the agent that should fix it. "
        f"Use the assign_fixes tool."
    )


def call_assign_fixes(model_name, report, agents):
    prompt = _build_assign_fixes_prompt(report, agents)
    result = chat_with_tools(
        model=model_name,
        messages=[
            {"role": "system", "content": ASSIGN_FIXES_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=[tool_schema_to_openai(ASSIGN_FIXES_TOOL)],
        tool_choice={"type": "function", "function": {"name": "assign_fixes"}},
        max_tokens=4096,
        temperature=0,
    )
    if result["tool_name"] != "assign_fixes" or not isinstance(result["tool_args"], dict):
        raise RuntimeError(
            f"Assign-fixes call returned no valid assign_fixes tool call. tool={result['tool_name']} text={result['assistant_text']}"
        )
    return result["tool_args"]


# ---------------------------------------------------------------------------
# Dependency-ordered wave scheduling
# ---------------------------------------------------------------------------

def topological_waves(agents):
    """Group agents into waves based on depends_on. Wave N agents depend only on wave <N agents."""
    remaining = {a["id"]: set(a["depends_on"]) for a in agents}
    agent_by_id = {a["id"]: a for a in agents}
    waves = []
    while remaining:
        ready = [aid for aid, deps in remaining.items() if not deps]
        if not ready:
            unresolved = {aid: sorted(list(deps)) for aid, deps in remaining.items()}
            raise ValueError(f"Dependency cycle detected in decomposition DAG: {unresolved}")
        waves.append([agent_by_id[aid] for aid in ready])
        for aid in ready:
            del remaining[aid]
        for deps in remaining.values():
            deps -= set(ready)
    return waves


# ---------------------------------------------------------------------------
# Boilerplate wrapper
# ---------------------------------------------------------------------------

def build_agent_task(agent_def, all_agents, branch=None):
    """Wrap the LLM-generated task with generic protocol/workspace boilerplate."""
    agent_id = agent_def["id"]
    task_type = agent_def.get("task_type", "build")
    role = agent_def["role"]
    task = agent_def["task"]
    inputs = agent_def.get("inputs", [])
    outputs = agent_def.get("outputs", [])
    acceptance = agent_def.get("acceptance", [])
    depends_on = agent_def["depends_on"]
    num_agents = len(all_agents)

    teammate_lines = []
    for a in all_agents:
        if a["id"] != agent_id:
            dep_marker = " (you depend on this agent)" if a["id"] in depends_on else ""
            teammate_lines.append(f"  - {a['id']} [{a.get('task_type', 'build')}]: {a['role']}{dep_marker}")
    teammates_str = "\n".join(teammate_lines)

    if depends_on:
        dep_names = ", ".join(depends_on)
        wait_block = (
            f"These agents have already completed their work: {dep_names}\n"
            f"Their files are already in your workspace. "
            f"Read them to understand what's available, then start your work immediately."
        )
    else:
        wait_block = (
            "You have NO dependencies — start your work immediately. "
            "Other agents may be waiting on you, so work efficiently."
        )

    input_block = "\n".join(f"- {x}" for x in inputs) if inputs else "- (none declared)"
    output_block = "\n".join(f"- {x}" for x in outputs) if outputs else "- (derive explicit outputs from task)"
    acceptance_block = "\n".join(f"- {x}" for x in acceptance) if acceptance else "- (self-verify with targeted checks)"

    return (
        f"You are {agent_id} on a {num_agents}-node task graph. Your role: {role}\n"
        f"NODE TYPE: {task_type}\n\n"
        f"TEAMMATES:\n{teammates_str}\n\n"
        f"DEPENDENCIES: {wait_block}\n\n"
        f"DECLARED INPUTS:\n{input_block}\n\n"
        f"REQUIRED OUTPUTS:\n{output_block}\n\n"
        f"ACCEPTANCE CHECKS:\n{acceptance_block}\n\n"
        f"YOUR TASK:\n{task}\n\n"
        f"SHARED WORKSPACE: Your working directory is shared with all agents. "
        f"All code and artifacts go here using relative paths (e.g. app.py, tests/). "
        f"Do NOT create a shared/ subdirectory.\n\n"
        f"COMPLETION GUARD:\n"
        f"- Prioritize writing required outputs and validating acceptance checks quickly.\n"
        f"- Avoid unnecessary planning/check_messages loops.\n"
        f"- As soon as required outputs exist and acceptance checks pass, call done immediately.\n\n"
        + (
            f"GIT WORKFLOW: You are working in a git repository on branch '{branch}'. "
            f"After writing or modifying files, commit your changes: git add -A && git commit -m \"description of changes\". "
            f"To see teammates' latest work: git pull --rebase. Commit frequently.\n\n"
            if branch else ""
        ) +
        f"COMMUNICATION PROTOCOL:\n"
        f"- When you finish your work, you MUST call send_message to announce completion to the team. "
        f"Include what files you created and a brief summary.\n"
        f"- Other agents may be WAITING for your announcement before they can start.\n"
        f"- After sending your completion message, check messages once or twice for team updates, then call done."
    )


# ---------------------------------------------------------------------------
# QA task builder
# ---------------------------------------------------------------------------

def _list_workspace_files(shared_dir):
    """Return a string listing all files in the workspace with sizes."""
    lines = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            lines.append(f"  {rel} ({size} bytes)")
    return "\n".join(lines) if lines else "  (empty)"


def build_qa_task(original_task, agents, shared_dir):
    """Construct the QA agent's task string."""
    agent_summaries = "\n".join(
        f"  - {a['id']} [{a.get('task_type', 'build')}]: {a['role']}"
        for a in agents
    )
    file_listing = _list_workspace_files(shared_dir)

    return QA_AGENT_TASK_TEMPLATE.format(
        original_task=original_task,
        agent_summaries=agent_summaries,
        file_listing=file_listing,
    )


def read_qa_report(shared_dir):
    """Read and parse qa_report.json. Returns a synthetic fail report if QA timed out before writing one."""
    report_path = os.path.join(shared_dir, "qa_report.json")
    if not os.path.exists(report_path):
        print(f"[QA] No qa_report.json found — QA agent likely timed out.")
        return {
            "status": "fail",
            "summary": "QA agent timed out before writing qa_report.json.",
            "errors": [{
                "id": "err_timeout",
                "severity": "critical",
                "category": "runtime_crash",
                "file": "qa_report.json",
                "description": "QA agent timed out before completing testing and writing its report.",
                "evidence": "qa_report.json not found after QA wave completed.",
                "suggested_fix": "Increase QA_ITERATIONS or FIX_RUNTIME_SECONDS.",
            }],
            "files_tested": [],
            "commands_run": [],
        }
    with open(report_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fix task builder
# ---------------------------------------------------------------------------

def build_fix_defs(assignments, agents, report):
    """Build fix agent definitions from LLM error assignments."""
    # Index agents and errors by ID for lookup
    agent_by_id = {a["id"]: a for a in agents}
    error_by_id = {e["id"]: e for e in report["errors"]}

    fix_defs = []
    for assignment in assignments["assignments"]:
        agent_id = assignment["agent_id"]
        agent = agent_by_id[agent_id]

        # Format the specific errors assigned to this agent
        error_details_lines = []
        for eid in assignment["error_ids"]:
            err = error_by_id[eid]
            error_details_lines.append(
                f"[{eid}] ({err['severity']}) {err['file']}: {err['description']}\n"
                f"  Evidence: {err['evidence']}\n"
                f"  Suggested fix: {err.get('suggested_fix', 'N/A')}"
            )
        error_details = "\n\n".join(error_details_lines)

        fix_task_str = FIX_AGENT_TASK_TEMPLATE.format(
            agent_id=agent_id,
            original_role=agent["role"],
            error_details=error_details,
            fix_task=assignment["fix_task"],
        )
        fix_defs.append({"id": agent_id, "task": fix_task_str})

    return fix_defs


# ---------------------------------------------------------------------------
# Agent subprocess launcher
# ---------------------------------------------------------------------------

DOCKER_ENV_ALLOWLIST = {
    # Provider credentials used by model/tool clients inside containers.
    "OPENAI",
    "ANTHROPIC",
    "TAVILY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY",
    # Orchestrator/agent runtime configuration.
    "SETTINGS_PACK",
    "ORCHESTRATOR_MODEL",
    "AGENT_MODEL",
    "MAX_ITERATIONS",
    "MAX_RUNTIME_SECONDS",
    "RESI_RUN_INDEFINITELY",
    "MAX_TOKENS",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_API_BASE",
    "MODEL_TEMPERATURE",
    "MODEL_TOP_P",
    "MODEL_TOP_K",
    # Collaboration protocol + shared workspace wiring (required in docker mode).
    "PROTOCOL_ENABLED",
    "BROKER_MODE",
    "BROKER_ROUTER",
    "BROKER_SUB",
    "AGENT_ID",
    "AGENT_TASK_TYPE",
    "AGENT_ROLE",
    "AGENT_TOPICS",
    "WORK_QUEUE_ENABLED",
    "SHARED_WORKSPACE",
    "PROTOCOL_HEARTBEAT_INTERVAL_SECONDS",
    "BROKER_HEARTBEAT_TIMEOUT_SECONDS",
    "BROKER_LEASE_TIMEOUT_SECONDS",
    "BROKER_SWEEP_INTERVAL_SECONDS",
    "BROKER_MAX_REDELIVERIES",
    "BROKER_MAX_FAIL_RETRIES",
    "BROKER_REDELIVERY_BACKOFF_BASE_SECONDS",
    "BROKER_REDELIVERY_BACKOFF_MAX_SECONDS",
    # BYOA adapter mode controls.
    "COLLAB_AGENT_MODE",
    "COLLAB_AGENT_ADAPTER_CMD",
    "COLLAB_AGENT_ADAPTER_ENTRY",
    # Network/runtime defaults.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "PYTHONIOENCODING",
}


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _broker_port_from_endpoint(endpoint: str, default: int) -> int:
    value = str(endpoint or "").strip()
    match = re.search(r":(\d+)$", value)
    if not match:
        return default
    try:
        parsed = int(match.group(1))
    except ValueError:
        return default
    if 1 <= parsed <= 65535:
        return parsed
    return default


def _safe_container_name(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")
    return safe[:63] or f"agent-{int(time.time())}"


def _ensure_docker_available() -> None:
    proc = subprocess.run(["docker", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError("Docker is not available. Install/start Docker or use --executor host.")


def _ensure_docker_image(image: str, auto_build: bool) -> None:
    inspect = subprocess.run(["docker", "image", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if inspect.returncode == 0:
        return
    if not auto_build:
        raise RuntimeError(
            f"Docker image '{image}' not found. Build it first with: docker build -t {image} . "
            "or set COLLAB_DOCKER_AUTO_BUILD=1."
        )
    print(f"[DOCKER] Image '{image}' not found. Building...")
    subprocess.run(["docker", "build", "-t", image, "."], cwd=PROJECT_ROOT, check=True)


def _setup_docker_runtime(shared_dir: str, image: str) -> Dict[str, str]:
    run_tag = _safe_container_name(f"collab-{int(time.time() * 1000)}")
    network_name = _safe_container_name(f"{run_tag}-net")
    broker_name = _safe_container_name(f"{run_tag}-broker")

    subprocess.run(["docker", "network", "create", network_name], check=True)
    broker_cmd = [
        "docker", "run", "-d", "--rm",
        "--name", broker_name,
        "--network", network_name,
        "--entrypoint", "python",
    ]
    for key in (
        "BROKER_HEARTBEAT_TIMEOUT_SECONDS",
        "BROKER_LEASE_TIMEOUT_SECONDS",
        "BROKER_SWEEP_INTERVAL_SECONDS",
        "BROKER_MAX_REDELIVERIES",
        "BROKER_MAX_FAIL_RETRIES",
        "BROKER_REDELIVERY_BACKOFF_BASE_SECONDS",
        "BROKER_REDELIVERY_BACKOFF_MAX_SECONDS",
    ):
        value = os.environ.get(key)
        if value:
            broker_cmd.extend(["-e", f"{key}={value}"])
    broker_cmd.extend([image, "-m", "agent_protocol.broker_server"])
    subprocess.run(broker_cmd, check=True, stdout=subprocess.DEVNULL)
    time.sleep(0.8)

    return {
        "executor": "docker",
        "docker_image": image,
        "docker_network": network_name,
        "docker_broker_container": broker_name,
        "broker_router": f"tcp://{broker_name}:5555",
        "broker_sub": f"tcp://{broker_name}:5556",
        "shared_workspace_path": "/workspace",
        "docker_user": os.environ.get("COLLAB_DOCKER_USER", ""),
    }


def _cleanup_docker_runtime(state: Dict[str, str]) -> None:
    broker_name = state.get("docker_broker_container")
    network_name = state.get("docker_network")
    if broker_name:
        subprocess.run(["docker", "stop", broker_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if network_name:
        subprocess.run(["docker", "network", "rm", network_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _build_agent_env(agent_def, agent_model, max_iterations, max_runtime, shared_dir, runtime):
    shared_workspace_path = runtime.get("shared_workspace_path", shared_dir)
    env = {
        **os.environ,
        "PROTOCOL_ENABLED": "true",
        "BROKER_MODE": "connect",
        "AGENT_ID": agent_def["id"],
        "AGENT_MODEL": agent_model,
        "MAX_ITERATIONS": str(max_iterations),
        "MAX_RUNTIME_SECONDS": str(max_runtime),
        "SHARED_WORKSPACE": shared_workspace_path,
        "AGENT_TASK_TYPE": str(agent_def.get("task_type", "")),
        "AGENT_ROLE": str(agent_def.get("role", "")),
        "BROKER_ROUTER": runtime.get("broker_router", os.environ.get("BROKER_ROUTER", "tcp://localhost:5555")),
        "BROKER_SUB": runtime.get("broker_sub", os.environ.get("BROKER_SUB", "tcp://localhost:5556")),
        "PYTHONIOENCODING": "utf-8",
    }
    return env


def _run_agent_host(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir, runtime):
    env = _build_agent_env(agent_def, agent_model, max_iterations, max_runtime, shared_dir, runtime)
    entry_script = runtime.get("host_agent_entry", os.path.join(PROJECT_ROOT, "runtime", "agent_main.py"))
    if str(entry_script).endswith(".py"):
        launch_cmd = [sys.executable, entry_script, agent_def["task"]]
    else:
        launch_cmd = [entry_script, agent_def["task"]]
    proc = subprocess.Popen(
        launch_cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        print(f"[{log_prefix}] {text}", flush=True)
    proc.wait()
    return proc.returncode


def _run_agent_docker(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir, runtime):
    env = _build_agent_env(agent_def, agent_model, max_iterations, max_runtime, shared_dir, runtime)
    container_name = _safe_container_name(f"{runtime.get('docker_network', 'collab')}-{agent_def['id']}-{int(time.time() * 1000)}")

    cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "--network", runtime["docker_network"],
        "-v", f"{os.path.abspath(shared_dir)}:{runtime['shared_workspace_path']}",
    ]

    docker_user = runtime.get("docker_user", "").strip()
    if docker_user:
        cmd.extend(["-u", docker_user])

    for key, value in env.items():
        if key in DOCKER_ENV_ALLOWLIST and value is not None:
            cmd.extend(["-e", f"{key}={value}"])

    container_entry = runtime.get("docker_agent_entry", "/home/agent/bin/agent-main")
    fallback_py_entry = runtime.get("docker_agent_fallback_py", "/home/agent/runtime/agent_main.py")
    fallback_bins = runtime.get("docker_agent_fallback_bins", [])
    fallback_bin_checks = " ".join(f'"{str(path)}"' for path in fallback_bins if str(path).strip())
    if str(container_entry).endswith(".py"):
        cmd.extend(["--entrypoint", "python"])
        cmd.extend([runtime["docker_image"], container_entry, agent_def["task"]])
    else:
        # Prefer compiled entrypoint, but gracefully fall back to a python entry script
        # when running against older images that do not contain the compiled binary path.
        cmd.extend(["--entrypoint", "/bin/sh"])
        fallback_script = (
            'entry="$1"; task="$2"; '
            'if [ -x "$entry" ]; then exec "$entry" "$task"; fi; '
            f'for candidate in {fallback_bin_checks}; do if [ -x "$candidate" ]; then exec "$candidate" "$task"; fi; done; '
            f'if [ -f "{fallback_py_entry}" ]; then exec python "{fallback_py_entry}" "$task"; fi; '
            'echo "missing agent entrypoint: $entry" >&2; '
            "exit 127"
        )
        cmd.extend([runtime["docker_image"], "-lc", fallback_script, "--", container_entry, agent_def["task"]])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        print(f"[{log_prefix}] {text}", flush=True)
    proc.wait()
    return proc.returncode


def run_agent(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir, runtime):
    if runtime.get("executor") == "docker":
        return _run_agent_docker(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir, runtime)
    return _run_agent_host(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir, runtime)


# ---------------------------------------------------------------------------
# Wave runner
# ---------------------------------------------------------------------------

def run_wave(agent_defs, wave_name, agent_model, max_iterations, max_runtime, shared_dir, runtime):
    """Launch all agents in a wave, wait for all to complete. Returns {name: exit_code}."""
    print(f"\n{'─' * 70}")
    print(f"  WAVE: {wave_name}  ({len(agent_defs)} agent(s))")
    print(f"{'─' * 70}\n")

    threads = []
    results = {}

    for agent_def in agent_defs:
        name = f"{wave_name}:{agent_def['id'].upper()}"

        def target(a=agent_def, n=name):
            per_agent_iterations = int(a.get("max_iterations", max_iterations))
            results[n] = run_agent(a, n, agent_model, per_agent_iterations, max_runtime, shared_dir, runtime)

        t = threading.Thread(target=target)
        t.start()
        threads.append(t)
        time.sleep(1)

    for t in threads:
        t.join()

    # Print wave results
    for name, code in results.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  [{name}] {status}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_workspace(shared_dir):
    print(f"\n  Shared workspace contents:")
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for f in sorted(files):
            path = os.path.join(root, f)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            print(f"    {rel} ({size} bytes)")


def parse_args():
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Dynamic multi-agent orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--branch", help="Git branch to work on (for hierarchical tree mode)")
    parser.add_argument("--shared-dir", help="Use an existing shared directory instead of creating one")
    parser.add_argument("--executor", choices=["host", "docker"], help="Execution backend for all agents/broker")
    parser.add_argument("--docker-image", help="Docker image to use when --executor=docker")
    parser.add_argument("--agent-mode", choices=["native", "adapter"], help="Agent runtime mode")
    parser.add_argument("--adapter-cmd", help="External adapter command when --agent-mode=adapter")
    parser.add_argument(
        "--adapter-entry",
        help="Python function entrypoint '<module_or_file>:<run_function>' when --agent-mode=adapter",
    )
    parser.add_argument("--pattern", help="Optional pattern label; must resolve to DAG for this entrypoint")
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
        if resolved != "dag":
            parser.error(
                f"dag_orchestrator.py only supports DAG/fanout pattern, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved

    return args


def main():
    args = parse_args()
    task = args.task
    branch = args.branch
    shared_dir_arg = args.shared_dir

    # Load manifest config
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
    agent_mode = (args.agent_mode or os.environ.get("COLLAB_AGENT_MODE", "native")).strip().lower()
    adapter_cmd = (args.adapter_cmd or os.environ.get("COLLAB_AGENT_ADAPTER_CMD", "")).strip()
    adapter_entry = (args.adapter_entry or os.environ.get("COLLAB_AGENT_ADAPTER_ENTRY", "")).strip()
    native_agent_binary = os.environ.get("COLLAB_AGENT_MAIN_BINARY", "").strip()
    byoa_runner_binary = os.environ.get("COLLAB_BYOA_RUNNER_BINARY", "").strip()
    docker_native_agent_entry = os.environ.get("COLLAB_DOCKER_AGENT_MAIN_BINARY", "/home/agent/bin/agent-main").strip()
    docker_byoa_runner_entry = os.environ.get("COLLAB_DOCKER_BYOA_RUNNER_BINARY", "/home/agent/bin/byoa-runner").strip()

    if agent_mode not in {"native", "adapter"}:
        raise RuntimeError(f"Unsupported agent mode: {agent_mode}")
    if agent_mode == "adapter" and not (adapter_entry or adapter_cmd):
        raise RuntimeError(
            "COLLAB_AGENT_MODE=adapter requires COLLAB_AGENT_ADAPTER_ENTRY or "
            "COLLAB_AGENT_ADAPTER_CMD (or pass --adapter-entry/--adapter-cmd)."
        )
    os.environ["COLLAB_AGENT_MODE"] = agent_mode
    if adapter_cmd:
        os.environ["COLLAB_AGENT_ADAPTER_CMD"] = adapter_cmd
    if adapter_entry:
        os.environ["COLLAB_AGENT_ADAPTER_ENTRY"] = adapter_entry

    print("=" * 70)
    print("  DYNAMIC ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Pattern: {os.environ.get('COLLAB_PATTERN', 'dag')}")
    print(f"Orchestrator model: {orchestrator_model}")
    print(f"Agent model: {agent_model}")
    print(f"Agent runtime: {agent_mode}")
    if adapter_entry:
        print(f"Adapter entry: {adapter_entry}")
    elif adapter_cmd:
        print(f"Adapter command: {adapter_cmd}")
    print(f"Executor: {executor}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    if run_indefinitely:
        print("Mode: unlimited iterations/runtime/QA-fix waves (until completion or manual stop)")
    print(f"QA waves: {'unlimited' if run_indefinitely else max_waves} | QA iters: {qa_iterations} | Fix iters: {fix_iterations}")
    if branch:
        print(f"Branch: {branch}")
    print()

    # ── Decompose ──────────────────────────────────────────────────────────
    print("[ORCHESTRATOR] Decomposing task...\n")
    result = call_orchestrator(orchestrator_model, task)

    try:
        agents = normalize_decomposition_result(result)
    except Exception as exc:
        print(f"[ORCHESTRATOR] Invalid decomposition output: {exc}")
        print(f"[ORCHESTRATOR] Raw output: {json.dumps(result, indent=2)[:4000]}")
        raise

    print(f"[ORCHESTRATOR] Created {len(agents)} DAG nodes:")
    for a in agents:
        ttype = a.get("task_type", "build")
        deps = f" (depends on: {', '.join(a['depends_on'])})" if a["depends_on"] else ""
        print(f"  - {a['id']} [{ttype}]: {a['role']}{deps}")
    print()

    # Wrap tasks with boilerplate
    agent_defs = []
    for a in agents:
        full_task = build_agent_task(a, agents, branch=branch)
        node_iterations = max_iterations
        if a.get("task_type") in {"reduce", "verify"}:
            node_iterations = max_iterations + 5
        agent_defs.append(
            {
                "id": a["id"],
                "task": full_task,
                "task_type": a.get("task_type", "build"),
                "role": a.get("role", ""),
                "max_iterations": node_iterations,
            }
        )

    # ── Setup ──────────────────────────────────────────────────────────────
    if shared_dir_arg:
        shared_dir = shared_dir_arg
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"shared-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[SHARED] Workspace: {shared_dir}")

    # If branch specified, checkout that branch in the workspace
    if branch:
        subprocess.run(["git", "checkout", branch], cwd=shared_dir, check=True)
        print(f"[GIT] Checked out branch: {branch}")

    broker_mode = os.environ.get("BROKER_MODE_ORCHESTRATOR", "host")
    broker = None
    docker_runtime = None
    runtime = {
        "executor": "host",
        "broker_router": os.environ.get("BROKER_ROUTER", "tcp://localhost:5555"),
        "broker_sub": os.environ.get("BROKER_SUB", "tcp://localhost:5556"),
        "shared_workspace_path": shared_dir,
        "host_agent_entry": os.path.join(
            PROJECT_ROOT,
            "runtime",
            "byoa_runner.py" if agent_mode == "adapter" else "agent_main.py",
        ),
        "docker_agent_entry": docker_byoa_runner_entry if agent_mode == "adapter" else docker_native_agent_entry,
        "docker_agent_fallback_py": "/home/agent/runtime/byoa_runner.py"
        if agent_mode == "adapter"
        else "/home/agent/runtime/agent_main.py",
        "docker_agent_fallback_bins": [
            "/home/agent/bin/byoa-runner",
            "/home/agent/bin/byoa-runner-linux",
        ]
        if agent_mode == "adapter"
        else [
            "/home/agent/bin/agent-main",
            "/home/agent/bin/agent-main-linux",
        ],
    }
    if agent_mode == "adapter" and byoa_runner_binary and os.path.isfile(byoa_runner_binary):
        runtime["host_agent_entry"] = byoa_runner_binary
    elif agent_mode == "native" and native_agent_binary and os.path.isfile(native_agent_binary):
        runtime["host_agent_entry"] = native_agent_binary

    if executor == "docker":
        if broker_mode != "host":
            raise RuntimeError("COLLAB_EXECUTOR=docker requires BROKER_MODE_ORCHESTRATOR=host.")
        _ensure_docker_available()
        _ensure_docker_image(docker_image, auto_build=_truthy(os.environ.get("COLLAB_DOCKER_AUTO_BUILD")))
        docker_runtime = _setup_docker_runtime(shared_dir, docker_image)
        runtime.update(docker_runtime)
        print(f"[DOCKER] Network: {docker_runtime['docker_network']}")
        print(f"[BROKER] Container: {docker_runtime['docker_broker_container']} (tcp://{docker_runtime['docker_broker_container']}:5555 / :5556)")
    elif broker_mode == "host":
        router_port = _broker_port_from_endpoint(runtime.get("broker_router", ""), 5555)
        sub_port = _broker_port_from_endpoint(runtime.get("broker_sub", ""), 5556)
        heartbeat_timeout = float(os.environ.get("BROKER_HEARTBEAT_TIMEOUT_SECONDS", "30"))
        lease_timeout = float(os.environ.get("BROKER_LEASE_TIMEOUT_SECONDS", "60"))
        sweep_interval = float(os.environ.get("BROKER_SWEEP_INTERVAL_SECONDS", "1"))
        max_redeliveries = int(os.environ.get("BROKER_MAX_REDELIVERIES", "5"))
        max_fail_retries = int(os.environ.get("BROKER_MAX_FAIL_RETRIES", "0"))
        backoff_base = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_BASE_SECONDS", "0"))
        backoff_max = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_MAX_SECONDS", "30"))
        broker = MessageBroker(
            router_port=router_port,
            pub_port=sub_port,
            enable_logging=False,
            heartbeat_timeout_seconds=heartbeat_timeout,
            lease_timeout_seconds=lease_timeout,
            sweep_interval_seconds=sweep_interval,
            max_redeliveries=max_redeliveries,
            max_fail_retries=max_fail_retries,
            redelivery_backoff_base_seconds=backoff_base,
            redelivery_backoff_max_seconds=backoff_max,
        )
        broker.start()
        time.sleep(0.5)
        print(f"[BROKER] Started on :{router_port}/:{sub_port}")
    else:
        print("[BROKER] Connecting to existing broker")

    # ── BUILD (dependency-ordered waves) ──────────────────────────────────
    build_failures = []
    qa_passed = False
    qa_waves_run = 0
    report = None
    try:
        waves = topological_waves(agents)
        for wave_idx, wave_agents in enumerate(waves):
            wave_ids = {a["id"] for a in wave_agents}
            wave_defs = [d for d in agent_defs if d["id"] in wave_ids]
            print(f"[ORCHESTRATOR] BUILD-{wave_idx + 1} nodes: {', '.join(sorted(wave_ids))}")
            wave_results = run_wave(
                wave_defs,
                f"BUILD-{wave_idx + 1}",
                agent_model,
                max_iterations,
                max_runtime,
                shared_dir,
                runtime,
            )
            failed = [name for name, code in wave_results.items() if code != 0]
            build_failures.extend(failed)

        build_passed = len(build_failures) == 0
        if not build_passed:
            print(f"[ORCHESTRATOR] Build stage had {len(build_failures)} failed node(s): {build_failures}")

        # ── QA Loop ───────────────────────────────────────────────────────────
        if qa_enabled:
            wave_num = 0
            while run_indefinitely or wave_num < max_waves:
                qa_waves_run += 1
                # Delete stale qa_report.json
                report_path = os.path.join(shared_dir, "qa_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)

                # Run QA agent
                qa_task = build_qa_task(task, agents, shared_dir)
                qa_def = [{"id": "qa", "task": qa_task}]
                run_wave(
                    qa_def,
                    f"QA-{wave_num + 1}",
                    agent_model,
                    qa_iterations,
                    fix_runtime,
                    shared_dir,
                    runtime,
                )

                # Read report
                report = read_qa_report(shared_dir)
                print(f"\n[QA-{wave_num + 1}] Status: {report['status']}")
                print(f"[QA-{wave_num + 1}] Summary: {report['summary']}")

                if report["status"] == "pass":
                    print(f"\n{'=' * 70}")
                    print(f"  QA PASSED on wave {wave_num + 1}!")
                    print(f"{'=' * 70}")
                    qa_passed = True
                    break

                # Print errors
                for err in report.get("errors", []):
                    print(f"  [{err['id']}] {err['severity']} — {err['file']}: {err['description']}")

                if not run_indefinitely and wave_num == max_waves - 1:
                    print(f"\n[ORCHESTRATOR] QA FAILED after {max_waves} wave(s). Errors remain.")
                    break

                # Assign errors to agents
                print(f"\n[ORCHESTRATOR] Assigning {len(report['errors'])} error(s) to agents...")
                assignments = call_assign_fixes(orchestrator_model, report, agents)

                for a in assignments["assignments"]:
                    print(f"  - {a['agent_id']}: {', '.join(a['error_ids'])}")

                # Build and run fix wave
                fix_defs = build_fix_defs(assignments, agents, report)
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
        else:
            build_passed = len(build_failures) == 0
    finally:
        # ── Cleanup ────────────────────────────────────────────────────────────
        if broker:
            broker.stop()
        if docker_runtime:
            _cleanup_docker_runtime(docker_runtime)

    build_passed = len(build_failures) == 0

    print("\n" + "=" * 70)
    print("  ALL WAVES COMPLETE")
    print("=" * 70)
    final_passed = build_passed and (qa_passed if qa_enabled else True)
    print(f"  Build result: {'PASSED' if build_passed else 'FAILED'}")
    if qa_enabled:
        print(f"  QA result: {'PASSED' if qa_passed else 'FAILED'}")
    else:
        print("  QA result: SKIPPED (MAX_WAVES=0)")

    _print_workspace(shared_dir)

    # ── Write run summary ─────────────────────────────────────────────────
    # Collect workspace file listing
    workspace_files = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, shared_dir)
            workspace_files.append({"path": rel, "size_bytes": os.path.getsize(path)})

    # Load last QA report if it exists
    last_qa_report = None
    if qa_enabled:
        report_path = os.path.join(shared_dir, "qa_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                last_qa_report = json.load(f)

    summary = {
        "status": "pass" if final_passed else "fail",
        "pattern": "dag",
        "agent_mode": agent_mode,
        "workspace": shared_dir,
        "task": task,
        "agents": [
            {
                "id": a["id"],
                "task_type": a.get("task_type", "build"),
                "role": a["role"],
                "depends_on": a.get("depends_on", []),
                "inputs": a.get("inputs", []),
                "outputs": a.get("outputs", []),
                "acceptance": a.get("acceptance", []),
            }
            for a in agents
        ],
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
