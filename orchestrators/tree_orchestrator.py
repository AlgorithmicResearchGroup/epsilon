#!/usr/bin/env python3
"""
Hierarchical tree orchestrator: decomposes a task into teams, each team
runs its own DAG sub-orchestrator with QA, then merges and runs integration QA.

Usage:
  python orchestrators/tree_orchestrator.py "Build an e-commerce platform"
  python orchestrators/tree_orchestrator.py --prompts challenge_prompts.json --prompt 5

Env vars:
  MAX_ITERATIONS       — per-worker iteration budget (default 40)
  MAX_RUNTIME_SECONDS  — per-worker runtime budget (default 300)
  MAX_WAVES            — QA/fix waves per team (default 2)
  QA_ITERATIONS        — QA agent iterations per team (default 30)
  FIX_ITERATIONS       — fix agent iterations per team (default 15)
  FIX_RUNTIME_SECONDS  — fix agent runtime budget (default 120)
  INTEGRATION_WAVES    — integration QA/fix waves after merge (default 2)
"""
import argparse
import os
import sys
import json
import time
import subprocess
import threading
import re
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from agent_protocol.broker import MessageBroker
from agent.models.litellm_client import chat_with_tools
from agent.utils import tool_schema_to_openai
from orchestrators.patterns import resolve_pattern


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


# ---------------------------------------------------------------------------
# Team decomposition tool schema
# ---------------------------------------------------------------------------

DECOMPOSE_TEAMS_TOOL = {
    "name": "decompose_teams",
    "description": "Decompose a task into 2-8 teams for hierarchical parallel execution. Each team will independently decompose further into 2-5 worker agents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "teams": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short lowercase team identifier (e.g. 'backend', 'frontend', 'data').",
                        },
                        "name": {
                            "type": "string",
                            "description": "Human-readable team name (e.g. 'Backend API Team').",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "The team's sub-task. Be extremely specific: name exact files, APIs, schemas. "
                                "This will be passed to a sub-orchestrator that further decomposes it into workers."
                            ),
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Team IDs this team must wait for. Empty if no dependencies.",
                        },
                    },
                    "required": ["id", "name", "task", "depends_on"],
                },
            },
        },
        "required": ["teams"],
    },
}

TEAM_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a technical architect decomposing a large software task into team-level assignments.

Each team will be managed by its own sub-orchestrator that further breaks the work into 2-5 worker agents. \
Teams work in parallel on separate git branches and their results are merged at the end.

Rules:
1. Create 2-8 teams. Each team should own a coherent, self-contained piece of the system.
2. Each team gets a unique lowercase ID (e.g. backend, frontend, data, infra, testing).
3. Teams with no dependencies start immediately. Dependent teams wait.
4. Be EXTREMELY specific in task descriptions. Name exact files, APIs, schemas, ports.
5. Think about integration boundaries: teams need clear interfaces (API contracts, shared schemas, file formats).
6. Do NOT create a dedicated testing team — each team runs its own QA. Integration testing happens after merge.
7. Keep dependency chains shallow — prefer wide parallelism.

Call the decompose_teams tool with your team assignments."""


# ---------------------------------------------------------------------------
# Integration QA template
# ---------------------------------------------------------------------------

INTEGRATION_QA_TEMPLATE = """\
You are the integration QA agent. Multiple teams built different parts of this project on separate branches. \
Their work has been merged into a single codebase. Your job is to test that everything works together.

THE ORIGINAL TASK:
{original_task}

TEAMS THAT BUILT THIS:
{team_summaries}

FILES IN THE MERGED WORKSPACE:
{file_listing}

YOUR TESTING PROTOCOL:
1. Read the key source files to understand how the pieces fit together.
2. Install all dependencies (requirements.txt, package.json, etc).
3. Run any existing test suites (pytest, npm test, etc).
4. Test cross-team integration points:
   - Do imports between modules from different teams work?
   - Do API contracts match between frontend and backend?
   - Are shared data schemas consistent?
   - Do database migrations / seed scripts from different teams compose correctly?
5. Try to run the full application end-to-end.
6. Check for dependency conflicts (different teams requiring incompatible versions).

WRITE YOUR REPORT:
Write qa_report.json with this schema:
{{
  "status": "pass" or "fail",
  "summary": "one paragraph summary",
  "errors": [
    {{
      "id": "err_1",
      "severity": "critical" or "warning",
      "category": "import_error|missing_dependency|runtime_crash|test_failure|integration_bug|merge_conflict",
      "file": "the file that needs fixing",
      "description": "what is wrong",
      "evidence": "exact error output",
      "suggested_fix": "what should change",
      "team": "which team's code is involved"
    }}
  ],
  "files_tested": ["list of files examined"],
  "commands_run": [{{"cmd": "command", "exit_code": 0}}]
}}

CRITICAL RULES:
- Actually RUN the code. Do not just read it.
- Focus on integration issues — things that work in isolation but break when combined.
- Write qa_report.json BEFORE calling done.
- If everything works, status is "pass" and errors is empty."""


# ---------------------------------------------------------------------------
# LLM calls for team decomposition
# ---------------------------------------------------------------------------

def call_decompose_teams(model_name, task):
    result = chat_with_tools(
        model=model_name,
        messages=[
            {"role": "system", "content": TEAM_ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this task into teams:\n\n{task}"},
        ],
        tools=[tool_schema_to_openai(DECOMPOSE_TEAMS_TOOL)],
        tool_choice={"type": "function", "function": {"name": "decompose_teams"}},
        max_tokens=4096,
        temperature=0,
    )
    if result["tool_name"] != "decompose_teams" or not isinstance(result["tool_args"], dict):
        raise RuntimeError(
            f"Team decomposition call returned no valid decompose_teams tool call. "
            f"tool={result['tool_name']} text={result['assistant_text']}"
        )
    return result["tool_args"]


# ---------------------------------------------------------------------------
# Team runner (launches DAG sub-orchestrator as subprocess)
# ---------------------------------------------------------------------------

def run_team(team_def, shared_dir, agent_model, max_iterations, max_runtime,
             max_waves, qa_iterations, fix_iterations, fix_runtime):
    """Run a team's sub-orchestrator. Returns exit code."""
    team_id = team_def["id"]
    branch = f"team/{team_id}"
    log_prefix = f"TEAM:{team_id.upper()}"

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "orchestrators", "dag_orchestrator.py"),
        team_def["task"],
        "--branch", branch,
        "--shared-dir", shared_dir,
    ]
    env = {
        **os.environ,
        "AGENT_MODEL": agent_model,
        "MAX_ITERATIONS": str(max_iterations),
        "MAX_RUNTIME_SECONDS": str(max_runtime),
        "MAX_WAVES": str(max_waves),
        "QA_ITERATIONS": str(qa_iterations),
        "FIX_ITERATIONS": str(fix_iterations),
        "FIX_RUNTIME_SECONDS": str(fix_runtime),
        "BROKER_MODE_ORCHESTRATOR": "connect",  # don't start a new broker
    }

    proc = subprocess.Popen(
        cmd, env=env, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        print(f"[{log_prefix}] {text}", flush=True)
    proc.wait()
    return proc.returncode


def run_team_wave(team_defs, shared_dir, agent_model, max_iterations, max_runtime,
                  max_waves, qa_iterations, fix_iterations, fix_runtime):
    """Launch all teams in parallel, wait for completion. Returns {team_id: exit_code}."""
    print(f"\n{'─' * 70}")
    print(f"  TEAM WAVE  ({len(team_defs)} team(s))")
    print(f"{'─' * 70}\n")

    threads = []
    results = {}

    for team_def in team_defs:
        def target(t=team_def):
            results[t["id"]] = run_team(
                t, shared_dir, agent_model, max_iterations, max_runtime,
                max_waves, qa_iterations, fix_iterations, fix_runtime,
            )

        t = threading.Thread(target=target)
        t.start()
        threads.append(t)
        time.sleep(2)  # stagger team launches

    for t in threads:
        t.join()

    for team_id, code in results.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  [TEAM:{team_id.upper()}] {status}")

    return results


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def git_run(args, cwd, check=True):
    """Run a git command in the workspace."""
    result = subprocess.run(
        ["git"] + args, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if check and result.returncode != 0:
        print(f"[GIT] Command failed: git {' '.join(args)}")
        print(f"[GIT] Output: {result.stdout}")
        raise subprocess.CalledProcessError(result.returncode, ["git"] + args)
    return result


def setup_git_repo(shared_dir, team_ids):
    """Initialize git repo and create team branches."""
    git_run(["init"], shared_dir)
    git_run(["config", "user.email", "orchestrator@agent"], shared_dir)
    git_run(["config", "user.name", "Orchestrator"], shared_dir)

    # Initial commit on main so branches have a base
    readme = os.path.join(shared_dir, ".gitkeep")
    with open(readme, "w") as f:
        f.write("")
    git_run(["add", "."], shared_dir)
    git_run(["commit", "-m", "Initial commit"], shared_dir)

    # Create team branches
    for team_id in team_ids:
        branch = f"team/{team_id}"
        git_run(["branch", branch], shared_dir)
        print(f"[GIT] Created branch: {branch}")

    # Stay on default branch (main or master depending on git config)
    result = git_run(["branch", "--show-current"], shared_dir)
    print(f"[GIT] On branch: {result.stdout.strip()}")


def get_default_branch(shared_dir):
    """Get the name of the default branch (main or master)."""
    result = git_run(["rev-parse", "--abbrev-ref", "HEAD"], shared_dir)
    return result.stdout.strip()


def merge_team_branches(shared_dir, team_ids):
    """Merge all team branches into the default branch. Returns list of merge conflicts."""
    default_branch = get_default_branch(shared_dir)
    git_run(["checkout", default_branch], shared_dir)

    conflicts = []
    for team_id in team_ids:
        branch = f"team/{team_id}"
        print(f"[GIT] Merging {branch} into main...")
        result = git_run(
            ["merge", branch, "--no-edit", "-m", f"Merge {branch}"],
            shared_dir, check=False,
        )
        if result.returncode != 0:
            print(f"[GIT] Merge conflict on {branch}: {result.stdout}")
            conflicts.append(team_id)
            # Abort the failed merge so we can continue
            git_run(["merge", "--abort"], shared_dir, check=False)
        else:
            print(f"[GIT] Merged {branch} successfully")

    return conflicts


# ---------------------------------------------------------------------------
# Integration QA helpers
# ---------------------------------------------------------------------------

def _list_workspace_files(shared_dir):
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


def build_integration_qa_task(original_task, teams, shared_dir):
    team_summaries = "\n".join(f"  - {t['id']}: {t['name']} — {t['task'][:150]}..." for t in teams)
    file_listing = _list_workspace_files(shared_dir)
    return INTEGRATION_QA_TEMPLATE.format(
        original_task=original_task,
        team_summaries=team_summaries,
        file_listing=file_listing,
    )


def run_integration_qa(task, teams, shared_dir, agent_model, qa_iterations, fix_runtime, runtime):
    """Run integration QA agent, return the report."""
    from orchestrators.dag_orchestrator import run_wave, read_qa_report

    # Delete stale report
    report_path = os.path.join(shared_dir, "qa_report.json")
    if os.path.exists(report_path):
        os.remove(report_path)

    qa_task = build_integration_qa_task(task, teams, shared_dir)
    qa_def = [{"id": "integration-qa", "task": qa_task}]
    run_wave(
        qa_def,
        "INTEGRATION-QA",
        agent_model,
        qa_iterations,
        fix_runtime,
        shared_dir,
        runtime,
    )

    return read_qa_report(shared_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical tree orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--shared-dir", help="Use an existing workspace directory instead of creating runs/tree-*")
    parser.add_argument("--pattern", help="Optional pattern label; must resolve to TREE for this entrypoint")
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
        if resolved != "tree":
            parser.error(
                f"tree_orchestrator.py only supports TREE/hierarchy pattern, got '{args.pattern}'. "
                "Use orchestrate.py for automatic routing."
            )
        os.environ["COLLAB_PATTERN"] = resolved

    return args


def main():
    args = parse_args()
    task = args.task

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
    max_waves = int(os.environ.get("MAX_WAVES", "2"))
    qa_iterations = int(os.environ.get("QA_ITERATIONS", "30"))
    fix_iterations = int(os.environ.get("FIX_ITERATIONS", "15"))
    fix_runtime = os.environ.get("FIX_RUNTIME_SECONDS", "120")
    integration_waves = int(os.environ.get("INTEGRATION_WAVES", "2"))

    print("=" * 70)
    print("  HIERARCHICAL TREE ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Pattern: {os.environ.get('COLLAB_PATTERN', 'tree')}")
    print(f"Orchestrator model: {orchestrator_model}")
    print(f"Agent model: {agent_model}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    print(f"Team QA waves: {max_waves} | Integration waves: {integration_waves}")
    print()

    # ── Decompose into teams ──────────────────────────────────────────────
    print("[ROOT] Decomposing task into teams...\n")
    result = call_decompose_teams(orchestrator_model, task)

    teams = result["teams"]
    print(f"[ROOT] Created {len(teams)} teams:")
    for t in teams:
        deps = f" (depends on: {', '.join(t['depends_on'])})" if t["depends_on"] else ""
        print(f"  - {t['id']}: {t['name']}{deps}")
    print()

    # ── Setup git repo ────────────────────────────────────────────────────
    if args.shared_dir:
        shared_dir = args.shared_dir
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(PROJECT_ROOT, "runs", f"tree-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[ROOT] Workspace: {shared_dir}")

    team_ids = [t["id"] for t in teams]
    setup_git_repo(shared_dir, team_ids)

    # ── Start broker ──────────────────────────────────────────────────────
    broker_router = os.environ.get("BROKER_ROUTER", "tcp://localhost:5555")
    broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
    router_port = _broker_port_from_endpoint(broker_router, 5555)
    sub_port = _broker_port_from_endpoint(broker_sub, 5556)
    broker = MessageBroker(
        router_port=router_port,
        pub_port=sub_port,
        enable_logging=False,
    )
    broker.start()
    time.sleep(0.5)
    print(f"[BROKER] Started on :{router_port}/:{sub_port}")
    runtime = {
        "executor": "host",
        "broker_router": broker_router,
        "broker_sub": broker_sub,
        "shared_workspace_path": shared_dir,
    }

    team_results: Dict[str, int] = {}
    team_failures = []
    conflicts = []
    integration_passed = False
    integration_waves_run = 0
    last_integration_report: Dict[str, Any] | None = None

    try:
        # ── Launch teams (respecting dependencies) ────────────────────────
        # For now: simple approach — launch all teams at once.
        # Teams with dependencies will wait via check_messages at the agent level.
        # TODO: wave-based launch for teams with deps (like dag_orchestrator does)
        team_defs = [{"id": t["id"], "task": t["task"]} for t in teams]
        team_results = run_team_wave(
            team_defs, shared_dir, agent_model, max_iterations, max_runtime,
            max_waves, qa_iterations, fix_iterations, fix_runtime,
        )
        team_failures = [team_id for team_id, code in team_results.items() if code != 0]
        if team_failures:
            print(f"[ROOT] Team stage had failures: {team_failures}")

        # ── Merge team branches ───────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("  MERGING TEAM BRANCHES")
        print(f"{'=' * 70}\n")

        conflicts = merge_team_branches(shared_dir, team_ids)
        if conflicts:
            print(f"\n[ROOT] MERGE CONFLICTS in teams: {', '.join(conflicts)}")
            print("[ROOT] Skipping integration QA — resolve conflicts first.")
        elif integration_waves <= 0:
            integration_passed = True
            print("\n[ROOT] Integration QA skipped (INTEGRATION_WAVES=0).")
        else:
            print("\n[ROOT] All branches merged cleanly into main.")
            for wave_num in range(integration_waves):
                integration_waves_run += 1
                print(f"\n{'=' * 70}")
                print(f"  INTEGRATION QA — Wave {wave_num + 1}/{integration_waves}")
                print(f"{'=' * 70}\n")

                report = run_integration_qa(
                    task, teams, shared_dir, agent_model, qa_iterations, fix_runtime, runtime,
                )
                last_integration_report = report
                print(f"\n[INTEGRATION-QA-{wave_num + 1}] Status: {report['status']}")
                print(f"[INTEGRATION-QA-{wave_num + 1}] Summary: {report['summary']}")

                if report["status"] == "pass":
                    print(f"\n{'=' * 70}")
                    print(f"  INTEGRATION QA PASSED on wave {wave_num + 1}!")
                    print(f"{'=' * 70}")
                    integration_passed = True
                    break

                for err in report.get("errors", []):
                    print(f"  [{err['id']}] {err['severity']} — {err.get('file', '?')}: {err['description']}")

                if wave_num == integration_waves - 1:
                    print(f"\n[ROOT] Integration QA FAILED after {integration_waves} wave(s).")
                    break

                # TODO: assign integration fixes back to teams and re-run
                print("[ROOT] Integration fix assignment not yet implemented — running QA again.")
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        broker.stop()

    print("\n" + "=" * 70)
    print("  TREE ORCHESTRATION COMPLETE")
    print("=" * 70)

    # Print final workspace
    print(f"\n  Workspace: {shared_dir}")
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            print(f"    {rel} ({size} bytes)")

    team_passed = len(team_failures) == 0
    merge_passed = len(conflicts) == 0
    integration_required = merge_passed and integration_waves > 0
    qa_passed = integration_passed if integration_required else merge_passed
    final_passed = team_passed and merge_passed and qa_passed

    summary = {
        "status": "pass" if final_passed else "fail",
        "pattern": "tree",
        "workspace": shared_dir,
        "task": task,
        "teams": teams,
        "team_results": team_results,
        "team_failures": team_failures,
        "merge_conflicts": conflicts,
        "integration_waves_requested": integration_waves,
        "integration_waves_run": integration_waves_run,
        "integration_passed": integration_passed,
        "integration_report": last_integration_report,
    }
    summary_path = os.path.join(shared_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Run summary written to: {summary_path}")

    sys.exit(0 if final_passed else 1)


if __name__ == "__main__":
    main()
