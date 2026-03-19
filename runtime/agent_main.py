#!/usr/bin/env python3
"""
Epsilon
"""
import os
import sys
import json
import time
import threading
from collections import deque
from datetime import timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from agent.worker import Worker


def _truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main():
    """Main entry point for agent"""
    project_root = PROJECT_ROOT
    task_instructions = ""

    if len(sys.argv) > 1:
        task_instructions = " ".join(sys.argv[1:])
    else:
        task_instructions = os.environ.get("TASK_DESCRIPTION", "")

    if not task_instructions:
        print("Usage: python runtime/agent_main.py '<task description>' or set TASK_DESCRIPTION env var")
        return

    print(f"Epsilon | Task: {task_instructions[:100]}...", flush=True)

    # Resolve project root for packaged onefile builds that extract to temp dirs.
    manifest_path = os.path.join(project_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        cwd_manifest = os.path.join(os.getcwd(), "manifest.json")
        if os.path.isfile(cwd_manifest):
            project_root = os.getcwd()
            manifest_path = cwd_manifest

    # Load configuration from manifest.json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]

    model_name = os.environ.get("AGENT_MODEL", settings["model"])
    max_iterations = int(os.environ.get("MAX_ITERATIONS", settings["max_iterations"]))
    max_runtime_seconds = int(os.environ.get("MAX_RUNTIME_SECONDS", settings["max_runtime_seconds"]))
    run_indefinitely = _truthy(os.environ.get("RESI_RUN_INDEFINITELY"))
    max_tokens = int(os.environ.get("MAX_TOKENS", settings["max_tokens"]))

    if run_indefinitely:
        print(
            f"\nConfig: {settings_pack_name} | Model: {model_name} | Max: unlimited runtime / unlimited steps",
            flush=True,
        )
    else:
        print(
            f"\nConfig: {settings_pack_name} | Model: {model_name} | Max: {max_runtime_seconds}s / {max_iterations} steps",
            flush=True,
        )

    agents_md_path = os.path.join(os.getcwd(), "Agents.md")
    agents_md = ""
    if os.path.exists(agents_md_path):
        with open(agents_md_path, "r") as f:
            agents_md = f.read()
        print(f"Loaded Agents.md ({len(agents_md)} chars)", flush=True)

    run_id = int(time.time() * 1000)

    shared_workspace = os.environ.get("SHARED_WORKSPACE", "")
    if shared_workspace:
        work_dir = shared_workspace
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = os.path.join(project_root, "runs", str(run_id))
        os.makedirs(work_dir)
    print(f"Working directory: {work_dir}", flush=True)

    # --- Protocol setup ---
    protocol_enabled = os.environ.get("PROTOCOL_ENABLED", "false").lower() == "true"
    protocol_agent = None
    broker = None
    protocol_config = None
    message_buffer = None
    message_lock = None

    if protocol_enabled:
        from agent_protocol.agent import Agent as ProtocolAgent
        from agent_protocol.broker import MessageBroker
        from agent.tool_registry import _protocol_state

        agent_id = os.environ.get("AGENT_ID", f"agent-{run_id}")
        broker_router = os.environ.get("BROKER_ROUTER", os.environ.get("BROKER_PUSH", "tcp://localhost:5555"))
        broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
        broker_mode = os.environ.get("BROKER_MODE", "connect")
        work_queue_enabled = os.environ.get("WORK_QUEUE_ENABLED", "false").lower() == "true"
        heartbeat_interval = float(os.environ.get("PROTOCOL_HEARTBEAT_INTERVAL_SECONDS", "5"))
        topics = [t.strip() for t in os.environ.get("AGENT_TOPICS", "general").split(",")]

        message_buffer = deque()
        message_lock = threading.Lock()

        def message_handler(message):
            with message_lock:
                message_buffer.append(message)

        if broker_mode == "host":
            heartbeat_timeout = float(os.environ.get("BROKER_HEARTBEAT_TIMEOUT_SECONDS", "30"))
            lease_timeout = float(os.environ.get("BROKER_LEASE_TIMEOUT_SECONDS", "60"))
            sweep_interval = float(os.environ.get("BROKER_SWEEP_INTERVAL_SECONDS", "1"))
            max_redeliveries = int(os.environ.get("BROKER_MAX_REDELIVERIES", "5"))
            max_fail_retries = int(os.environ.get("BROKER_MAX_FAIL_RETRIES", "0"))
            backoff_base = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_BASE_SECONDS", "0"))
            backoff_max = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_MAX_SECONDS", "30"))
            broker = MessageBroker(
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
            print(f"Protocol broker started (hosting)", flush=True)

        protocol_agent = ProtocolAgent(
            agent_id=agent_id,
            broker_router=broker_router,
            broker_sub=broker_sub,
            topics=topics,
            message_handler=message_handler,
            enable_logging=False,
            heartbeat_interval_seconds=heartbeat_interval,
            heartbeat_enabled=True,
        )
        protocol_agent.start()
        time.sleep(0.3)

        _protocol_state["agent"] = protocol_agent
        _protocol_state["message_buffer"] = message_buffer
        _protocol_state["message_lock"] = message_lock

        protocol_config = {"agent_id": agent_id, "topics": topics, "work_queue_enabled": work_queue_enabled}

        print(f"Protocol enabled: agent_id={agent_id}, topics={topics}, broker_mode={broker_mode}, work_queue={work_queue_enabled}", flush=True)

    work_queue_enabled = protocol_config.get("work_queue_enabled", False) if protocol_config else False

    worker = Worker(
        user_id=1,
        run_id=run_id,
        user_query=task_instructions,
        worker_number=1,
        model_name=model_name,
        max_tokens=max_tokens,
        work_dir=work_dir,
        agents_md=agents_md,
        protocol_enabled=protocol_enabled,
        protocol_config=protocol_config,
        message_buffer=message_buffer,
        message_lock=message_lock,
        shared_workspace=shared_workspace,
        work_queue_enabled=work_queue_enabled,
    )

    print(f"Running worker with model {model_name}", flush=True)

    start_time = time.time()
    step_count = 0

    while True:
        step_count += 1
        elapsed_seconds = time.time() - start_time

        if not run_indefinitely:
            if step_count > max_iterations:
                print(f"\nSTEP BUDGET REACHED after {max_iterations} steps.", flush=True)
                break

            remaining_seconds = max_runtime_seconds - elapsed_seconds
            if remaining_seconds <= 0:
                print(f"\nTIMEOUT REACHED after {elapsed_seconds:.1f} seconds!", flush=True)
                break

        if run_indefinitely:
            print(f"\n[Step {step_count}] runtime unlimited")
        else:
            print(f"\n[Step {step_count}/{max_iterations}] {remaining_seconds / 60:.0f}m remaining")

        elapsed_time = timedelta(seconds=elapsed_seconds)
        result = worker.run_step(elapsed_time)

        subtask_result = result.get("subtask_result", {})
        if isinstance(subtask_result, dict) and subtask_result.get("tool") == "done":
            print(f"\nAgent finished: {subtask_result.get('stdout', '')}", flush=True)
            break

    # --- Protocol cleanup ---
    if protocol_agent:
        protocol_agent.stop()
    if broker:
        broker.stop()

    total_tokens = worker.total_prompt_tokens + worker.total_response_tokens
    print(
        f"\nTokens: {total_tokens:,} ({worker.total_prompt_tokens:,} in / {worker.total_response_tokens:,} out) | Cost: ${worker.total_cost:.4f} | Steps: {step_count}"
    )
    print("Agent execution completed.", flush=True)


if __name__ == "__main__":
    main()
