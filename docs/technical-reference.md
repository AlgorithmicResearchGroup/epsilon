# Technical Reference

This file holds the lower-level operational and implementation details that were moved out of the main `README.md`.

For the higher-level launch surfaces, see:

- [architecture.md](architecture.md)
- [release.md](release.md)
- [control-plane-telemetry.md](control-plane-telemetry.md)

## Pattern Details

### Multi-Agent With QA Loop (`dag`)

```bash
python orchestrate.py --pattern dag "Build a real-time chat app with Flask, Socket.IO, JWT auth, and SQLite"
```

Fully containerized:

```bash
docker build -t epsilon .
COLLAB_EXECUTOR=docker COLLAB_DOCKER_IMAGE=epsilon \
  python orchestrate.py --pattern dag "Build a real-time chat app with Flask, Socket.IO, JWT auth, and SQLite"
```

The orchestrator will:

1. decompose the task into a typed DAG via an LLM call
2. execute nodes in dependency-ordered waves
3. run a QA agent
4. assign failures back to responsible agents
5. repeat until QA passes or `MAX_WAVES` is exhausted

### Hierarchical Tree Orchestrator

For large tasks, `tree` decomposes work into 2-8 teams, each with its own sub-orchestrator and git branch.

```bash
python orchestrate.py --pattern tree "Build an e-commerce platform with product catalog, cart, checkout, and admin dashboard"
python orchestrate.py --pattern tree --prompts challenge_prompts.json --prompt 5
```

The tree orchestrator will:

1. decompose the task into teams
2. create a git branch per team
3. run each team in parallel
4. merge all branches
5. run integration QA

### Pattern Selection

```bash
python orchestrate.py --pattern dag "Build a URL shortener service"
python orchestrate.py --pattern tree "Build an e-commerce platform"
python orchestrate.py --pattern pipeline "Build a notes API with staged delivery"
python orchestrate.py --pattern supervisor "Build a notes API and adaptively recover from failed stages"
python orchestrate.py --pattern work_queue "Build a notes API with pull-based worker daemons"
python orchestrate.py --pattern sharded_queue --task-manifest manifests/large-job.json
python orchestrate.py --pattern map_reduce --task-manifest manifests/reduce-job.json
python orchestrate.py --list-patterns
```

## Bring Your Own Agent (BYOA Adapter Mode)

You can plug in your own agent process without changing orchestrators.

Recommended function-entrypoint path:

```bash
COLLAB_AGENT_MODE=adapter \
COLLAB_AGENT_ADAPTER_ENTRY="examples/byoa/simple_run_agent.py:run" \
python3 orchestrate.py --pattern dag "Write hello.txt with hello world"
```

Advanced stdin/stdout action-protocol path:

```bash
COLLAB_AGENT_MODE=adapter \
COLLAB_AGENT_ADAPTER_CMD="python3 examples/byoa/while_loop_agent.py" \
python3 orchestrate.py --pattern dag "Write hello.txt with hello world"
```

Supported adapter actions:

- `{"action":"log","message":"..."}`
- `{"action":"send_message","content":"...","topic":"general"}`
- `{"action":"check_messages","limit":25}`
- `{"action":"done","summary":"..."}`
- `{"action":"fail","error":"..."}`

Helper SDK:

- `runtime/byoa_sdk.py`
- `examples/byoa/while_loop_agent.py`
- `examples/byoa/simple_run_agent.py`

Modular-public example:

```bash
COLLAB_AGENT_MODE=adapter \
COLLAB_AGENT_ADAPTER_ENTRY="examples/byoa/modular_public_adapter.py:run" \
MODULAR_MAX_STEPS=5 \
python3 orchestrate.py --pattern dag "create hello.txt with hello world"
```

## Scale Benchmark Harness

Start a benchmark run:

```bash
python scripts/run_scale_benchmark.py \
  --benchmark wiki \
  --task-count 300 \
  --executor direct_wiki \
  --start-broker \
  --broker-router tcp://<broker-host>:5555 \
  --broker-sub tcp://<broker-host>:5556
```

Start worker daemons:

```bash
python runtime/worker_daemon.py \
  --worker-id worker-01 \
  --broker-router tcp://<broker-host>:5555 \
  --broker-sub tcp://<broker-host>:5556 \
  --max-concurrent-local 1
```

Outputs:

- `scale_report.json`
- `task_results.ndjson`
- `queue_samples.ndjson`

Benchmark modes:

- `--benchmark wiki`
- `--benchmark compiler`

Executors:

- `--executor direct_wiki`
- `--executor agent`

Tier runner:

```bash
scripts/run_scale_tiers.sh
```

Useful env overrides:

- `TIERS_CSV=3,10,20`
- `TASKS_PER_AGENT=5`
- `EXECUTOR=direct_wiki`

## Architecture

For the current overview diagrams and component framing, start with [architecture.md](architecture.md). The ASCII sketch below is kept here as an implementation-oriented reference.

```text
┌──────────────────────────────────────────────────────────┐
│            orchestrators/tree_orchestrator.py           │
│          (team decomposition + git branching)           │
│                                                          │
│  Decompose → team branches → merge → integration QA     │
└─────────┬────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│          orchestrators/dag_orchestrator.py     │
│         (orchestrator + wave loop)             │
│                                                 │
│  Decompose → BUILD wave → QA wave → FIX wave   │
└──────────┬──────────────────────────┬───────────┘
           │                          │
    ┌──────▼──────┐           ┌───────▼───────┐
    │ MessageBroker│           │ Shared        │
    │ ROUTER :5555 │           │ Workspace     │
    │ PUB    :5556 │           │ runs/shared-* │
    └──────┬──────┘           └───────────────┘
           │
    ┌──────┼──────────┬───────────┐
    ▼      ▼          ▼           ▼
  Alice   Bob      Charlie      Dave
 (DEALER) (DEALER)  (DEALER)   (DEALER)
  + SUB    + SUB     + SUB      + SUB
```

Each agent is an independent subprocess running `runtime/agent_main.py` with:

- `DEALER` socket to the broker router
- `SUB` socket to the broker publisher
- a shared filesystem workspace

## Agent Tools

| Tool | Description |
|------|-------------|
| `run_bash` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Create/overwrite files |
| `edit_file` | Surgical string replacement |
| `sql_query` | Parameterized SQL query execution |
| `web_search` | Search the web |
| `fetch_url` | Fetch URL contents |
| `call_llm` | Call an allowlisted delegate LLM |
| `plan` | Enter planning mode |
| `submit_plan` | Submit subtasks after planning |
| `mark_complete` | Advance to the next subtask |
| `done` | Signal task completion |
| `send_message` | Broadcast or direct a message |
| `check_messages` | Receive messages |
| `submit_task` | Add task to the shared queue |
| `request_task` | Pull the next queue task |

## Configuration

### `manifest.json`

```json
{
  "defaultSettingsPack": "default",
  "settingsPacks": {
    "default": {
      "model": "openai/gpt-5.2",
      "max_iterations": 100,
      "max_runtime_seconds": 600,
      "max_tokens": 4096
    },
    "anthropic": {
      "model": "anthropic/claude-opus-4-6",
      "max_iterations": 100,
      "max_runtime_seconds": 600,
      "max_tokens": 4096
    }
  },
  "delegate_llm": {
    "enabled": true,
    "default_model": "openai/gpt-5.2",
    "allowed_models": [
      "openai/gpt-5.2",
      "anthropic/claude-opus-4-6"
    ]
  }
}
```

`call_llm` only allows models listed in `delegate_llm.allowed_models`.

### Environment Variables

#### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI` | — | OpenAI API key |
| `ANTHROPIC` | — | Anthropic API key |
| `SETTINGS_PACK` | `default` | Config pack from `manifest.json` |
| `AGENT_MODEL` | from pack | LiteLLM model override |
| `ORCHESTRATOR_MODEL` | from pack | Model override for decomposition/review calls |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout for each model call |
| `LLM_MAX_RETRIES` | `2` | Retries per model call |
| `LLM_API_BASE` | unset | Optional LiteLLM API base override |
| `SQL_DATABASE_URL` | unset | Default SQLAlchemy DB URL |
| `MAX_ITERATIONS` | `100` | Max tool calls per agent |
| `MAX_RUNTIME_SECONDS` | `600` | Hard timeout per agent |
| `SHARED_WORKSPACE` | auto | Shared directory path |

#### Multi-Agent Protocol

| Variable | Default | Description |
|----------|---------|-------------|
| `PROTOCOL_ENABLED` | `false` | Enable ZeroMQ messaging |
| `AGENT_ID` | auto | Unique agent identifier |
| `BROKER_MODE` | — | `host` or `connect` |
| `BROKER_ROUTER` | `tcp://localhost:5555` | Broker router address |
| `BROKER_SUB` | `tcp://localhost:5556` | Broker pub address |
| `AGENT_TOPICS` | `general` | Subscription topics |
| `WORK_QUEUE_ENABLED` | `false` | Enable work queue tools |
| `PROTOCOL_HEARTBEAT_INTERVAL_SECONDS` | `5` | Agent heartbeat interval |
| `BROKER_HEARTBEAT_TIMEOUT_SECONDS` | `30` | Broker liveness timeout |
| `BROKER_LEASE_TIMEOUT_SECONDS` | `60` | Task lease timeout |
| `BROKER_SWEEP_INTERVAL_SECONDS` | `1` | Broker maintenance sweep interval |
| `BROKER_MAX_REDELIVERIES` | `5` | Max redeliveries before dead-letter |
| `BROKER_MAX_FAIL_RETRIES` | `0` | Max retries after explicit `TASK_FAIL` |
| `BROKER_REDELIVERY_BACKOFF_BASE_SECONDS` | `0` | Redelivery backoff base |
| `BROKER_REDELIVERY_BACKOFF_MAX_SECONDS` | `30` | Max redelivery backoff |

#### Orchestrator

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WAVES` | `3` | QA/fix retry waves |
| `QA_ITERATIONS` | `30` | QA agent iteration budget |
| `FIX_ITERATIONS` | `15` | Fix agent iteration budget |
| `FIX_RUNTIME_SECONDS` | `120` | Fix agent runtime budget |
| `ORCHESTRATOR_MODEL` | from pack | Model for task decomposition |
| `COLLAB_EXECUTOR` | `host` | `host` or `docker` backend |
| `COLLAB_DOCKER_IMAGE` | `epsilon` | Docker image |
| `COLLAB_DOCKER_AUTO_BUILD` | `0` | Auto-build missing image |
| `COLLAB_DOCKER_USER` | unset | Optional container user override |
| `COLLAB_AGENT_MAIN_BINARY` | unset | Native agent binary override |
| `COLLAB_BYOA_RUNNER_BINARY` | unset | BYOA runner binary override |
| `COLLAB_DOCKER_AGENT_MAIN_BINARY` | `/home/agent/bin/agent-main` | Docker agent entrypoint |
| `COLLAB_DOCKER_BYOA_RUNNER_BINARY` | `/home/agent/bin/byoa-runner` | Docker BYOA entrypoint |

#### Tree Orchestrator

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WAVES` | `2` | QA/fix waves per team |
| `INTEGRATION_WAVES` | `2` | Integration QA waves after merge |

## QA Loop

When `MAX_WAVES > 0`, the orchestrator runs a QA agent after the build wave. The QA agent:

1. reads source files
2. installs dependencies
3. runs tests
4. starts the server and exercises endpoints
5. checks for common integration mistakes
6. writes `qa_report.json`

If QA fails, the orchestrator assigns errors back to responsible agents, reruns fix tasks, and repeats until QA passes or the wave budget is exhausted.

## Messaging Protocol

The protocol is split into three planes:

- transport plane: ZeroMQ sockets move bytes
- topology plane: routing policy decides delivery
- coordination plane: heartbeats, leases, renewals, and redelivery

Current reliability semantics:

- at-least-once delivery for work queue tasks
- lease-based queue ownership
- heartbeat-driven liveness eviction
- dead-letter protection for poison tasks
- bounded retries for explicit task failure
- broadcast and directed messaging
- last-value cache replay for topic state

Detailed contract: `PROTOCOL_CONTRACT.md`

## Examples

```bash
python runtime/agent_main.py "Write a Python script that converts CSV to JSON"
MAX_WAVES=0 python orchestrate.py --pattern dag "Build a URL shortener with Flask and SQLite"
MAX_ITERATIONS=80 MAX_RUNTIME_SECONDS=600 MAX_WAVES=2 \
  python orchestrate.py --pattern dag "Build a real-time chat app with Flask, Socket.IO, JWT auth, SQLite, and an HTML client"
python orchestrate.py --pattern tree "Build an e-commerce platform with catalog, cart, checkout, and admin"
python orchestrate.py --pattern pipeline "Build a data ETL service with validation gates"
python orchestrate.py --pattern supervisor "Build a service with dynamic stage supervision"
python orchestrate.py --pattern work_queue "Build a service with queue-based execution and worker daemons"
python orchestrate.py --pattern sharded_queue --task-manifest manifests/large-job.json
python orchestrate.py --pattern map_reduce --task-manifest manifests/reduce-job.json
python orchestrate.py --pattern tree --prompts challenge_prompts.json --prompt 3
SETTINGS_PACK=anthropic python orchestrate.py --pattern dag "Build a REST API for a bookstore"
```

## Docker

Build the image:

```bash
docker build -t epsilon .
```

The image includes common CLI tooling and compiled agent binaries at `/home/agent/bin/agent-main` and `/home/agent/bin/byoa-runner`.

Examples:

```bash
docker run --env-file .env epsilon "Build a URL shortener microservice"
```

```bash
COLLAB_EXECUTOR=docker COLLAB_DOCKER_IMAGE=epsilon \
  python orchestrate.py --pattern dag "Build a URL shortener microservice"
```

```bash
python orchestrate.py --pattern dag \
  --executor docker \
  --docker-image epsilon \
  "Build a URL shortener microservice"
```

```bash
COLLAB_EXECUTOR=docker COLLAB_DOCKER_IMAGE=epsilon \
  python orchestrate.py --pattern dag --prompts challenge_prompts.json --prompt 2
```

## Publish To GHCR

Local publish:

```bash
export GHCR_TOKEN=...
export GHCR_IMAGE=ghcr.io/algorithmicresearchgroup/multiagent
export GHCR_TAG=latest

scripts/publish_ghcr.sh
```

GitHub Actions publish:

- workflow: `.github/workflows/publish.yaml`
- triggers:
  - push to `main` or `master`
  - tag push `v*`
  - manual dispatch
- default image: `ghcr.io/<owner>/<repo>` (lowercased)

## Project Structure

```text
├── orchestrate.py
├── orchestrators/
│   ├── dag_orchestrator.py
│   ├── pipeline_orchestrator.py
│   ├── supervisor_orchestrator.py
│   ├── tree_orchestrator.py
│   ├── work_queue_orchestrator.py
│   ├── sharded_queue_orchestrator.py
│   ├── map_reduce_orchestrator.py
│   └── patterns.py
├── runtime/
│   ├── agent_main.py
│   └── worker_daemon.py
├── scripts/
│   └── run_scale_benchmark.py
├── manifest.json
├── challenge_prompts.json
├── Dockerfile
├── agent/
├── agent_protocol/
└── runs/
```
