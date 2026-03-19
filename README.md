# ε Epsilon

![Epsilon](/image/header.png "Epsilon")

Epsilon is a multi-agent AI coding system for software development and large-scale agent workflows.

It can run:

- a single coding agent on one task
- small collaborative teams with QA and fix loops
- hierarchical team topologies for larger builds
- large manifest-backed workloads with `work_queue`, `sharded_queue`, and `map_reduce`

## What It Is

Epsilon gives you a shared runtime for LLM-powered agents that can:

- write and edit code
- run shell commands
- share a workspace
- coordinate across multiple agents
- retry failed leaf tasks
- combine agent outputs with deterministic reducers

You can use it as a coding assistant, a multi-agent software builder, or a foundation for large-scale agent workflows.

## Why It Is Useful

Most AI tools are good at one-shot help. They are less good at:

- splitting work across many agents
- keeping intermediate outputs structured
- handling failures without restarting everything
- escalating only the tasks that need deeper reasoning

Epsilon is useful when you want a system that can do more than "send one big prompt to one model."

Examples:

- build and test a software project with multiple agents
- split a larger product task into teams or stages
- run hundreds of independent agent tasks over a dataset
- summarize, extract, merge, and adjudicate outputs in multiple phases
- plug in your own agent implementation with BYOA adapter mode

## What You Can Do With It

### Single-Agent Coding

Run one agent directly against a task:

```bash
python runtime/agent_main.py "Build a Flask REST API with SQLite storage for a todo app"
```

### Multi-Agent Software Builds

Use orchestrators to decompose and coordinate work:

```bash
python orchestrate.py --pattern dag "Build a real-time chat app with Flask, Socket.IO, JWT auth, and SQLite"
```

### Large-Scale Agent Workloads

Use manifest-backed topologies when you want large fan-out or hierarchical aggregation:

```bash
python orchestrate.py --pattern sharded_queue --task-manifest manifests/large-job.json
python orchestrate.py --pattern map_reduce --task-manifest manifests/reduce-job.json
```

### Bring Your Own Agent

Keep the orchestration layer and swap in your own agent implementation:

```bash
COLLAB_AGENT_MODE=adapter \
COLLAB_AGENT_ADAPTER_ENTRY="examples/byoa/simple_run_agent.py:run" \
python3 orchestrate.py --pattern dag "Write hello.txt with hello world"
```

## How To Use It

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Set your model credentials in the environment:

```bash
export OPENAI_API_KEY=...
# or
export ANTHROPIC_API_KEY=...
```

### 2. Start With A Simple Run

```bash
python runtime/agent_main.py "Write a Python script that converts CSV to JSON"
```

### 3. Move Up To Multi-Agent Orchestration

```bash
python orchestrate.py --pattern dag "Build a URL shortener service"
python orchestrate.py --pattern tree "Build an e-commerce platform"
python orchestrate.py --pattern pipeline "Build a notes API with staged delivery"
python orchestrate.py --pattern supervisor "Build a service with adaptive recovery"
python orchestrate.py --pattern work_queue "Build a service with pull-based workers"
```

To see supported patterns:

```bash
python orchestrate.py --list-patterns
```

## Choosing A Topology

Use:

- `dag` for parallel build work with QA/fix loops
- `tree` for larger tasks that should be split into teams
- `pipeline` for staged delivery
- `supervisor` for adaptive retries and task splitting
- `work_queue` for pull-based worker execution
- `sharded_queue` for very large independent item sets
- `map_reduce` for large aggregation workloads

`sharded_queue` and `map_reduce` are intentionally manifest-backed. They are meant for explicit large-scale workloads, not free-form task decomposition.

## Featured Demo

The Hugging Face entity-graph demo shows a workload that is more useful than a simple fan-out pipeline:

1. agents summarize and extract entities from many document clusters
2. reducers detect ambiguous entities across documents
3. a second wave of agents adjudicates only those ambiguous cases
4. the finalizer writes a canonical entity graph

Run it with:

```bash
OPENAI_API_KEY=... \
PYTHONPATH=. python examples/hf_entity_graph/run_demo.py \
  --sample-size 100 \
  --sample-mode random \
  --sample-seed 17 \
  --worker-count 8
```

More detail: [examples/hf_entity_graph/README.md](examples/hf_entity_graph/README.md)

## Learn More

- Technical reference: [docs/technical-reference.md](docs/technical-reference.md)
- HF entity graph demo: [examples/hf_entity_graph/README.md](examples/hf_entity_graph/README.md)
- Modular-public adapter example: [examples/modular-public/README.md](examples/modular-public/README.md)
