# Minimal Telemetry Schema For A Future Control Plane

This document defines the smallest useful telemetry contract for a future Epsilon web control plane.

The goal is not to instrument everything now. The goal is to define a stable event envelope that the runtime can emit later without forcing a redesign of the product surface.

## Design Goals

The control plane should be able to answer:

- what ran?
- which topology was used?
- what stage or task is in progress?
- what failed or retried?
- what artifacts were produced?
- which worker executed the work?
- how many tokens and how much estimated cost were consumed?

## Event Families

The initial schema only needs these event types:

- `run.started`
- `run.completed`
- `run.failed`
- `stage.started`
- `stage.completed`
- `stage.failed`
- `task.enqueued`
- `task.started`
- `task.completed`
- `task.failed`
- `task.retried`
- `artifact.produced`
- `worker.heartbeat`
- `usage.reported`

## Event Envelope

Every event should include:

- `schema_version`
- `event_id`
- `event_type`
- `emitted_at`
- `run`
- `source`

Optional payload sections are attached when relevant:

- `stage`
- `task`
- `worker`
- `artifact`
- `usage`
- `error`
- `metadata`

## Example

```json
{
  "schema_version": "2026-03-20",
  "event_id": "evt_01HZZZZZZZZZZZZZZZZZZZZZZ",
  "event_type": "task.completed",
  "emitted_at": "2026-03-20T20:11:43Z",
  "run": {
    "run_id": "run_20260320_001",
    "workflow_name": "benchmark_scout",
    "topology": "map_reduce"
  },
  "source": {
    "component": "worker_daemon",
    "host": "worker-08"
  },
  "stage": {
    "stage_id": "phase1",
    "stage_name": "extraction"
  },
  "task": {
    "task_id": "map-paper-00042",
    "task_type": "map",
    "attempt": 1,
    "status": "success"
  },
  "worker": {
    "worker_id": "worker-08"
  },
  "artifact": {
    "artifact_id": "artifact_map_00042",
    "path": "results/maps/paper-00042.json",
    "kind": "json"
  },
  "usage": {
    "model": "openai/gpt-5.2",
    "prompt_tokens": 1842,
    "response_tokens": 611,
    "total_tokens": 2453,
    "estimated_cost_usd": 0.0041
  }
}
```

## JSON Schema

The machine-readable schema lives at:

- [schemas/control-plane-event.schema.json](schemas/control-plane-event.schema.json)

## Implementation Guidance

When runtime instrumentation is added later, prefer these rules:

- emit events at orchestration boundaries, not every internal function call
- keep artifact references explicit
- include stable `run_id`, `stage_id`, and `task_id` values
- report usage/cost close to the step that incurred it
- make retries first-class instead of inferring them from log parsing

## Non-Goals

This schema does not attempt to model:

- full distributed tracing
- provider-specific raw response payloads
- desktop-only UI events
- every intermediate debug log line

It is intentionally minimal. The point is to support a practical v1 control plane focused on run visibility, lineage, cost, and governance.
