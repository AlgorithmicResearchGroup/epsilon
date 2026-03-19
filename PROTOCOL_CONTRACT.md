# Multi-Agent Protocol Contract

This project treats collaboration as three separable planes so topologies can evolve without rewriting agent logic.

## Planes

1. **Transport plane**
- Interface: `TransportAdapter` (`agent_protocol/contracts.py`)
- Current implementation: `ZmqTransportAdapter`
- Responsibility: send/receive envelopes only.

2. **Topology plane**
- Interface: `TopologyManager`
- Current implementation: `DefaultTopologyManager`
- Responsibility: resolve directed targets and broadcast policy.

3. **Coordination plane**
- Interface: `CoordinationEngine`
- Current implementation: `LeaseCoordinationEngine`
- Responsibility: heartbeats, liveness, queue leases, renewal, timeout redelivery.

## Reliability Semantics (Phase A/B)

- Task delivery is **at-least-once**.
- Queue assignments are lease-based (`lease_id`, `lease_timeout_seconds`, `lease_expires_at`).
- Workers should renew leases while processing (`TASK_RENEW`).
- On timeout, broker requeues task for redelivery.
- Redelivery is bounded by `max_redeliveries` (global or per-task payload override).
- Explicit worker failures (`TASK_FAIL`) can be retried with `max_fail_retries` (global or per-task payload override).
- Tasks exceeding redelivery limits are dead-lettered (`status=failed`).
- Agents emit periodic heartbeats (`HEARTBEAT`); stale agents are evicted after broker timeout.
- When stale agents are evicted, their active leases are released and requeued.

## Message Types

- Existing: `DATA`, `CONTROL`, `HEARTBEAT`, `DISCOVERY`, `ACK`, `REGISTER`, `TASK_SUBMIT`, `TASK_REQUEST`, `TASK_ASSIGN`, `TASK_COMPLETE`
- Added for lease lifecycle: `TASK_RENEW`, `TASK_FAIL`

## Backward Compatibility

- Agent APIs used by existing tools remain valid:
- `send_message`, `send_data`, `submit_task`, `request_task`, `complete_task`
- New APIs are additive:
- `renew_task(task_id, lease_id)`
- `fail_task(task_id, error, lease_id=None)`
- `complete_task(..., lease_id=None)` accepts optional lease.

## Tuning Knobs

- `PROTOCOL_HEARTBEAT_INTERVAL_SECONDS` (agent side)
- `BROKER_HEARTBEAT_TIMEOUT_SECONDS`
- `BROKER_LEASE_TIMEOUT_SECONDS`
- `BROKER_SWEEP_INTERVAL_SECONDS`
- `BROKER_MAX_REDELIVERIES`
- `BROKER_MAX_FAIL_RETRIES`
- `BROKER_REDELIVERY_BACKOFF_BASE_SECONDS`
- `BROKER_REDELIVERY_BACKOFF_MAX_SECONDS`

## Next Steps (Phase C+)

- Idempotency window keyed by `task_id`/`message_id`
- Topology plugins (tree, mesh, org-chart, cohort bridges)
- Federation controls (namespace/auth/admission)

## BYOA Adapter (Process Bridge)

`runtime/byoa_runner.py` allows an external process to participate in the protocol without importing internal worker code.

- Runner input: task text (argv/env) + standard protocol env (`AGENT_ID`, `BROKER_ROUTER`, `BROKER_SUB`, `SHARED_WORKSPACE`).
- Child stdin first line: `{"type":"run_context", ...}`.
- Child stdout commands (JSON lines):
  - `log`
  - `send_message`
  - `check_messages` (runner writes back `{"type":"messages","messages":[...]}`)
  - `done` / `fail`
  - optional queue actions: `submit_task`, `request_task`, `renew_task`, `complete_task`, `fail_task`

Enable from orchestrators with:
- `COLLAB_AGENT_MODE=adapter`
- `COLLAB_AGENT_ADAPTER_CMD="..."` for low-level action-protocol adapters
- `COLLAB_AGENT_ADAPTER_ENTRY="<module_or_file>:<run_function>"` for simple function-style adapters
