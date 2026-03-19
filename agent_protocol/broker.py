import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zmq

from .contracts import CoordinationEngine, Delivery, TopologyManager, TransportAdapter
from .messages import Message, MessageType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZmqTransportAdapter(TransportAdapter):
    """ZeroMQ transport implementation for broker ingress/egress."""

    def __init__(self, router_port: int, pub_port: int, bind_address: str = "*"):
        self.router_port = router_port
        self.pub_port = pub_port
        self.bind_address = bind_address

        self._context: Optional[zmq.Context] = None
        self._router_socket: Optional[zmq.Socket] = None
        self._pub_socket: Optional[zmq.Socket] = None
        self._poller: Optional[zmq.Poller] = None

    def start(self) -> None:
        self._context = zmq.Context()

        self._router_socket = self._context.socket(zmq.ROUTER)
        self._router_socket.setsockopt(zmq.LINGER, 0)
        self._router_socket.bind(f"tcp://{self.bind_address}:{self.router_port}")

        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.LINGER, 0)
        self._pub_socket.bind(f"tcp://{self.bind_address}:{self.pub_port}")

        self._poller = zmq.Poller()
        self._poller.register(self._router_socket, zmq.POLLIN)

    def stop(self) -> None:
        if self._router_socket is not None:
            self._router_socket.close()
            self._router_socket = None
        if self._pub_socket is not None:
            self._pub_socket.close()
            self._pub_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._poller = None

    def poll(self, timeout_ms: int) -> List[Delivery]:
        if self._poller is None or self._router_socket is None:
            return []

        deliveries: List[Delivery] = []
        socks = dict(self._poller.poll(timeout=timeout_ms))
        if self._router_socket not in socks:
            return deliveries

        # ROUTER frame shape: [identity, empty_delimiter, message_bytes]
        frames = self._router_socket.recv_multipart()
        identity = frames[0]
        payload = frames[-1]
        try:
            msg = Message.from_bytes(payload)
            deliveries.append(Delivery(identity=identity, message=msg))
        except Exception:
            logger.exception("Failed to decode router message")
        return deliveries

    def send_direct(self, identity: bytes, message: Message) -> None:
        if self._router_socket is None:
            return
        self._router_socket.send_multipart([identity, b"", message.to_bytes()])

    def publish(self, message: Message) -> None:
        if self._pub_socket is None:
            return
        if message.topic:
            prefix = f"{message.topic}:".encode("utf-8")
            self._pub_socket.send_multipart([prefix, message.to_bytes()])
        else:
            self._pub_socket.send(message.to_bytes())


class DefaultTopologyManager(TopologyManager):
    """Default topology: directed messages by target ID, else broadcast."""

    def resolve_direct_target(self, target_agent_id: str, registry: Dict[str, bytes]) -> Optional[bytes]:
        return registry.get(target_agent_id)

    def should_broadcast(self, message: Message) -> bool:
        return not bool(message.target)


class LeaseCoordinationEngine(CoordinationEngine):
    """Heartbeat/liveness + lease-based queue coordination engine."""

    def __init__(
        self,
        topology: TopologyManager,
        enable_logging: bool = True,
        heartbeat_timeout_seconds: float = 30.0,
        lease_timeout_seconds: float = 60.0,
        sweep_interval_seconds: float = 1.0,
        max_redeliveries: int = 5,
        max_fail_retries: int = 0,
        redelivery_backoff_base_seconds: float = 0.0,
        redelivery_backoff_max_seconds: float = 30.0,
    ):
        self.topology = topology
        self.enable_logging = enable_logging
        self.heartbeat_timeout_seconds = max(2.0, float(heartbeat_timeout_seconds))
        self.lease_timeout_seconds = max(5.0, float(lease_timeout_seconds))
        self.sweep_interval_seconds = max(0.2, float(sweep_interval_seconds))
        self.max_redeliveries = max(0, int(max_redeliveries))
        self.max_fail_retries = max(0, int(max_fail_retries))
        self.redelivery_backoff_base_seconds = max(0.0, float(redelivery_backoff_base_seconds))
        self.redelivery_backoff_max_seconds = max(
            self.redelivery_backoff_base_seconds,
            float(redelivery_backoff_max_seconds),
        )

        self._lock = threading.Lock()
        self._last_sweep_at = time.time()

        # Agent registry + liveness
        self.agent_registry: Dict[str, bytes] = {}
        self.agent_topics: Dict[str, List[str]] = {}
        self.agent_last_seen: Dict[str, float] = {}

        # Last value cache by topic
        self.lvc: Dict[str, Message] = {}

        # Queue + lease state
        self.task_queue: deque[str] = deque()
        self.task_log: List[Dict[str, Any]] = []
        self.tasks_by_id: Dict[str, Dict[str, Any]] = {}
        self.active_leases: Dict[str, Dict[str, Any]] = {}
        self.task_to_lease: Dict[str, str] = {}
        self.dead_letter_tasks: Dict[str, Dict[str, Any]] = {}

        self.stats = {
            "messages_received": 0,
            "messages_broadcast": 0,
            "messages_routed": 0,
            "start_time": time.time(),
            "connected_agents": set(),
            "tasks_submitted": 0,
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_redelivered": 0,
            "task_renewals": 0,
            "leases_expired": 0,
            "leases_released_stale_agent": 0,
            "tasks_dead_lettered": 0,
            "tasks_retried_after_fail": 0,
            "heartbeats_received": 0,
            "stale_agents_evicted": 0,
        }

    def on_message(self, identity: bytes, message: Message) -> Iterable[Tuple[str, Any]]:
        with self._lock:
            now = time.time()
            actions: List[Tuple[str, Any]] = []

            self.stats["messages_received"] += 1
            self.stats["connected_agents"].add(message.agent_id)
            self._upsert_presence(identity, message.agent_id, now)

            if self.enable_logging:
                logger.info("Received from %s: %s", message.agent_id, message)

            msg_type = message.message_type
            if msg_type == MessageType.REGISTER:
                actions.extend(self._handle_register(identity, message, now))
            elif msg_type == MessageType.HEARTBEAT:
                self.stats["heartbeats_received"] += 1
            elif msg_type == MessageType.TASK_SUBMIT:
                actions.extend(self._handle_task_submit(identity, message, now))
            elif msg_type == MessageType.TASK_REQUEST:
                actions.extend(self._handle_task_request(identity, message, now))
            elif msg_type == MessageType.TASK_RENEW:
                actions.extend(self._handle_task_renew(identity, message, now))
            elif msg_type == MessageType.TASK_COMPLETE:
                actions.extend(self._handle_task_complete(identity, message, now, failed=False))
            elif msg_type == MessageType.TASK_FAIL:
                actions.extend(self._handle_task_complete(identity, message, now, failed=True))
            elif msg_type == MessageType.CONTROL:
                actions.extend(self._handle_control(identity, message))
            elif message.target:
                actions.extend(self._route_to_target(identity, message))
            elif self.topology.should_broadcast(message):
                actions.extend(self._broadcast(message))

            return actions

    def tick(self) -> Iterable[Tuple[str, Any]]:
        with self._lock:
            now = time.time()
            if now - self._last_sweep_at < self.sweep_interval_seconds:
                return []
            self._last_sweep_at = now

            actions: List[Tuple[str, Any]] = []

            # Evict stale agents and release their leases immediately.
            stale_agents = [
                aid
                for aid, seen_at in self.agent_last_seen.items()
                if now - seen_at > self.heartbeat_timeout_seconds
            ]
            for agent_id in stale_agents:
                self._evict_agent(agent_id, reason="heartbeat_timeout")

            # Requeue expired leases.
            expired_leases = [
                lease_id
                for lease_id, lease in self.active_leases.items()
                if lease["expires_at"] <= now
            ]
            for lease_id in expired_leases:
                self._requeue_lease(lease_id, reason="lease_timeout")

            if stale_agents or expired_leases:
                control = Message(
                    agent_id="broker",
                    message_type=MessageType.CONTROL,
                    topic="broker",
                    payload={
                        "event": "coordination_tick",
                        "stale_agents": stale_agents,
                        "expired_leases": len(expired_leases),
                        "tasks_pending": len(self.task_queue),
                    },
                )
                actions.extend(self._broadcast(control))

            return actions

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_stats_locked()

    def _snapshot_stats_locked(self) -> Dict[str, Any]:
        uptime = time.time() - self.stats["start_time"]
        return {
            "messages_received": self.stats["messages_received"],
            "messages_broadcast": self.stats["messages_broadcast"],
            "messages_routed": self.stats["messages_routed"],
            "connected_agents": len(self.stats["connected_agents"]),
            "unique_agents": list(self.stats["connected_agents"]),
            "registered_agents": list(self.agent_registry.keys()),
            "tasks_submitted": self.stats["tasks_submitted"],
            "tasks_assigned": self.stats["tasks_assigned"],
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "tasks_pending": len(self.task_queue),
            "tasks_inflight": len(self.active_leases),
            "tasks_redelivered": self.stats["tasks_redelivered"],
            "task_renewals": self.stats["task_renewals"],
            "leases_expired": self.stats["leases_expired"],
            "leases_released_stale_agent": self.stats["leases_released_stale_agent"],
            "tasks_dead_lettered": self.stats["tasks_dead_lettered"],
            "tasks_retried_after_fail": self.stats["tasks_retried_after_fail"],
            "dead_letter_queue_size": len(self.dead_letter_tasks),
            "heartbeats_received": self.stats["heartbeats_received"],
            "stale_agents_evicted": self.stats["stale_agents_evicted"],
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "lease_timeout_seconds": self.lease_timeout_seconds,
            "sweep_interval_seconds": self.sweep_interval_seconds,
            "max_redeliveries": self.max_redeliveries,
            "max_fail_retries": self.max_fail_retries,
            "redelivery_backoff_base_seconds": self.redelivery_backoff_base_seconds,
            "redelivery_backoff_max_seconds": self.redelivery_backoff_max_seconds,
            "uptime_seconds": uptime,
        }

    def _ack(self, status: str, payload: Optional[Dict[str, Any]] = None) -> Message:
        body = {"status": status}
        if payload:
            body.update(payload)
        return Message(
            agent_id="broker",
            message_type=MessageType.ACK,
            payload=body,
            topic="control",
        )

    def _upsert_presence(self, identity: bytes, agent_id: str, seen_at: float) -> None:
        self.agent_registry[agent_id] = identity
        self.agent_last_seen[agent_id] = seen_at

    def _handle_register(self, identity: bytes, message: Message, now: float) -> List[Tuple[str, Any]]:
        topics = message.payload.get("subscribed_topics", [])
        if not isinstance(topics, list):
            topics = []
        self.agent_topics[message.agent_id] = [str(t) for t in topics]

        actions: List[Tuple[str, Any]] = []
        actions.append(("direct", (identity, self._ack("registered", {"agent_id": message.agent_id}))))

        # Replay latest topic value for each requested topic.
        for topic in self.agent_topics[message.agent_id]:
            cached = self.lvc.get(topic)
            if not cached:
                continue
            replay = Message(
                agent_id=cached.agent_id,
                message_type=cached.message_type,
                payload=cached.payload,
                topic=cached.topic,
                metadata={**cached.metadata, "lvc_replay": True},
                timestamp=cached.timestamp,
            )
            actions.append(("direct", (identity, replay)))

        actions.extend(self._broadcast(message))
        return actions

    def _route_to_target(self, identity: bytes, message: Message) -> List[Tuple[str, Any]]:
        target_identity = self.topology.resolve_direct_target(message.target, self.agent_registry)
        if target_identity is None:
            nack = self._ack(
                "target_not_found",
                {"target": message.target, "message_id": message.message_id},
            )
            return [("direct", (identity, nack))]

        self.stats["messages_routed"] += 1
        return [("direct", (target_identity, message))]

    def _broadcast(self, message: Message) -> List[Tuple[str, Any]]:
        if message.message_type == MessageType.DATA and message.topic:
            self.lvc[message.topic] = message
        self.stats["messages_broadcast"] += 1
        return [("publish", message)]

    def _handle_control(self, identity: bytes, message: Message) -> List[Tuple[str, Any]]:
        actions: List[Tuple[str, Any]] = []
        if message.payload.get("command") == "stats":
            stats_msg = Message(
                agent_id="broker",
                message_type=MessageType.CONTROL,
                payload={"stats": self._snapshot_stats_locked()},
                topic="broker",
            )
            actions.extend(self._broadcast(stats_msg))

        if self.topology.should_broadcast(message):
            actions.extend(self._broadcast(message))
        return actions

    def _handle_task_submit(self, identity: bytes, message: Message, now: float) -> List[Tuple[str, Any]]:
        task_id = message.message_id
        task_entry: Dict[str, Any] = {
            "task_id": task_id,
            "submitted_by": message.agent_id,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "status": "pending",
            "assigned_to": None,
            "attempts": 0,
            "fail_attempts": 0,
            "redelivery_count": 0,
            "lease_id": None,
            "lease_expires_at": None,
            "available_at": now,
            "updated_at": now,
        }
        self.tasks_by_id[task_id] = task_entry
        self.task_log.append(task_entry)
        self.task_queue.append(task_id)
        self.stats["tasks_submitted"] += 1

        ack = self._ack(
            "task_queued",
            {
                "task_id": task_id,
                "tasks_pending": len(self.task_queue),
            },
        )
        return [("direct", (identity, ack))]

    def _handle_task_request(self, identity: bytes, message: Message, now: float) -> List[Tuple[str, Any]]:
        assignment = self._assign_next_task(worker_id=message.agent_id, now=now)
        if assignment:
            assign_msg = Message(
                agent_id="broker",
                message_type=MessageType.TASK_ASSIGN,
                payload=assignment,
                topic="tasks",
            )
            return [("direct", (identity, assign_msg))]

        no_tasks = Message(
            agent_id="broker",
            message_type=MessageType.TASK_ASSIGN,
            payload={"status": "no_tasks"},
            topic="tasks",
        )
        return [("direct", (identity, no_tasks))]

    def _assign_next_task(self, worker_id: str, now: float) -> Optional[Dict[str, Any]]:
        queue_len = len(self.task_queue)
        for _ in range(queue_len):
            task_id = self.task_queue.popleft()
            task = self.tasks_by_id.get(task_id)
            if not task:
                continue
            if task.get("status") != "pending":
                continue
            if float(task.get("available_at", 0) or 0) > now:
                self.task_queue.append(task_id)
                continue

            lease_id = str(uuid.uuid4())
            expires_at = now + self.lease_timeout_seconds

            task["status"] = "assigned"
            task["assigned_to"] = worker_id
            task["attempts"] = int(task.get("attempts", 0)) + 1
            task["lease_id"] = lease_id
            task["lease_expires_at"] = expires_at
            task["available_at"] = now
            task["updated_at"] = now

            lease = {
                "lease_id": lease_id,
                "task_id": task_id,
                "assigned_to": worker_id,
                "expires_at": expires_at,
                "issued_at": now,
                "renewals": 0,
            }
            self.active_leases[lease_id] = lease
            self.task_to_lease[task_id] = lease_id

            self.stats["tasks_assigned"] += 1
            if task["attempts"] > 1:
                self.stats["tasks_redelivered"] += 1

            return {
                "task_id": task_id,
                "submitted_by": task["submitted_by"],
                "payload": task["payload"],
                "timestamp": task["timestamp"],
                "status": "assigned",
                "assigned_to": worker_id,
                "attempt": task["attempts"],
                "lease_id": lease_id,
                "lease_timeout_seconds": self.lease_timeout_seconds,
                "lease_expires_at": expires_at,
                "redelivery_count": int(task.get("redelivery_count", 0)),
            }

        return None

    def _handle_task_renew(self, identity: bytes, message: Message, now: float) -> List[Tuple[str, Any]]:
        payload = message.payload if isinstance(message.payload, dict) else {}
        task_id = payload.get("task_id")
        lease_id = payload.get("lease_id")
        if not task_id or not lease_id:
            return [("direct", (identity, self._ack("lease_invalid_request", {"task_id": task_id})))]

        lease = self.active_leases.get(lease_id)
        if not lease or lease.get("task_id") != task_id:
            return [("direct", (identity, self._ack("lease_not_found", {"task_id": task_id, "lease_id": lease_id})))]

        if lease.get("assigned_to") != message.agent_id:
            return [("direct", (identity, self._ack("lease_owner_mismatch", {"task_id": task_id, "lease_id": lease_id})))]

        if lease.get("expires_at", 0) <= now:
            self._requeue_lease(lease_id, reason="lease_timeout")
            return [("direct", (identity, self._ack("lease_expired", {"task_id": task_id, "lease_id": lease_id})))]

        lease["renewals"] = int(lease.get("renewals", 0)) + 1
        lease["expires_at"] = now + self.lease_timeout_seconds
        self.stats["task_renewals"] += 1

        task = self.tasks_by_id.get(task_id)
        if task:
            task["lease_expires_at"] = lease["expires_at"]
            task["updated_at"] = now

        ack = self._ack(
            "lease_renewed",
            {
                "task_id": task_id,
                "lease_id": lease_id,
                "lease_expires_at": lease["expires_at"],
            },
        )
        return [("direct", (identity, ack))]

    def _handle_task_complete(self, identity: bytes, message: Message, now: float, failed: bool) -> List[Tuple[str, Any]]:
        payload = message.payload if isinstance(message.payload, dict) else {}
        task_id = payload.get("task_id")
        lease_id = payload.get("lease_id")

        if not task_id:
            return [("direct", (identity, self._ack("task_id_required")))]

        task = self.tasks_by_id.get(task_id)
        if task is None:
            return [("direct", (identity, self._ack("task_not_found", {"task_id": task_id})))]

        if task.get("status") in {"completed", "failed"}:
            return [("direct", (identity, self._ack("task_already_finalized", {"task_id": task_id})))]

        current_lease_id = self.task_to_lease.get(task_id)
        if int(task.get("attempts", 0)) > 0 and not current_lease_id:
            return [
                (
                    "direct",
                    (
                        identity,
                        self._ack(
                            "lease_not_found",
                            {"task_id": task_id, "lease_id": lease_id},
                        ),
                    ),
                )
            ]

        if current_lease_id:
            if lease_id and current_lease_id != lease_id:
                return [
                    (
                        "direct",
                        (
                            identity,
                            self._ack(
                                "lease_mismatch",
                                {
                                    "task_id": task_id,
                                    "lease_id": lease_id,
                                    "expected_lease_id": current_lease_id,
                                },
                            ),
                        ),
                    )
                ]

            lease = self.active_leases.get(current_lease_id)
            if lease and lease.get("assigned_to") != message.agent_id:
                return [
                    (
                        "direct",
                        (
                            identity,
                            self._ack(
                                "lease_owner_mismatch",
                                {"task_id": task_id, "lease_id": current_lease_id},
                            ),
                        ),
                    )
                ]

            self.active_leases.pop(current_lease_id, None)
            self.task_to_lease.pop(task_id, None)

        task["status"] = "failed" if failed else "completed"
        task["updated_at"] = now
        task["completed_by"] = message.agent_id
        task["result"] = payload.get("result")
        task["error"] = payload.get("error")
        task["lease_id"] = None
        task["lease_expires_at"] = None

        if failed:
            task["fail_attempts"] = int(task.get("fail_attempts", 0)) + 1
            if task["fail_attempts"] <= self._task_max_fail_retries(task):
                task["status"] = "pending"
                task["assigned_to"] = None
                task["result"] = None
                task["last_requeue_reason"] = "task_fail"
                task["redelivery_count"] = int(task.get("redelivery_count", 0)) + 1
                if task["redelivery_count"] > self._task_max_redeliveries(task):
                    self._move_to_dead_letter(task, "task_fail:redelivery_limit_exceeded", now)
                    return [("direct", (identity, self._ack("task_dead_lettered", {"task_id": task_id})))]

                delay = self._redelivery_delay_seconds(task["redelivery_count"])
                task["available_at"] = now + delay
                if delay > 0:
                    self.task_queue.append(task_id)
                else:
                    self.task_queue.appendleft(task_id)
                self.stats["tasks_retried_after_fail"] += 1
                return [
                    (
                        "direct",
                        (
                            identity,
                            self._ack(
                                "task_requeued",
                                {
                                    "task_id": task_id,
                                    "reason": "task_fail",
                                    "fail_attempts": task["fail_attempts"],
                                },
                            ),
                        ),
                    )
                ]
            self.stats["tasks_failed"] += 1
            status = "task_failed"
        else:
            self.stats["tasks_completed"] += 1
            status = "task_completed"

        return [("direct", (identity, self._ack(status, {"task_id": task_id})))]

    def _evict_agent(self, agent_id: str, reason: str) -> None:
        self.agent_registry.pop(agent_id, None)
        self.agent_topics.pop(agent_id, None)
        self.agent_last_seen.pop(agent_id, None)
        self.stats["stale_agents_evicted"] += 1

        stale_leases = [
            lease_id
            for lease_id, lease in self.active_leases.items()
            if lease.get("assigned_to") == agent_id
        ]
        for lease_id in stale_leases:
            self._requeue_lease(lease_id, reason=f"worker_stale:{reason}")
            self.stats["leases_released_stale_agent"] += 1

    def _requeue_lease(self, lease_id: str, reason: str) -> None:
        lease = self.active_leases.pop(lease_id, None)
        if not lease:
            return

        task_id = lease["task_id"]
        self.task_to_lease.pop(task_id, None)

        task = self.tasks_by_id.get(task_id)
        if not task:
            return
        if task.get("status") == "completed":
            return

        task["status"] = "pending"
        task["assigned_to"] = None
        task["lease_id"] = None
        task["lease_expires_at"] = None
        now = time.time()
        task["updated_at"] = now
        task["redelivery_count"] = int(task.get("redelivery_count", 0)) + 1
        task["last_requeue_reason"] = reason

        if reason.startswith("lease_timeout"):
            self.stats["leases_expired"] += 1
        if task["redelivery_count"] > self._task_max_redeliveries(task):
            self._move_to_dead_letter(task, reason, now)
            return

        delay = self._redelivery_delay_seconds(task["redelivery_count"])
        task["available_at"] = now + delay
        if delay > 0:
            self.task_queue.append(task_id)
        else:
            self.task_queue.appendleft(task_id)

    def _task_max_redeliveries(self, task: Dict[str, Any]) -> int:
        payload = task.get("payload")
        if isinstance(payload, dict):
            override = payload.get("max_redeliveries")
            if override is not None:
                try:
                    return max(0, int(override))
                except (TypeError, ValueError):
                    pass
        return self.max_redeliveries

    def _task_max_fail_retries(self, task: Dict[str, Any]) -> int:
        payload = task.get("payload")
        if isinstance(payload, dict):
            override = payload.get("max_fail_retries")
            if override is not None:
                try:
                    return max(0, int(override))
                except (TypeError, ValueError):
                    pass
        return self.max_fail_retries

    def _redelivery_delay_seconds(self, redelivery_count: int) -> float:
        if self.redelivery_backoff_base_seconds <= 0:
            return 0.0
        exponent = max(0, int(redelivery_count) - 1)
        delay = self.redelivery_backoff_base_seconds * (2 ** exponent)
        return min(delay, self.redelivery_backoff_max_seconds)

    def _move_to_dead_letter(self, task: Dict[str, Any], reason: str, now: float) -> None:
        task_id = task["task_id"]
        task["status"] = "failed"
        task["assigned_to"] = None
        task["lease_id"] = None
        task["lease_expires_at"] = None
        task["available_at"] = now
        task["updated_at"] = now
        task["dead_letter_reason"] = reason
        task["dead_lettered_at"] = now
        self.dead_letter_tasks[task_id] = task
        self.stats["tasks_failed"] += 1
        self.stats["tasks_dead_lettered"] += 1


class MessageBroker:
    def __init__(
        self,
        router_port: int = 5555,
        pub_port: int = 5556,
        bind_address: str = "*",
        enable_logging: bool = True,
        pull_port: int = None,  # backward compat alias
        heartbeat_timeout_seconds: float = 30.0,
        lease_timeout_seconds: float = 60.0,
        sweep_interval_seconds: float = 1.0,
        max_redeliveries: int = 5,
        max_fail_retries: int = 0,
        redelivery_backoff_base_seconds: float = 0.0,
        redelivery_backoff_max_seconds: float = 30.0,
    ):
        self.router_port = pull_port or router_port
        self.pub_port = pub_port
        self.bind_address = bind_address
        self.enable_logging = enable_logging

        self.transport: TransportAdapter = ZmqTransportAdapter(
            router_port=self.router_port,
            pub_port=self.pub_port,
            bind_address=self.bind_address,
        )
        self.topology: TopologyManager = DefaultTopologyManager()
        self.coordination: CoordinationEngine = LeaseCoordinationEngine(
            topology=self.topology,
            enable_logging=enable_logging,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            lease_timeout_seconds=lease_timeout_seconds,
            sweep_interval_seconds=sweep_interval_seconds,
            max_redeliveries=max_redeliveries,
            max_fail_retries=max_fail_retries,
            redelivery_backoff_base_seconds=redelivery_backoff_base_seconds,
            redelivery_backoff_max_seconds=redelivery_backoff_max_seconds,
        )

        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        logger.info("Starting message broker on ROUTER:%s, PUB:%s", self.router_port, self.pub_port)
        self.transport.start()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Message broker started successfully")

    def _run(self) -> None:
        while self.running:
            deliveries = self.transport.poll(timeout_ms=100)
            for delivery in deliveries:
                actions = self.coordination.on_message(delivery.identity, delivery.message)
                self._apply_actions(actions)

            tick_actions = self.coordination.tick()
            self._apply_actions(tick_actions)

    def _apply_actions(self, actions: Iterable[Tuple[str, Any]]) -> None:
        for action, payload in actions:
            if action == "direct":
                identity, message = payload
                self.transport.send_direct(identity, message)
            elif action == "publish":
                self.transport.publish(payload)

    def stop(self) -> None:
        logger.info("Stopping message broker...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.transport.stop()
        logger.info("Message broker stopped")

    def get_stats(self) -> Dict[str, Any]:
        return self.coordination.get_stats()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
