"""Protocol contracts for transport, topology, and coordination layers.

These interfaces are intentionally narrow so transports/topologies can evolve
without changing the agent-facing API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .messages import Message


@dataclass
class Delivery:
    """Inbound broker delivery from transport."""

    identity: bytes
    message: Message


class TransportAdapter(ABC):
    """Moves bytes/messages; no routing or reliability policy."""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def poll(self, timeout_ms: int) -> List[Delivery]:
        raise NotImplementedError

    @abstractmethod
    def send_direct(self, identity: bytes, message: Message) -> None:
        raise NotImplementedError

    @abstractmethod
    def publish(self, message: Message) -> None:
        raise NotImplementedError


class TopologyManager(ABC):
    """Decides how messages route across participants."""

    @abstractmethod
    def resolve_direct_target(self, target_agent_id: str, registry: Dict[str, bytes]) -> Optional[bytes]:
        raise NotImplementedError

    @abstractmethod
    def should_broadcast(self, message: Message) -> bool:
        raise NotImplementedError


class CoordinationEngine(ABC):
    """Owns liveness, leases, redelivery, and queue semantics."""

    @abstractmethod
    def on_message(self, identity: bytes, message: Message) -> Iterable[Tuple[str, Any]]:
        """Return broker actions, e.g. ('direct', (identity, msg)), ('publish', msg)."""
        raise NotImplementedError

    @abstractmethod
    def tick(self) -> Iterable[Tuple[str, Any]]:
        """Periodic maintenance actions."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError
