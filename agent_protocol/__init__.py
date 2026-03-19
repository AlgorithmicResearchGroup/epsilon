"""
Multi-agent communication protocol library using ZeroMQ
"""

from .broker import MessageBroker
from .agent import Agent
from .messages import Message, MessageType
from .contracts import TransportAdapter, TopologyManager, CoordinationEngine

__version__ = "0.1.0"
__all__ = [
    "MessageBroker",
    "Agent",
    "Message",
    "MessageType",
    "TransportAdapter",
    "TopologyManager",
    "CoordinationEngine",
]
