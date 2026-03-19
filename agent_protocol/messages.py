import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(Enum):
    DATA = "DATA"
    CONTROL = "CONTROL"
    HEARTBEAT = "HEARTBEAT"
    DISCOVERY = "DISCOVERY"
    ACK = "ACK"
    REGISTER = "REGISTER"
    TASK_SUBMIT = "TASK_SUBMIT"
    TASK_REQUEST = "TASK_REQUEST"
    TASK_ASSIGN = "TASK_ASSIGN"
    TASK_RENEW = "TASK_RENEW"
    TASK_COMPLETE = "TASK_COMPLETE"
    TASK_FAIL = "TASK_FAIL"


class Message:
    def __init__(
        self,
        agent_id: str,
        message_type: MessageType,
        payload: Any,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        target: Optional[str] = None,
    ):
        self.message_id = message_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.message_type = message_type
        self.payload = payload
        self.topic = topic or "general"
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.target = target

    def to_json(self) -> str:
        return json.dumps({
            "message_id": self.message_id,
            "agent_id": self.agent_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "topic": self.topic,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "target": self.target,
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        return cls(
            agent_id=data["agent_id"],
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            topic=data.get("topic"),
            metadata=data.get("metadata"),
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            target=data.get("target"),
        )

    def to_bytes(self) -> bytes:
        return self.to_json().encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        return cls.from_json(data.decode('utf-8'))

    def __repr__(self) -> str:
        return f"Message(id={self.message_id[:8]}..., agent={self.agent_id}, type={self.message_type.value}, topic={self.topic})"

    def __str__(self) -> str:
        return f"[{self.agent_id}] {self.message_type.value}: {self.payload}"
