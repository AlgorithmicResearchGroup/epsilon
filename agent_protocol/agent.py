import zmq
import threading
import time
import logging
from typing import Optional, Callable, Any, List, Dict
from .messages import Message, MessageType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        agent_id: str,
        broker_router: str = "tcp://localhost:5555",
        broker_sub: str = "tcp://localhost:5556",
        topics: Optional[List[str]] = None,
        message_handler: Optional[Callable[[Message], None]] = None,
        enable_logging: bool = True,
        broker_push: str = None,  # backward compat alias
        heartbeat_interval_seconds: float = 5.0,
        heartbeat_enabled: bool = True,
    ):
        self.agent_id = agent_id
        self.broker_router = broker_push or broker_router
        self.broker_sub = broker_sub
        self.topics = topics or []
        self.message_handler = message_handler or self._default_message_handler
        self.enable_logging = enable_logging
        self.heartbeat_interval_seconds = max(1.0, float(heartbeat_interval_seconds))
        self.heartbeat_enabled = bool(heartbeat_enabled)

        self.context = zmq.Context()
        self.dealer_socket = None
        self.sub_socket = None
        self.running = False
        self.thread = None
        self.heartbeat_thread = None
        self.registered = False
        self._send_lock = threading.Lock()

        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "heartbeats_sent": 0,
            "start_time": None,
        }

    def start(self) -> None:
        logger.info(f"Starting agent {self.agent_id}")

        # DEALER socket with identity
        self.dealer_socket = self.context.socket(zmq.DEALER)
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.agent_id.encode('utf-8'))
        self.dealer_socket.connect(self.broker_router)

        # SUB socket (unchanged)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(self.broker_sub)

        if not self.topics:
            self.sub_socket.subscribe(b"")
        else:
            for topic in self.topics:
                topic_filter = f"{topic}:".encode('utf-8')
                self.sub_socket.subscribe(topic_filter)
                logger.info(f"Agent {self.agent_id} subscribed to topic: {topic}")

        self.running = True
        self.stats["start_time"] = time.time()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        self._send_register()

        if self.heartbeat_enabled:
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

        logger.info(f"Agent {self.agent_id} started successfully")

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.dealer_socket, zmq.POLLIN)
        poller.register(self.sub_socket, zmq.POLLIN)

        while self.running:
            socks = dict(poller.poll(timeout=100))

            if self.dealer_socket in socks:
                self._receive_dealer_message()

            if self.sub_socket in socks:
                self._receive_sub_message()

    def _receive_dealer_message(self) -> None:
        """Receive from DEALER (control plane: ACKs, directed messages, task assignments)."""
        frames = self.dealer_socket.recv_multipart()
        # DEALER receives: [empty_delimiter, message_bytes]
        message_bytes = frames[-1]
        message = Message.from_bytes(message_bytes)

        if message.agent_id == self.agent_id:
            return

        self.stats["messages_received"] += 1

        if self.enable_logging:
            logger.info(f"Agent {self.agent_id} received (DEALER): {message}")

        # Handle registration ACK internally
        if message.message_type == MessageType.ACK and message.payload.get("status") == "registered":
            self.registered = True
            logger.info(f"Agent {self.agent_id} registration confirmed")
            return

        self.message_handler(message)

    def _receive_sub_message(self) -> None:
        """Receive from SUB (broadcast plane)."""
        result = self.sub_socket.recv_multipart()

        if len(result) == 2:
            topic_prefix, message_bytes = result
        else:
            message_bytes = result[0] if result else b""

        message = Message.from_bytes(message_bytes)

        if message.agent_id == self.agent_id:
            return

        self.stats["messages_received"] += 1

        if self.enable_logging:
            logger.info(f"Agent {self.agent_id} received (SUB): {message}")

        self.message_handler(message)

    def _default_message_handler(self, message: Message) -> None:
        print(f"[{self.agent_id}] Received: {message}")

    def _send_via_dealer(self, message: Message) -> None:
        """Send a message through the DEALER socket to the broker."""
        with self._send_lock:
            self.dealer_socket.send_multipart([b"", message.to_bytes()])
        self.stats["messages_sent"] += 1
        if self.enable_logging:
            logger.info(f"Agent {self.agent_id} sent: {message}")

    def send_message(
        self,
        payload: Any,
        message_type: MessageType = MessageType.DATA,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        target: Optional[str] = None,
    ) -> None:
        """Send a message. If target is set, broker routes directly. Otherwise broadcast via PUB."""
        message = Message(
            agent_id=self.agent_id,
            message_type=message_type,
            payload=payload,
            topic=topic,
            metadata=metadata,
            target=target,
        )
        self._send_via_dealer(message)

    def send_data(self, data: Any, topic: Optional[str] = None, target: Optional[str] = None) -> None:
        self.send_message(data, MessageType.DATA, topic, target=target)

    def send_directed(self, data: Any, target: str, topic: Optional[str] = None) -> None:
        """Send a message directly to a specific agent."""
        self.send_message(data, MessageType.DATA, topic, target=target)

    def send_control(self, command: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload = {"command": command}
        if params:
            payload.update(params)
        self.send_message(payload, MessageType.CONTROL, topic="control")

    def _send_register(self) -> None:
        """Send REGISTER message to broker."""
        self.send_message(
            payload={
                "agent_id": self.agent_id,
                "subscribed_topics": self.topics,
                "timestamp": time.time(),
            },
            message_type=MessageType.REGISTER,
            topic="discovery",
        )

    def send_heartbeat(self) -> None:
        self.send_message(
            payload={"status": "alive", "timestamp": time.time()},
            message_type=MessageType.HEARTBEAT,
            topic="heartbeat",
        )
        self.stats["heartbeats_sent"] += 1

    def _heartbeat_loop(self) -> None:
        while self.running:
            self.send_heartbeat()
            time.sleep(self.heartbeat_interval_seconds)

    # --- Work Queue ---

    def submit_task(self, task_payload: Any) -> None:
        """Submit a task to the broker's work queue."""
        self.send_message(task_payload, MessageType.TASK_SUBMIT, topic="tasks")

    def request_task(self) -> None:
        """Request a task from the broker's work queue. Response arrives via DEALER."""
        self.send_message(
            {"agent_id": self.agent_id},
            MessageType.TASK_REQUEST,
            topic="tasks",
        )

    def renew_task(self, task_id: str, lease_id: str) -> None:
        """Renew a task lease to prevent redelivery."""
        self.send_message(
            {"task_id": task_id, "lease_id": lease_id},
            MessageType.TASK_RENEW,
            topic="tasks",
        )

    def complete_task(self, task_id: str, result: Any = None, lease_id: Optional[str] = None) -> None:
        """Report task completion to the broker."""
        payload = {"task_id": task_id, "result": result}
        if lease_id:
            payload["lease_id"] = lease_id
        self.send_message(
            payload,
            MessageType.TASK_COMPLETE,
            topic="tasks",
        )

    def fail_task(self, task_id: str, error: Any = None, lease_id: Optional[str] = None) -> None:
        """Report terminal task failure to the broker."""
        payload = {"task_id": task_id, "error": error}
        if lease_id:
            payload["lease_id"] = lease_id
        self.send_message(
            payload,
            MessageType.TASK_FAIL,
            topic="tasks",
        )

    # --- Subscriptions ---

    def subscribe_topic(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics.append(topic)
            topic_filter = f"{topic}:".encode('utf-8')
            self.sub_socket.subscribe(topic_filter)
            logger.info(f"Agent {self.agent_id} subscribed to topic: {topic}")

    def unsubscribe_topic(self, topic: str) -> None:
        if topic in self.topics:
            self.topics.remove(topic)
            topic_filter = f"{topic}:".encode('utf-8')
            self.sub_socket.unsubscribe(topic_filter)
            logger.info(f"Agent {self.agent_id} unsubscribed from topic: {topic}")

    def stop(self) -> None:
        logger.info(f"Stopping agent {self.agent_id}...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=2)

        if self.dealer_socket:
            self.dealer_socket.close()
        if self.sub_socket:
            self.sub_socket.close()

        logger.info(f"Agent {self.agent_id} stopped")

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        return {
            "agent_id": self.agent_id,
            "messages_sent": self.stats["messages_sent"],
            "messages_received": self.stats["messages_received"],
            "heartbeats_sent": self.stats["heartbeats_sent"],
            "uptime_seconds": uptime,
            "subscribed_topics": self.topics,
            "registered": self.registered,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
