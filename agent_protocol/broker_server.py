#!/usr/bin/env python3
"""Container-friendly broker entrypoint."""

import os
import signal
import threading
import time

from .broker import MessageBroker


def main() -> None:
    router_port = int(os.environ.get("BROKER_ROUTER_PORT", "5555"))
    pub_port = int(os.environ.get("BROKER_PUB_PORT", "5556"))
    bind_address = os.environ.get("BROKER_BIND_ADDRESS", "*")
    heartbeat_timeout = float(os.environ.get("BROKER_HEARTBEAT_TIMEOUT_SECONDS", "30"))
    lease_timeout = float(os.environ.get("BROKER_LEASE_TIMEOUT_SECONDS", "60"))
    sweep_interval = float(os.environ.get("BROKER_SWEEP_INTERVAL_SECONDS", "1"))
    max_redeliveries = int(os.environ.get("BROKER_MAX_REDELIVERIES", "5"))
    max_fail_retries = int(os.environ.get("BROKER_MAX_FAIL_RETRIES", "0"))
    backoff_base = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_BASE_SECONDS", "0"))
    backoff_max = float(os.environ.get("BROKER_REDELIVERY_BACKOFF_MAX_SECONDS", "30"))

    broker = MessageBroker(
        router_port=router_port,
        pub_port=pub_port,
        bind_address=bind_address,
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

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        broker.stop()


if __name__ == "__main__":
    main()
