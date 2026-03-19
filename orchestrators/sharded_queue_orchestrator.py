#!/usr/bin/env python3
"""Manifest-backed sharded queue orchestrator."""

from __future__ import annotations

from orchestrators.scale_queue_orchestrator import main_for_pattern


if __name__ == "__main__":
    main_for_pattern("sharded_queue")
