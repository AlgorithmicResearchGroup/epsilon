#!/usr/bin/env python3
"""CLI entrypoint for the standalone benchmark/reporting demo."""

from __future__ import annotations

import json

from examples.benchmark_report.workflow import parse_args, run_demo


if __name__ == "__main__":
    result = run_demo(parse_args())
    print(json.dumps(result, indent=2))
