#!/usr/bin/env python3
"""CLI entrypoint for the benchmark scout example."""

from __future__ import annotations

import json

from examples.benchmark_scout.workflow import parse_args, run_demo


def main() -> None:
    result = run_demo(parse_args())
    print(json.dumps(result["final_summary"], indent=2))


if __name__ == "__main__":
    main()
