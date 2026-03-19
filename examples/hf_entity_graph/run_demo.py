#!/usr/bin/env python3
"""CLI entrypoint for the HF entity graph demo."""

from __future__ import annotations

import json

from examples.hf_entity_graph.workflow import parse_args, run_demo


def main() -> None:
    result = run_demo(parse_args())
    print(json.dumps(result["final_summary"], indent=2))


if __name__ == "__main__":
    main()
