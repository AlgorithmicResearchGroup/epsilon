#!/usr/bin/env python3
"""CLI entrypoint for the population-search CSV optimization demo."""

from __future__ import annotations

import json

from examples.population_search_csv.workflow import parse_args, run_demo


def main() -> None:
    result = run_demo(parse_args())
    print(json.dumps(result["final_summary"], indent=2))


if __name__ == "__main__":
    main()
