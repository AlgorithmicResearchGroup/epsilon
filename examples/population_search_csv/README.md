# Population Search CSV Demo

This example shows the `population_search` topology on a public, self-contained task:

- many agents optimize the same Python CSV aggregation function in parallel
- Epsilon evaluates every candidate deterministically
- a generation brief summarizes what worked
- the next generation uses the best solutions as context
- the loop repeats until improvement stalls

Run it with:

```bash
OPENAI_API_KEY=... \
PYTHONPATH=. python examples/population_search_csv/run_demo.py \
  --population-size 8 \
  --max-generations 4 \
  --worker-count 4
```

Main artifacts:

- `run_summary.json`
- `demo_summary.json`
- `population_search_csv/leaderboard.json`
- `population_search_csv/leaderboard.csv`
- `population_search_csv/generation_history.json`
- `population_search_csv/best_solution.py`
