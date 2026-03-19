# Benchmark Report Demo

## What This Demo Is

This is a standalone reporting wrapper around existing Epsilon demos.

It does not change the agent runtime or orchestrators. Instead, it:

- runs the deterministic scale benchmark as a black-box workload
- runs the Benchmark Scout example in extraction-only and two-pass modes
- collects their summaries
- writes a single benchmark bundle with charts and a short Markdown report

## What It Measures

The demo produces two kinds of numbers.

### 1. Scale numbers

Using the deterministic `local_reduce` workload, it measures:

- throughput in queue tasks per minute
- p50 and p95 latency
- failure and missing-task counts
- how throughput changes with worker count
- how `work_queue`, `sharded_queue`, and `map_reduce` compare

### 2. Agentic-value numbers

Using Benchmark Scout, it compares:

- extraction-only output
- full two-pass output with ambiguity detection and adjudication

It reports:

- papers processed
- benchmark records extracted
- ambiguity candidates generated
- adjudication tasks completed
- judged records
- comparable groups created

## The Three Charts

The bundle includes:

1. `throughput_by_workers.svg`
2. `selective_reasoning_funnel.svg`
3. `one_pass_vs_two_pass.svg`

Together, they tell a simple story:

- can Epsilon scale operationally?
- does the second wave activate selectively rather than everywhere?
- does the two-pass workflow materially enrich the output?

## How To Run It

```bash
OPENAI_API_KEY=... \
PYTHONPATH=. python examples/benchmark_report/run_demo.py \
  --corpus-root /home/matt/gcs-downloads/s2orc_computer_science_7_14_parquet \
  --semantic-sample-size 40 \
  --semantic-sample-seed 17 \
  --semantic-worker-count 8 \
  --scale-worker-counts 2,8,24 \
  --scale-task-count 480
```

Useful options:

- `--skip-scale`: only run the Benchmark Scout comparison
- `--skip-semantic`: only run the scale benchmark
- `--scale-topologies`: comma-separated topology list
- `--semantic-max-ambiguities`: cap second-pass adjudication volume
- `--agent-model`: override the model used by Benchmark Scout

## Outputs

The output directory contains:

- `benchmark_bundle.json`
- `scale_metrics.csv`
- `semantic_metrics.csv`
- `throughput_by_workers.svg`
- `selective_reasoning_funnel.svg`
- `one_pass_vs_two_pass.svg`
- `report.md`

It also keeps the raw child-run outputs under:

- `scale_runs/`
- `semantic_runs/`

## Why This Is Useful

This demo gives you a lightweight, reproducible way to say more than “the system seems good.”

It gives you:

- a scale curve
- a selective-reasoning funnel
- a concrete before/after comparison for one-pass vs two-pass curation

That is enough to communicate value to technical users without building a full research evaluation stack.
