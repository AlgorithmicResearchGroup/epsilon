# Benchmark Scout

## What This Example Is

This example turns a local research-paper corpus into a structured benchmark dataset.

It is built around a local S2ORC computer science corpus and shows a different kind of agentic workflow than the graph example:

- one agent reads each paper
- extracts benchmark-style result records
- reducers detect comparisons that may be misleading or non-equivalent
- a second wave of agents decides whether those records are actually comparable

The output is a machine-readable benchmark dataset, not a graph database.

## Why This Is Useful

A traditional pipeline can extract rows like:

- model
- dataset
- metric
- score

But it often cannot answer the harder question:

- should these two rows actually be compared?

That requires judgment about setup, split, evaluation mode, and scope.

This example is meant to show that Epsilon can do both:

1. extract benchmark records in parallel
2. escalate only the questionable comparisons to a second-pass adjudication workflow

## Workflow

### 1. Slice the local corpus

The runner scans the local S2ORC parquet corpus and selects papers that look like LLM/evaluation papers using deterministic keyword and metadata filters.

The default profile is `text_llm_eval`, which is tuned for text-centric LLM benchmark papers rather than broad multimodal evaluation papers.

### 2. Extract benchmark records

The first phase uses `map_reduce`.

Each paper agent produces strict JSON containing:

- paper summary
- benchmark task
- dataset
- metric
- model/system name
- score text and score value
- evaluation mode
- split
- setup notes
- evidence snippets

### 3. Detect questionable comparisons

Deterministic reducers merge all paper outputs and look for records that appear to belong to the same benchmark bucket but may not actually be comparable.

### 4. Adjudicate comparability

The second phase uses `sharded_queue`.

Each adjudication agent decides whether two benchmark records are:

- `comparable`
- `not_comparable`
- `uncertain`

### 5. Write the final dataset

The final artifacts include:

- `benchmark_results.json`
- `benchmark_results.csv`
- `comparison_judgments.json`
- `paper_index.json`
- `run_report.md`

## How To Run It

```bash
OPENAI_API_KEY=... \
PYTHONPATH=. python examples/benchmark_scout/run_demo.py \
  --corpus-root /home/matt/gcs-downloads/s2orc_computer_science_7_14_parquet \
  --sample-size 40 \
  --sample-mode random \
  --sample-seed 17 \
  --worker-count 8
```

Useful options:

- `--sample-size`: number of papers to process
- `--sample-mode first|random`: deterministic prefix sample or seeded random sample
- `--sample-seed`: seed for reproducible sampling
- `--keyword-profile`: corpus slicing profile, defaulting to `text_llm_eval`
- `--min-year`: lower year bound for candidate papers
- `--candidate-pool-size`: number of slice candidates to collect before sampling
- `--max-scan-files`: upper bound on scanned parquet shards

## Why It Is More Than A Pipeline

The key value of this example is not just “extract benchmark rows from papers.”

The value is that it:

1. extracts results from many papers in parallel
2. detects where comparison is ambiguous
3. spends additional reasoning only on those ambiguous cases
4. emits a cleaner benchmark dataset with explicit comparison judgments

That is the difference between a one-pass extractor and an agentic curation workflow.
