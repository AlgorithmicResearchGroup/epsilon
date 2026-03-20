# Launching Epsilon: A Multi-Agent Runtime With Real Numbers

Most AI developer products are easy to demo with one impressive prompt.

That is not the problem Epsilon is trying to solve.

Epsilon is built for the harder class of workloads:

- many agents working at once
- structured intermediate outputs
- multiple coordination patterns
- deterministic aggregation where it helps
- and a second wave of reasoning only when the first wave surfaces ambiguity

That kind of system needs a different standard of proof than "it feels good."

So we ran a benchmark that asked two practical questions:

1. Can Epsilon scale cleanly across different large-agent topologies?
2. Does the second wave of agents do useful curation work, or is it just extra runtime?

The short version is:

- yes, Epsilon scales well on large manifest-backed workloads
- yes, topology choice matters in measurable ways
- and yes, the second wave activates selectively and changes the final artifact

## What Epsilon Actually Is

At a high level, Epsilon is a runtime for LLM-powered agents that can operate in several coordination modes.

The key architectural idea is that **one topology is not enough**.

Some workloads are mostly flat and independent. Others need hierarchical aggregation. Others need a second pass over only the hard cases. Epsilon treats those as different execution shapes instead of pretending one agent loop should handle everything.

The main large-scale patterns in this benchmark were:

- `work_queue`
- `sharded_queue`
- `map_reduce`

### `work_queue`

This is the simplest large-scale pattern.

- a coordinator submits work units
- worker daemons pull tasks from a broker
- agents process independent items
- results are collected without extra hierarchy

This is the right shape for flat, independent workloads where the final output is mostly the union of leaf outputs.

### `sharded_queue`

This is the same basic pull-based execution model, but partitioned into shard-local coordination.

It is useful when:

- the item count is large
- bookkeeping becomes non-trivial
- and you want to keep coordination local rather than fully flat

### `map_reduce`

This is the topology for workloads with a real aggregation boundary.

- map agents process many independent items
- deterministic reducers combine those outputs in a structured way
- final reducers produce a higher-level artifact

This matters because many useful agent systems are not just "N agents in parallel." They are "N agents in parallel, followed by structured merge logic, followed by a final artifact."

### Broker, workers, and deterministic reducers

Under the hood, Epsilon uses:

- a broker for task delivery and result routing
- worker daemons that pull tasks and execute them
- agent tasks for reasoning-heavy leaves
- deterministic Python handlers and reducers for merge/finalization stages

That last point is important.

The architecture is not "LLMs all the way down." Epsilon deliberately mixes:

- LLM agents where judgment is needed
- deterministic reducers where consistency and cost matter more than raw flexibility

That blend is what makes the second-wave workflows practical.

## What We Benchmarked

We used two benchmark tracks.

### 1. Scale track

For scale, we used the deterministic `local_reduce` workload.

The point of this benchmark was to measure orchestration behavior, not model or network variance.

Configuration:

- topologies: `work_queue`, `sharded_queue`, `map_reduce`
- worker counts: `2`, `8`, `24`
- input items: `240`

That produced `9` scale runs total.

### 2. Semantic track

For semantic behavior, we used the Benchmark Scout demo on a seeded random slice of the local S2ORC computer science corpus.

Configuration:

- corpus: `/home/matt/gcs-downloads/s2orc_computer_science_7_14_parquet`
- sample size: `40` papers
- sample mode: `random`
- sample seed: `17`
- worker count: `8`

We ran it in two modes:

- `extraction_only`
- `two_pass`

The Benchmark Scout workflow is useful because it exercises a realistic multi-stage pattern:

1. agents read papers and extract benchmark result rows
2. reducers detect suspiciously similar rows
3. a second wave of agents decides whether those rows are actually comparable

That is the kind of problem where agentic coordination is genuinely better than a single one-pass extractor.

## Result 1: Epsilon Scales Cleanly

All `9` scale runs completed successfully.

- failures: `0`
- missing tasks: `0`

The fastest run was `work_queue` at `24` workers:

- `1850.90` queue tasks/min
- `p95 latency = 2ms`

Here is the scale curve:

![Throughput by workers](assets/benchmark-report/throughput_by_workers.svg)

### Throughput By Topology

| Topology | 2 workers | 8 workers | 24 workers | Scale-up |
| --- | ---: | ---: | ---: | ---: |
| `work_queue` | 194.81 | 723.25 | 1850.90 | 9.50x |
| `sharded_queue` | 187.90 | 571.49 | 1106.34 | 5.89x |
| `map_reduce` | 188.97 | 650.68 | 1433.16 | 7.58x |

### What This Tells Us About The Architecture

These numbers are not just "more workers = more throughput."

They reflect a real architectural distinction:

- `work_queue` wins when work is flat and independent
- `map_reduce` pays extra coordination cost, but still scales strongly when hierarchical reduction is part of the workload
- `sharded_queue` sits in the middle, where partitioned coordination helps but still carries some orchestration overhead

That is exactly what we want from a topology-driven runtime. Different workload shapes should produce different winners.

If one topology won everything, the topology model would be mostly cosmetic. These results suggest it is real.

## Result 2: The Second Wave Is Selective

The semantic benchmark was designed to answer a different question:

Does Epsilon do expensive reasoning everywhere, or only where ambiguity appears?

The answer from this run is that the second wave is selective.

Extraction-only run:

- `40` papers
- `86` extracted benchmark records
- `0` ambiguity cases
- `0` adjudications

Two-pass run:

- `40` papers
- `78` extracted benchmark records
- `14` ambiguity candidates
- `14` adjudication tasks
- `16` records touched by explicit judgments

That means:

- adjudication activated on `35%` of papers
- explicit judgment touched `20.5%` of extracted records

Here is the funnel:

![Selective reasoning funnel](assets/benchmark-report/selective_reasoning_funnel.svg)

### Why This Matters

This is the architectural pattern Epsilon is optimized for:

1. parallel extraction across the whole corpus
2. deterministic reducers to identify uncertainty
3. targeted second-pass reasoning on only the hard cases

That is much more interesting than "fan out 100 prompts."

It means the system can:

- use parallelism where the task is embarrassingly parallel
- use deterministic code where the merge logic should be stable
- and reserve expensive agent reasoning for the minority of items that actually need it

This is the core design principle behind the large-scale Epsilon examples.

## Result 3: Two-Pass Output Is Meaningfully Different

The most important question is not just whether the second wave ran. It is whether it changed the final artifact in a meaningful way.

In this run, the second pass:

- created `14` explicit comparison judgments
- touched `16` extracted records
- marked all `14` ambiguity cases as `not_comparable`

Here is the one-pass vs two-pass comparison:

![One-pass vs two-pass](assets/benchmark-report/one_pass_vs_two_pass.svg)

### Why Zero Comparable Groups Is Still A Good Result

This run produced `0` comparable groups.

That is not a failure.

It means the ambiguity detector found pairs that looked similar enough to justify review, and the adjudication agents then rejected them as invalid benchmark comparisons.

A few examples:

- MMLU subset `3-shot` vs full-suite MMLU `5-shot`
- HellaSwag zero-prompt ablation vs standard few-shot evaluation
- superficially similar percentage metrics across unrelated tasks

That is exactly the kind of thing a naive extraction pipeline tends to collapse into one benchmark table because the strings look close enough.

The second wave made those decisions explicit.

That is useful output.

## One Important Caveat

The extraction-only run produced `86` records, while the two-pass run produced `78`.

I would not use raw extracted-record count as the main headline metric.

These are separate live LLM runs, so some extraction variance is expected.

The stronger signal is:

- how many ambiguity cases were surfaced
- how many explicit judgments were produced
- how much of the final dataset was touched by curation

In this run, the answer was:

- `14` adjudications
- `16` judged records
- `20.5%` of extracted records touched by explicit second-pass decisions

## Why This Matters For Epsilon

The point of this benchmark is not that Epsilon is the fastest possible queue runner or that one particular semantic demo is "state of the art."

The point is that Epsilon now has a concrete, defensible product story:

- it can run large-agent workloads with thousands of task executions over time
- it can switch topology based on workload shape
- it can combine LLM leaves with deterministic reducers
- and it can run a second wave of targeted reasoning when the first wave surfaces ambiguity

That is a much more useful story than "multi-agent AI, but bigger."

It is a systems story:

- runtime
- topology
- structure
- selective escalation

## What We Have Now

This benchmark gives Epsilon a first real launch-quality proof point:

- a scale curve
- a selective-reasoning funnel
- a one-pass vs two-pass comparison
- and a benchmark bundle that can be rerun

It is not a NeurIPS paper.

It is something more useful for a product at this stage:

- reproducible
- visual
- technical
- honest
- and tied directly to the architecture

## Artifacts

The benchmark bundle for this run was written to:

- `/tmp/epsilon-benchmark-report-live1/benchmark_bundle.json`
- `/tmp/epsilon-benchmark-report-live1/scale_metrics.csv`
- `/tmp/epsilon-benchmark-report-live1/semantic_metrics.csv`
- `/tmp/epsilon-benchmark-report-live1/report.md`

The charts embedded above were copied into this repo from that run.
