# HF Entity Graph Demo

## What This Demo Is

This example runs a real multi-agent workflow on a public Hugging Face dataset:

- Dataset: `Awesome075/multi_news_parquet`
- Input: clusters of related news articles
- Output:
  - a short summary per document cluster
  - extracted entities and relations
  - a final canonical entity graph across the sampled corpus

The demo is designed to show a useful pattern for AI agents that goes beyond "run the same prompt 100 times in parallel."

## What Problem It Solves

Imagine you have a large news corpus and you want to answer questions like:

- Who are the main people, organizations, and places in this sample?
- Which entities are related to each other?
- When two documents say `Obama`, `Barack Obama`, and `President Obama`, are those the same entity?
- Which apparent duplicates should be merged, and which should stay separate?

That is a common data-processing problem in analytics, search, monitoring, and knowledge-graph construction. The hard part is usually not just extracting rows of data. The hard part is handling ambiguity and deciding when follow-up judgment is needed.

## What The Workflow Does

The workflow has four steps.

### 1. Sample and prepare documents

The runner downloads a parquet file from Hugging Face, selects a sample, and writes normalized JSON inputs locally.

Each input contains:

- `doc_id`
- `source_text`
- `reference_summary`

### 2. Fan out extraction work with agents

The system uses the `map_reduce` topology for the first phase.

One agent is assigned to each document cluster. Each agent reads the input and produces strict JSON containing:

- a short summary
- keywords
- typed entities such as `PERSON`, `ORG`, and `GPE`
- candidate relations between those entities
- short evidence snippets
- confidence scores

This is parallel work, but it is still structured. Agents are not just generating free-form prose. They are producing machine-readable intermediate outputs.

### 3. Detect ambiguity from the outputs

After the extraction phase, deterministic reducers merge all map outputs and look for likely duplicate or ambiguous entity clusters.

Examples:

- alias overlap such as `Barack Obama` and `Obama`
- similar names that may refer to the same person or organization
- conflicting type signals across documents

This is the key transition point: the system inspects the outputs of the first phase and decides whether more work is needed.

### 4. Spawn targeted adjudication agents

If ambiguity candidates exist, the system runs a second phase using `sharded_queue`.

Each adjudication agent gets one ambiguity case and decides:

- merge or keep separate
- canonical name
- entity type
- aliases worth preserving
- supporting evidence

The finalizer then applies those decisions and writes:

- `entity_graph.json`
- `entity_index.json`
- `document_summaries.jsonl`
- `run_report.md`

## Why This Is Useful

This pattern is useful when your workflow has both:

- high-volume independent work
- and a smaller amount of expensive judgment work that only becomes visible after the first pass

Examples:

- news and document intelligence
- entity resolution
- large-scale research summarization
- alert triage
- compliance and policy review
- log or incident analysis

In these settings, it is wasteful to do high-touch reasoning on every item up front. It is usually better to:

1. process everything cheaply and in parallel
2. detect where uncertainty or conflict exists
3. spend extra agent effort only on those cases

## Why This Is Better Than A Traditional Data Pipeline

A traditional pipeline can absolutely do part of this job. For example:

- parse documents
- run an extractor
- write records to a database
- deduplicate with heuristics

That works well when the rules are stable and ambiguity is simple.

This demo is better in cases where the hard part is not just transformation, but judgment.

### A traditional pipeline is good at:

- deterministic transforms
- repeatable batch processing
- simple aggregations
- rule-based validation

### This agentic workflow is better at:

- reading messy natural language
- producing structured summaries from long text
- handling open-ended entity mentions and aliases
- escalating only the ambiguous cases
- using evidence to make case-by-case merge decisions

The important difference is that this system is not only a fan-out pipeline. It is an output-driven workflow.

The second phase does not exist until the first phase creates evidence that more reasoning is required.

That makes it closer to how a human analyst team would work:

- junior analysts do the first pass
- a reviewer looks for conflicts
- specialists resolve only the tricky cases

## Why The Topologies Matter

This demo uses two different topologies because they fit two different kinds of work.

### `map_reduce`

Used for the first phase because:

- each document cluster can be processed independently
- results need to be merged into a global view
- hierarchical reduction scales better than one giant serial merge

### `sharded_queue`

Used for the second phase because:

- ambiguity cases are flat, independent follow-up tasks
- each case is small but judgment-heavy
- shard-local reducers make it easy to scale to many follow-up decisions

This is the broader lesson: large agent systems should not use one topology for everything. Different stages of the workflow benefit from different coordination patterns.

## What Makes This "Agentic"

Many so-called agent demos are really just prompt templates wrapped in a loop.

This demo is more meaningfully agentic because it includes:

- autonomous task execution by many agents
- structured intermediate artifacts
- topology changes across phases
- output-driven follow-up work
- targeted escalation instead of uniform processing
- recovery via retries when leaf tasks fail

The agents are not just generating text. They are participating in a larger workflow that decides what to do next based on what was discovered.

## Operational Notes

In practice, this demo also shows a few engineering points that matter in real systems:

- agent outputs need validation
- malformed leaf outputs must not collapse the whole run
- retries should be localized to failed tasks
- deterministic reducers are often the right way to aggregate agent outputs

That is why this example combines LLM agents with non-LLM reducers and finalizers.

## How To Run It

```bash
OPENAI_API_KEY=... \
PYTHONPATH=. python examples/hf_entity_graph/run_demo.py \
  --sample-size 100 \
  --sample-mode random \
  --sample-seed 17 \
  --worker-count 8
```

Useful options:

- `--sample-size`: number of document clusters to process
- `--sample-mode first|random`: deterministic prefix sample or seeded random sample
- `--sample-seed`: random seed for reproducible sampling
- `--worker-count`: number of queue workers to run

## How To Read The Output

- `final/document_summaries.jsonl`: one summary per sampled document cluster
- `final/entity_graph.json`: canonical entities plus relations
- `final/entity_index.json`: lightweight lookup table for entities
- `final/run_report.md`: high-level run summary
- `phase1_workspace/run_summary.json`: extraction and reduction details
- `phase2_workspace/run_summary.json`: adjudication details

## Main Takeaway

The value of this demo is not just parallelism.

The value is that the system can:

1. process a large corpus in parallel
2. discover where the data is ambiguous
3. launch a second wave of focused reasoning only where it is justified
4. fold those decisions back into a cleaner final result

That is the kind of workload where agentic orchestration can be more useful than a fixed one-pass data pipeline.
