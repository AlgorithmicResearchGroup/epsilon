# Examples

These are the main example workloads shipped with Epsilon.

## Recommended Starting Points

### 1. Benchmark Report

Best if you want a numbers-driven overview of the system.

It runs:

- deterministic topology scaling runs
- Benchmark Scout in extraction-only mode
- Benchmark Scout in two-pass adjudication mode

Outputs:

- CSV metrics
- SVG charts
- a short Markdown report

See [benchmark_report/README.md](benchmark_report/README.md).

### 2. Benchmark Scout

Best if you want to see a real two-pass agent workflow over a large paper corpus.

It:

- extracts benchmark rows from papers
- detects questionable comparisons
- escalates only those cases to a second wave of agents

See [benchmark_scout/README.md](benchmark_scout/README.md).

### 3. HF Entity Graph

Best if you want a document-to-graph example.

It:

- summarizes document clusters
- extracts entities and relations
- detects ambiguity across documents
- adjudicates only the ambiguous entities

See [hf_entity_graph/README.md](hf_entity_graph/README.md).

## Adapter / BYOA Examples

If you want to plug your own agent implementation into Epsilon without changing orchestrators, start here:

- [byoa/simple_run_agent.py](byoa/simple_run_agent.py)
- [byoa/while_loop_agent.py](byoa/while_loop_agent.py)
- [byoa/modular_public_adapter.py](byoa/modular_public_adapter.py)

The larger adapter example is here:

- [modular-public/README.md](modular-public/README.md)
