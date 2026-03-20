# Contributing

## Development Setup

Use the project virtualenv and pinned requirements:

```bash
make install
```

If you are not using `make`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install pytest-mock
```

## Running Tests

The repo currently treats the focused core suite as the default contributor path:

```bash
make test-core
```

This covers:

- core tools
- BYOA runtime paths
- broker reliability
- large-scale topology helpers
- benchmark/demo wrappers

Some optional or environment-specific tests may require extra local dependencies or external services. Keep the core suite green before sending a change.

## Coding Guidelines

- Keep changes scoped. Avoid broad refactors unless they are necessary for the task.
- Preserve the topology/runtime split. Do not fold demo-specific logic into shared orchestrator code unless it is genuinely reusable.
- Prefer deterministic reducers/finalizers for large-scale workflows when the aggregation step does not need open-ended model reasoning.
- Keep intermediate artifacts structured and machine-readable.
- Add tests for new example/demo behavior when practical.

## Pull Requests

Before opening a PR:

1. run `make test-core`
2. update docs if the user-facing workflow changed
3. include a concise description of the workload or topology affected
4. call out any new environment variables, artifacts, or operational assumptions
5. confirm that you agree to [CLA.md](CLA.md)

## Contributor License Agreement

This project uses a lightweight contributor license agreement.

By submitting a contribution, you agree that:

- you have the right to submit the work
- the contribution can be distributed as part of Epsilon
- the contribution will be licensed with the Project under Apache 2.0

See [CLA.md](CLA.md) for the full terms.

## Examples And Demos

Examples under `examples/` are treated as first-class launch surfaces. If you change one:

- keep the README for that example accurate
- preserve reproducible CLI flags where possible
- avoid adding product-specific UI assumptions to the example itself
