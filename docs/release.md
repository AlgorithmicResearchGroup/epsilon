# Release And Packaging

This repo ships with a local Docker build path and a GitHub Actions workflow for publishing images to GHCR.

## Local Build

Build the image:

```bash
make docker-build
```

That produces:

- `epsilon:local`

The default container entrypoint is:

```bash
python orchestrate.py --pattern dag
```

So you can pass a task directly:

```bash
docker run --rm -it \
  -e OPENAI_API_KEY \
  epsilon:local \
  "Build a notes API with SQLite"
```

## GHCR Publish Workflow

Workflow:

- [.github/workflows/publish.yaml](../.github/workflows/publish.yaml)

It publishes on:

- pushes to `main`
- pushes to `master`
- tags matching `v*`
- manual dispatch

The workflow:

- resolves the image name automatically
- lowercases the GHCR path
- emits branch, tag, SHA, and semver tags
- optionally emits `latest`
- uses Buildx and GitHub Actions layer cache

## Local GHCR Publish Script

Script:

- [scripts/publish_ghcr.sh](../scripts/publish_ghcr.sh)

Example:

```bash
GHCR_TOKEN=... GHCR_IMAGE=ghcr.io/<owner>/<repo> GHCR_TAG=v0.1.0 \
  scripts/publish_ghcr.sh
```

## Suggested OSS Release Checklist

1. Run `make test-core`
2. Update `CHANGELOG.md`
3. Verify the main README quickstart commands still work
4. Build the Docker image locally with `make docker-build`
5. Tag the release, for example `v0.1.0`
6. Push the tag and confirm the GHCR publish workflow succeeds
7. Attach benchmark/demo write-ups if they are part of the release story

## License

This repo is licensed under Apache 2.0. See [../LICENSE](../LICENSE).
