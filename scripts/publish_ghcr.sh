#!/usr/bin/env bash
set -euo pipefail

# Publish this repo's Docker image to GHCR.
#
# Usage:
#   GHCR_TOKEN=... scripts/publish_ghcr.sh
#   GHCR_TOKEN=... GHCR_IMAGE=ghcr.io/org/repo GHCR_TAG=v1.2.3 scripts/publish_ghcr.sh
#   GHCR_TOKEN=... GHCR_IMAGE=ghcr.io/org/repo GHCR_TAGS=latest,sha-abc123 scripts/publish_ghcr.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required." >&2
  exit 1
fi

REMOTE_URL="$(git config --get remote.origin.url || true)"
if [[ -z "${REMOTE_URL}" ]]; then
  echo "Could not determine git remote origin URL." >&2
  exit 1
fi

OWNER_REPO="$(echo "${REMOTE_URL}" | sed -E 's#^git@github.com:##; s#^https://github.com/##; s#\.git$##')"
OWNER="$(echo "${OWNER_REPO}" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')"
REPO="$(echo "${OWNER_REPO}" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')"

GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/${OWNER}/${REPO}}"
GHCR_TAG="${GHCR_TAG:-latest}"
GHCR_TAGS="${GHCR_TAGS:-}"
GHCR_USERNAME="${GHCR_USERNAME:-${GITHUB_ACTOR:-${OWNER}}}"
GHCR_TOKEN="${GHCR_TOKEN:-${CR_PAT:-}}"

if [[ -z "${GHCR_TOKEN}" ]]; then
  cat >&2 <<'EOF'
Missing GHCR_TOKEN (or CR_PAT).
Create a GitHub token with package write permissions and export:
  export GHCR_TOKEN=...
EOF
  exit 1
fi

echo "${GHCR_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin

TAGS=()
if [[ -n "${GHCR_TAGS}" ]]; then
  IFS=',' read -r -a RAW_TAGS <<<"${GHCR_TAGS}"
  for t in "${RAW_TAGS[@]}"; do
    trimmed="$(echo "${t}" | xargs)"
    [[ -n "${trimmed}" ]] && TAGS+=("${trimmed}")
  done
else
  TAGS+=("${GHCR_TAG}")
fi

if [[ "${#TAGS[@]}" -eq 0 ]]; then
  echo "No tags resolved." >&2
  exit 1
fi

BUILD_ARGS=()
for tag in "${TAGS[@]}"; do
  BUILD_ARGS+=("-t" "${GHCR_IMAGE}:${tag}")
done

echo "Building image ${GHCR_IMAGE} with tags: ${TAGS[*]}"
docker build "${BUILD_ARGS[@]}" .

for tag in "${TAGS[@]}"; do
  echo "Pushing ${GHCR_IMAGE}:${tag}"
  docker push "${GHCR_IMAGE}:${tag}"
done

echo "Published: ${GHCR_IMAGE} (${TAGS[*]})"
