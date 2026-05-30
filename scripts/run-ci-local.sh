#!/usr/bin/env bash
#
# Run the CI pipeline locally — mirrors .github/workflows/ci.yaml.
# Needed because the integration job loads real models and the GitHub
# [self-hosted, ci] runners are CPU-only. Run this on a GPU host with the
# model cache available.
#
# Usage:
#   HF_CACHE=/mnt/k8scache HF_HUB_OFFLINE=1 ./scripts/run-ci-local.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE=docker-compose.test.yaml
export HF_CACHE="${HF_CACHE:-/mnt/k8scache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "──────── 1/5  code quality ────────"
uv sync --project backend --extra dev
uv lock --project backend --check
uvx pre-commit run --all-files

echo "──────── 2/5  unit tests (mocked, no GPU) ────────"
uv run --project backend pytest backend/tests/test_api.py backend/tests/test_utils.py -v

echo "──────── 3/5  start ARBI-TR (docker compose) ────────"
docker compose -f "$COMPOSE_FILE" up --build -d
cleanup() { docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans; }
trap cleanup EXIT
echo "waiting for /health ..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health >/dev/null; then echo "✅ healthy"; break; fi
    [ "$i" = 60 ] && { echo "server did not become healthy"; docker compose -f "$COMPOSE_FILE" logs --tail=100; exit 1; }
    sleep 5
done

echo "──────── 4/5  generate + install client ────────"
bash scripts/generate-client.sh

echo "──────── 5/5  integration tests (through the generated client) ────────"
ARBI_TR_URL=http://localhost:8000 uv run --project backend pytest backend/tests/integration -v -m integration

echo "✅ local CI passed"
