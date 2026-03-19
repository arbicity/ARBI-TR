#!/usr/bin/env bash
# Run ARBI-TR integration tests against a freshly built server.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/run-integration-tests.sh
#   HF_TOKEN=hf_xxx TEST_AUDIO_FILE=/path/to/speech.wav ./scripts/run-integration-tests.sh
#
# Options via env vars:
#   HF_TOKEN           Required. HuggingFace token for pyannote gated models.
#   TEST_AUDIO_FILE    Optional. Path to a real speech WAV for content assertions.
#   ARBI_TR_URL        Override server URL (default: http://localhost:8000)
#   KEEP_RUNNING       Set to 1 to leave the server running after tests.
#   NO_REBUILD         Set to 1 to skip docker compose build (use existing image).

set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.test.yaml"
BASE_URL="${ARBI_TR_URL:-http://localhost:8000}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"
NO_REBUILD="${NO_REBUILD:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required for pyannote gated models."
    echo "  export HF_TOKEN=hf_xxx"
    exit 1
fi

cleanup() {
    if [[ "$KEEP_RUNNING" != "1" ]]; then
        echo ""
        echo "==> Stopping server..."
        docker compose -f "$COMPOSE_FILE" down --remove-orphans
    else
        echo "==> Server left running (KEEP_RUNNING=1). Stop with:"
        echo "    docker compose -f $COMPOSE_FILE down"
    fi
}
trap cleanup EXIT

# Build + start
echo "==> Building and starting ARBI-TR..."
if [[ "$NO_REBUILD" == "1" ]]; then
    docker compose -f "$COMPOSE_FILE" up -d
else
    docker compose -f "$COMPOSE_FILE" up --build -d
fi

# Wait for /health
echo "==> Waiting for server to be ready (model loading may take several minutes)..."
MAX_WAIT=300
ELAPSED=0
until curl -sf "$BASE_URL/health" > /dev/null 2>&1; do
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
        echo "Docker logs:"
        docker compose -f "$COMPOSE_FILE" logs --tail=50
        exit 1
    fi
    printf "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo ""
echo "==> Server is ready at $BASE_URL"

# Run integration tests from the backend directory
echo "==> Running integration tests..."
cd "$(dirname "$0")/../backend"

export ARBI_TR_URL="$BASE_URL"

uv run pytest tests/integration/ -v \
    --tb=short \
    -p no:timeout \
    "$@"
