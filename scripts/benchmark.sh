#!/usr/bin/env bash
# Benchmark ARBI-TR transcription + diarization speed.
#
# Usage:
#   ./scripts/benchmark.sh                          # uses bundled 1-min clip
#   ./scripts/benchmark.sh /path/to/audio.wav       # custom file
#   ARBI_TR_URL=http://host:8000 ./scripts/benchmark.sh
#
set -euo pipefail

BASE_URL="${ARBI_TR_URL:-http://localhost:8000}"
AUDIO_FILE="${1:-backend/tests/fixtures/agi_1min.ogg}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve relative paths from repo root
if [[ ! "$AUDIO_FILE" = /* ]]; then
    AUDIO_FILE="$SCRIPT_DIR/$AUDIO_FILE"
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
    echo "Error: $AUDIO_FILE not found"
    exit 1
fi

# Get audio duration via ffprobe
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$AUDIO_FILE" 2>/dev/null | cut -d. -f1)
if [[ -z "$DURATION" ]]; then
    echo "Warning: could not determine audio duration"
    DURATION="?"
fi

FILE_SIZE=$(du -h "$AUDIO_FILE" | cut -f1)
echo "=== ARBI-TR Benchmark ==="
echo "Server:   $BASE_URL"
echo "File:     $(basename "$AUDIO_FILE") ($FILE_SIZE, ${DURATION}s)"
echo ""

# Health check
if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
    echo "Error: server not reachable at $BASE_URL"
    exit 1
fi

# --- Transcribe + Diarize (async /transcribe/) ---
echo "--- Transcribe + Diarize ---"
T0=$(date +%s%N)
RESP=$(curl -sf -X POST "$BASE_URL/transcribe/" \
    -F "file=@$AUDIO_FILE" \
    -F "task_str=transcribe" \
    -F "size_of_model=large" \
    -F "speaker_number=2")
SID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

while true; do
    STATUS=$(curl -sf "$BASE_URL/task_status/$SID" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
        break
    fi
    sleep 0.5
done
T1=$(date +%s%N)
ELAPSED_MS=$(( (T1 - T0) / 1000000 ))
ELAPSED_S=$(python3 -c "print(f'{$ELAPSED_MS/1000:.1f}')")

RESULT=$(curl -sf "$BASE_URL/task_status/$SID")
N_SEGS=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('segments',[])))")
N_SPEAKERS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len({s['Speaker'] for s in d.get('segments',[])}))")

if [[ "$DURATION" != "?" ]]; then
    RTX=$(python3 -c "print(f'{$DURATION / ($ELAPSED_MS/1000):.0f}')")
    echo "  Time:     ${ELAPSED_S}s (${RTX}x realtime)"
else
    echo "  Time:     ${ELAPSED_S}s"
fi
echo "  Segments: $N_SEGS"
echo "  Speakers: $N_SPEAKERS"
echo "  Status:   $STATUS"
echo ""

# --- Transcribe only (sync /v1/audio/transcriptions) ---
echo "--- Transcribe Only (OpenAI endpoint) ---"
T0=$(date +%s%N)
RESP=$(curl -sf -X POST "$BASE_URL/v1/audio/transcriptions" \
    -F "file=@$AUDIO_FILE" \
    -F "model=whisper-large-v3" \
    -F "language=en")
T1=$(date +%s%N)
ELAPSED_MS=$(( (T1 - T0) / 1000000 ))
ELAPSED_S=$(python3 -c "print(f'{$ELAPSED_MS/1000:.1f}')")
TEXT_LEN=$(echo "$RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('text','')))")

if [[ "$DURATION" != "?" ]]; then
    RTX=$(python3 -c "print(f'{$DURATION / ($ELAPSED_MS/1000):.0f}')")
    echo "  Time:     ${ELAPSED_S}s (${RTX}x realtime)"
else
    echo "  Time:     ${ELAPSED_S}s"
fi
echo "  Text len: $TEXT_LEN chars"
echo ""

# --- Server info ---
echo "--- Server ---"
curl -sf "$BASE_URL/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Queue: {d[\"queue_length\"]}')"

# GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "  GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while read line; do
        echo "    $line"
    done
fi
