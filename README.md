# ARBI TR

Self-hosted, open-source audio transcription server with speaker diarization. Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote-audio](https://github.com/pyannote/pyannote-audio).

~75x realtime with speaker diarization, ~140x transcription-only on a single NVIDIA RTX 4090. A 2.5-hour podcast transcribes and diarizes in under 2 minutes.

## What You Get

**A transcription API server** that runs on your hardware. Send it an audio or video file, get back timestamped text with speaker labels. No data leaves your network.

**Two ways to use it:**

1. **API** — POST files to the server from any language or tool (curl, Python, JS, etc.)
2. **Web UI** (optional) — A Streamlit frontend for uploading files, pasting YouTube URLs, and viewing transcripts in your browser

## Quick Start

You need Docker, an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and a [HuggingFace token](https://huggingface.co/settings/tokens) (free — needed to download pyannote diarization models).

```bash
git clone https://github.com/arbitrationcity/ARBI-TR.git
cd ARBI-TR

# Set your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# Start the server + frontend
docker compose up
```

That's it. The server is at http://localhost:8000 and the web UI is at http://localhost:8501.

Models download automatically on first start and are cached on the host (at `~/.cache/huggingface` by default — override with `HF_CACHE` in `.env`). Subsequent starts are fast.

## API

Interactive docs are at http://localhost:8000/docs once the server is running.

### Transcribe with speaker diarization (async)

Submit a file, get a session ID, poll for results. This is the full pipeline — speech recognition + speaker identification.

```bash
# Submit
curl -X POST http://localhost:8000/transcribe/ \
  -F "file=@meeting.wav" \
  -F "task_str=transcribe" \
  -F "size_of_model=large"

# Response:
# {"session_id": "abc-123", "message": "Your request is queued for processing", "queue_position": 1}

# Poll for results
curl http://localhost:8000/task_status/abc-123

# Response (when complete):
# {
#   "status": "completed",
#   "segments": [
#     {"Start": "0:00:00", "End": "0:00:05", "Speaker": "SPEAKER_00", "Text": "Good morning everyone"},
#     {"Start": "0:00:05", "End": "0:00:09", "Speaker": "SPEAKER_01", "Text": "Hi, thanks for joining"},
#     ...
#   ]
# }
```

Optional parameters:
- `source_language` — ISO language code (e.g. `en`, `fr`). Omit for auto-detection.
- `speaker_number` — Expected number of speakers (1-8). Omit or `0` for auto-detection.

### OpenAI-compatible endpoints (sync)

Drop-in replacements for the OpenAI audio API. These are synchronous (response blocks until done) and return plain text without speaker labels.

```bash
# Transcribe
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=whisper-large-v3"

# {"text": "Good morning everyone. Hi, thanks for joining..."}

# Translate (any language to English)
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "file=@reunion.wav" \
  -F "model=whisper-large-v3"

# {"text": "Good morning everyone..."}
```

These endpoints accept the same parameters as the [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio): `model`, `language`, `prompt`, `response_format`, `temperature`.

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "queue_length": 0}
```

### All endpoints

| Method | Endpoint | Mode | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check with queue length |
| `POST` | `/transcribe/` | async | Transcribe + diarize. Returns session ID to poll. |
| `GET` | `/task_status/{session_id}` | — | Poll job status (`queued`, `completed`, `failed`) |
| `POST` | `/v1/audio/transcriptions` | sync | OpenAI-compatible transcription (no diarization) |
| `POST` | `/v1/audio/translations` | sync | OpenAI-compatible translation to English |

Supported input formats: WAV, MP3, MP4, M4A, FLAC, and anything else FFmpeg can decode.

## Web UI (Optional)

The Streamlit frontend runs alongside the API server. Open http://localhost:8501 in your browser.

**Features:**
- Upload audio/video files (WAV, MP3, MP4, M4A) for transcription
- Paste YouTube URLs to download and transcribe videos directly
- Choose between transcription and translation (any language to English)
- Configure model size (small for speed, large for accuracy) and speaker count
- View results as a table with start time, end time, speaker, and text

If you only need the API server without the web UI, use the test compose file:

```bash
docker compose -f docker-compose.test.yaml up
```

## Configuration

Set these in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for pyannote models | **(required)** |
| `HF_CACHE` | Host path for model cache (bind-mounted into containers) | `~/.cache/huggingface` |
| `WHISPER_MODEL_SIZE` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` | `large-v3` |
| `WHISPER_BEAM_SIZE` | Beam search width | `5` |
| `WHISPER_BATCH_SIZE` | Batched inference chunk count | `24` |
| `PYANNOTE_MODEL` | Pyannote pipeline model | `pyannote/speaker-diarization-community-1` |
| `PYANNOTE_SEG_BATCH` | Pyannote segmentation batch size | `32` |
| `PYANNOTE_EMB_BATCH` | Pyannote embedding batch size | `32` |
| `ENABLE_TF32` | Enable TF32 matmul on Ampere+ GPUs | `1` |
| `WHISPER_DEVICE` | GPU for whisper (e.g. `cuda:0`, `cpu`) | auto |
| `DIARIZE_DEVICE` | GPU for pyannote (e.g. `cuda:1`, `cpu`) | auto |

### Docker Compose variants

| File | What it runs |
|------|-------------|
| `docker-compose.yaml` | API server + web UI (default) |
| `docker-compose.test.yaml` | API server only (for testing / API-only deployments) |

## Development

### Running without Docker

```bash
# Backend
cd backend
uv sync
HF_TOKEN=hf_xxx uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
uv sync
API_ENDPOINT="http://localhost:8000" uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and NVIDIA CUDA 12.x+.

### Tests

```bash
# Backend unit tests (no GPU needed, models mocked)
cd backend && uv run pytest tests/ -v

# Frontend unit tests
cd frontend && uv run pytest tests/ -v

# Integration tests (requires running server with GPU)
HF_TOKEN=hf_xxx ./scripts/run-integration-tests.sh
```

### Benchmark

```bash
./scripts/benchmark.sh                        # bundled 1-min clip
./scripts/benchmark.sh /path/to/long.wav      # custom file
```

### Commits and versioning

This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/) with [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning and changelog generation.

```bash
uv tool install commitizen
cz commit    # interactive conventional commit
cz bump      # bump version + update CHANGELOG.md
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [pyannote-audio](https://github.com/pyannote/pyannote-audio), and [HuggingFace Transformers](https://github.com/huggingface/transformers).
