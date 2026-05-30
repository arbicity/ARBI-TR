# arbi-tr

Auto-generated Python client for the [ARBI-TR](https://github.com/arbicity/ARBI-TR) transcription API
(faster-whisper + pyannote diarization), generated from the service's OpenAPI spec by
`openapi-python-client`.

> This package is generated. Do not edit it by hand — run `./scripts/generate-client.sh` in the
> ARBI-TR repo to regenerate it from the current API.

## Install

```bash
pip install -e client/        # local, from a generated checkout
```

## Usage

Create a client pointed at your ARBI-TR deployment:

```python
from arbi_tr_client import Client

client = Client(base_url="http://localhost:8000")
```

### Async job API (transcribe → poll)

```python
from arbi_tr_client.api.default import (
    transcribe_audio_endpoint_transcribe_post as submit,
    get_task_status_task_status_session_id_get as status,
)
from arbi_tr_client.models import BodyTranscribeAudioEndpointTranscribePost
from arbi_tr_client.types import File

with open("meeting.mp3", "rb") as fh:
    body = BodyTranscribeAudioEndpointTranscribePost(
        file=File(payload=fh, file_name="meeting.mp3", mime_type="audio/mpeg"),
        task_str="transcribe",
    )
    submitted = submit.sync(client=client, body=body)

result = status.sync(client=client, session_id=submitted.session_id)
```

### OpenAI-compatible endpoint

The service also exposes `POST /v1/audio/transcriptions`, so the standard OpenAI SDK works as a
drop-in client. Use `transcribe_openai_v1_audio_transcriptions_post` from this package for a typed call.
