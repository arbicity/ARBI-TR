"""
Unit tests for ARBI-TR utils — pure functions only, no model loading.
"""

import pytest
from utils import convert_time, merge_transcription_with_diarization, _assign_speaker


# ---------------------------------------------------------------------------
# convert_time
# ---------------------------------------------------------------------------


def test_convert_time_zero():
    assert convert_time(0) == "0:00:00"


def test_convert_time_seconds():
    assert convert_time(65) == "0:01:05"


def test_convert_time_hours():
    assert convert_time(3661) == "1:01:01"


def test_convert_time_rounds():
    # 1.6 rounds to 2
    assert convert_time(1.6) == "0:00:02"


def test_convert_time_none():
    assert convert_time(None) == "00:00:00"


# ---------------------------------------------------------------------------
# _assign_speaker
# ---------------------------------------------------------------------------


def test_assign_speaker_exact_match():
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    assert _assign_speaker(0.0, 5.0, diar) == "SPEAKER_00"


def test_assign_speaker_max_overlap():
    diar = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    # chunk [0, 3]: overlap with S00 = 3s, overlap with S01 = 1s → S00 wins
    assert _assign_speaker(0.0, 3.0, diar) == "SPEAKER_00"
    # chunk [3, 6]: overlap with S00 = 0s, overlap with S01 = 3s → S01 wins
    assert _assign_speaker(3.0, 6.0, diar) == "SPEAKER_01"


def test_assign_speaker_no_overlap_defaults():
    diar = [{"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00"}]
    # chunk [0, 5] has no overlap → default SPEAKER_00
    assert _assign_speaker(0.0, 5.0, diar) == "SPEAKER_00"


def test_assign_speaker_empty_diarization():
    assert _assign_speaker(0.0, 5.0, []) == "SPEAKER_00"


# ---------------------------------------------------------------------------
# merge_transcription_with_diarization
# ---------------------------------------------------------------------------


def test_merge_empty_chunks():
    result = merge_transcription_with_diarization([], [])
    assert result == {"segments": []}


def test_merge_single_speaker():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "Hello"},
        {"timestamp": (2.0, 4.0), "text": "world"},
    ]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    result = merge_transcription_with_diarization(chunks, diar, merge_gap_s=1.0)
    # Both chunks same speaker, gap = 0 ≤ 1s → merged
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["Speaker"] == "SPEAKER_00"
    assert "Hello" in seg["Text"]
    assert "world" in seg["Text"]
    assert seg["Start"] == "0:00:00"
    assert seg["End"] == "0:00:04"


def test_merge_speaker_change():
    chunks = [
        {"timestamp": (0.0, 3.0), "text": "I think"},
        {"timestamp": (3.0, 6.0), "text": "I agree"},
    ]
    diar = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    result = merge_transcription_with_diarization(chunks, diar)
    assert len(result["segments"]) == 2
    assert result["segments"][0]["Speaker"] == "SPEAKER_00"
    assert result["segments"][1]["Speaker"] == "SPEAKER_01"


def test_merge_gap_prevents_merge():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "First"},
        {"timestamp": (5.0, 7.0), "text": "Second"},  # 3s gap
    ]
    diar = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
    # merge_gap_s=1.0 → gap of 3s > 1s, so NOT merged even same speaker
    result = merge_transcription_with_diarization(chunks, diar, merge_gap_s=1.0)
    assert len(result["segments"]) == 2


def test_merge_gap_allows_merge():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "First"},
        {"timestamp": (2.3, 4.0), "text": "Second"},  # 0.3s gap
    ]
    diar = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
    # merge_gap_s=0.5 → gap 0.3s ≤ 0.5s → merged
    result = merge_transcription_with_diarization(chunks, diar, merge_gap_s=0.5)
    assert len(result["segments"]) == 1


def test_merge_skips_chunks_with_none_end():
    chunks = [
        {"timestamp": (0.0, None), "text": "Bad chunk"},
        {"timestamp": (1.0, 3.0), "text": "Good chunk"},
    ]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    result = merge_transcription_with_diarization(chunks, diar)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["Text"] == "Good chunk"


def test_merge_skips_empty_text():
    chunks = [
        {"timestamp": (0.0, 1.0), "text": "   "},
        {"timestamp": (1.0, 3.0), "text": "Real text"},
    ]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    result = merge_transcription_with_diarization(chunks, diar)
    assert len(result["segments"]) == 1
    assert result["segments"][0]["Text"] == "Real text"


def test_merge_output_fields():
    chunks = [{"timestamp": (0.0, 2.0), "text": "Test"}]
    diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    result = merge_transcription_with_diarization(chunks, diar)
    seg = result["segments"][0]
    assert set(seg.keys()) == {"Start", "End", "Speaker", "Text"}


def test_merge_three_speakers_no_consecutive_merge():
    chunks = [
        {"timestamp": (0.0, 2.0), "text": "A"},
        {"timestamp": (2.0, 4.0), "text": "B"},
        {"timestamp": (4.0, 6.0), "text": "A again"},
    ]
    diar = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
    ]
    result = merge_transcription_with_diarization(chunks, diar)
    # S00 → S01 → S00: the last S00 is NOT merged with the first S00
    assert len(result["segments"]) == 3
