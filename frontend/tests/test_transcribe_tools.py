"""
Unit tests for transcribe_tools — pure logic only, no Streamlit or network calls.
"""

import os
import types
from unittest.mock import MagicMock, patch

import pytest

from transcribe_tools import (
    handle_response,
    load_languages,
    poll_status,
    process_file,
    secure_request,
)


# ---------------------------------------------------------------------------
# load_languages
# ---------------------------------------------------------------------------


def test_load_languages(tmp_path):
    langs_file = tmp_path / "langs.txt"
    langs_file.write_text("english\nfrench\ngerman\n")
    result = load_languages(str(langs_file))
    assert result == ["english", "french", "german"]


def test_load_languages_empty(tmp_path):
    langs_file = tmp_path / "langs.txt"
    langs_file.write_text("")
    result = load_languages(str(langs_file))
    # splitlines() on empty string returns []
    assert result == []


# ---------------------------------------------------------------------------
# handle_response
# ---------------------------------------------------------------------------


def _mock_response(status_code, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_handle_response_success():
    resp = _mock_response(200, {"session_id": "abc-123"})
    result = handle_response(resp, "test.wav")
    assert result["session_id"] == "abc-123"
    assert "message" in result
    assert "test.wav" in result["message"]


def test_handle_response_error():
    resp = _mock_response(500, text="Internal Server Error")
    result = handle_response(resp, "test.wav")
    assert result["session_id"] is None
    assert "error" in result
    assert "500" in result["error"]


def test_handle_response_422():
    resp = _mock_response(422, text="Validation error")
    result = handle_response(resp, "bad.mp3")
    assert result["session_id"] is None
    assert "422" in result["error"]


# ---------------------------------------------------------------------------
# secure_request (mTLS off)
# ---------------------------------------------------------------------------


@patch("transcribe_tools.requests.get")
def test_secure_request_get(mock_get):
    mock_get.return_value = _mock_response(200, {"status": "ok"})
    resp = secure_request("get", "http://localhost:8000/health")
    mock_get.assert_called_once()
    assert resp.status_code == 200


@patch("transcribe_tools.requests.post")
def test_secure_request_post(mock_post):
    mock_post.return_value = _mock_response(200, {"session_id": "x"})
    resp = secure_request("post", "http://localhost:8000/transcribe/")
    mock_post.assert_called_once()


@patch("transcribe_tools.requests")
def test_secure_request_unsupported_method(mock_requests):
    with pytest.raises(ValueError, match="Method not supported"):
        secure_request("delete", "http://localhost:8000/")


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


@patch("transcribe_tools.secure_request")
def test_process_file_success(mock_req):
    mock_req.return_value = _mock_response(200, {"session_id": "sess-1"})
    result = process_file(
        b"fake-audio-data", "test.wav",
        "large", "transcribe", "*Autodetect", "*Autodetect",
    )
    assert result["session_id"] == "sess-1"
    # Verify autodetect values are converted correctly
    call_kwargs = mock_req.call_args
    data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert data["source_language"] == ""
    assert data["speaker_number"] == "0"


@patch("transcribe_tools.secure_request")
def test_process_file_with_language(mock_req):
    mock_req.return_value = _mock_response(200, {"session_id": "sess-2"})
    result = process_file(
        b"fake-audio-data", "test.wav",
        "small", "translate", "french", "3",
    )
    call_kwargs = mock_req.call_args
    data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert data["source_language"] == "french"
    assert data["speaker_number"] == "3"
    assert data["task_str"] == "translate"
    assert data["size_of_model"] == "small"


# ---------------------------------------------------------------------------
# poll_status
# ---------------------------------------------------------------------------


@patch("transcribe_tools.secure_request")
@patch("transcribe_tools.time")
def test_poll_status_completed(mock_time, mock_req):
    mock_time.sleep = MagicMock()
    mock_req.return_value = _mock_response(
        200,
        {
            "status": "completed",
            "segments": [
                {"Start": "0:00:00", "End": "0:00:05", "Speaker": "SPEAKER_00", "Text": "Hello"}
            ],
        },
    )
    results = list(poll_status("sess-1", "test.wav"))
    # Should yield a DataFrame
    assert any(hasattr(r, "columns") for r in results)


@patch("transcribe_tools.secure_request")
@patch("transcribe_tools.time")
def test_poll_status_failed(mock_time, mock_req):
    mock_time.sleep = MagicMock()
    mock_req.return_value = _mock_response(
        200, {"status": "failed", "error": "CUDA OOM"}
    )
    results = list(poll_status("sess-1", "test.wav"))
    assert any("failed" in str(r).lower() or "CUDA OOM" in str(r) for r in results)


@patch("transcribe_tools.secure_request")
@patch("transcribe_tools.time")
def test_poll_status_queued_then_completed(mock_time, mock_req):
    mock_time.sleep = MagicMock()
    queued_resp = _mock_response(200, {"status": "queued", "position": 2})
    completed_resp = _mock_response(
        200,
        {
            "status": "completed",
            "segments": [
                {"Start": "0:00:00", "End": "0:00:03", "Speaker": "SPEAKER_00", "Text": "Hi"}
            ],
        },
    )
    mock_req.side_effect = [queued_resp, completed_resp]
    results = list(poll_status("sess-1", "test.wav"))
    # Should have a queue position message and a DataFrame
    assert any("queue position" in str(r).lower() for r in results)
    assert any(hasattr(r, "columns") for r in results)


@patch("transcribe_tools.secure_request")
@patch("transcribe_tools.time")
def test_poll_status_server_error(mock_time, mock_req):
    mock_time.sleep = MagicMock()
    mock_req.return_value = _mock_response(500, text="Internal error")
    results = list(poll_status("sess-1", "test.wav"))
    assert any("failed to fetch" in str(r).lower() or "error" in str(r).lower() for r in results)
