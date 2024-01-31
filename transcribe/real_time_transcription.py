import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from faster_whisper import WhisperModel

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

if "texts" not in st.session_state:
    st.session_state["texts"] = ""
    st.session_state["text"] = ""


@st.cache_resource
def load_model(model_size="large-v3"):
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("MODEL LOADED!")
    return model


def convert_to_whisper_format(sound_chunk):
    if sound_chunk.frame_rate != 16000:  # 16 kHz
        sound_chunk = sound_chunk.set_frame_rate(16000)
    if sound_chunk.sample_width != 2:  # int16
        sound_chunk = sound_chunk.set_sample_width(2)
    if sound_chunk.channels != 1:  # mono
        sound_chunk = sound_chunk.set_channels(1)
    arr = np.array(sound_chunk.get_array_of_samples())
    arr = arr.astype(np.float32) / 32768.0
    return arr


def main():
    st.header("Real Time Speech-to-Text")
    global model
    model = None
    # model_size = "tiny"
    # model_size = "medium"
    model_size = "large-v3"
    total_time = st.slider("total_time", 0, 60_000, 30_000)
    window_time = st.slider("window_time", 0, 60_000, 5_000)

    with st.spinner(f"Loading Model {model_size}..."):
        model = load_model(model_size)

    fulltext = st.empty()

    total_sound_chunk = app_sst(fulltext, total_time, window_time)

    if total_sound_chunk:
        st.audio(total_sound_chunk.export(format="wav").read(), format="audio/wav")


def transcribe(sound_chunk):
    arr = convert_to_whisper_format(sound_chunk)
    segments, info = model.transcribe(arr, beam_size=5)
    return "".join([x.text for x in segments])


def app_sst(fulltext, total_time, window_time):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024 * 100,  # TODO: return to default 1024
        rtc_configuration=None,
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    total_sound_chunk = pydub.AudioSegment.empty()
    sound_chunk = pydub.AudioSegment.empty()

    while len(total_sound_chunk) < total_time:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                status_indicator.write("Running. Say something!")

                for audio_frame in audio_frames:
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    sound_chunk += sound

                if len(sound_chunk) > window_time:
                    total_sound_chunk += sound_chunk
                    text = transcribe(sound_chunk)
                    st.session_state["texts"] += "\n" + text
                    fulltext.write(f"**Text:** {st.session_state['texts']}")
                    sound_chunk = pydub.AudioSegment.empty()

            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

    return total_sound_chunk


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: " "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
