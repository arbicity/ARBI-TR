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


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0**20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)" % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


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
    model = None
    model_size = "tiny"
    with st.spinner(f"Loading Model {model_size}..."):
        model = load_model(model_size)

    sound_chunk = app_sst()

    if sound_chunk:
        sound_chunk = sound_chunk.set_channels(1).set_frame_rate(48000)

        st.audio(sound_chunk.export(format="wav").read(), format="audio/wav")
        arr = convert_to_whisper_format(sound_chunk)
        segments, info = model.transcribe(arr, beam_size=5)
        text = "".join([x.text for x in segments])

        print("text: ", text)
        st.markdown(f"**Text:** {text}")

        # st.session_state["text"] += text
        # st.session_state["texts"] += str(st.session_state["text"])
        # st.markdown(f"**Text:** {st.session_state['texts']}")


def app_sst():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024 * 10,  # TODO: return to default 1024
        rtc_configuration=None,
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    sound_chunk = pydub.AudioSegment.empty()
    while True:
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

                if len(sound_chunk) > 5_000:
                    print(f"sample_width = {audio_frame.format.bytes}")
                    print(f"frame_rate = {audio_frame.sample_rate}")
                    print(f"channels = {len(audio_frame.layout.channels)}")
                    status_indicator.write("Finished!")
                    return sound_chunk

            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


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
