"""Create, test and delete the voice profile (reference-audio conditioning, not fine-tuning)."""
import time
from datetime import datetime

import streamlit as st

import tts
import ui
from audio import wav_info
from errors import AppError

ui.setup_page("Voice Setup", "🎤")
st.title("🎤 Voice setup")
st.caption("PocketTTS conditions on a short recording of your voice and saves a reusable *voice profile*. "
           "This is not model training. The recording and profile stay on this machine, are never served by URL, "
           "and are git-ignored.")

info = tts.profile_info()
if info:
    st.success(f"Voice profile configured (created {datetime.fromtimestamp(info['created']):%Y-%m-%d %H:%M}, "
               f"{info['profile_kb']:.0f} KB).")
else:
    st.info("No voice profile yet.")

st.subheader("1. Provide a recording")
st.markdown(f"Use **your own voice** (or a voice you have explicit permission to use). "
            f"Speak clearly for **10–20 s** (at least {tts.MIN_REF_S:.0f} s; longer recordings are trimmed to the first "
            f"{tts.USE_REF_S:.0f} s) in a quiet room. WAV only.")
consent = st.checkbox("This is my own voice, or I have explicit permission to use it.")
c1, c2 = st.columns(2)
uploaded = c1.file_uploader("Upload a WAV file", type=["wav"])
recorded = c2.audio_input("…or record now")
wav = uploaded.getvalue() if uploaded else (recorded.getvalue() if recorded else None)

if wav:
    st.audio(wav, format="audio/wav")  # preview; replace by uploading/recording again
    try:
        dur, rms = wav_info(wav)
        st.caption(f"{dur:.1f} s · level {rms:.3f}")
    except AppError as e:
        st.error(e.message)

st.subheader("2. Build the profile")
if st.button("Create / replace voice profile", type="primary", disabled=not (wav and consent)):
    with st.spinner("Loading PocketTTS (first run downloads the model) and encoding your voice…"):
        t = time.perf_counter()
        try:
            res = tts.build_profile(wav)
            st.success(f"Profile saved in {time.perf_counter() - t:.1f}s "
                       f"(voice encoding {res['voice_state_s']:.1f}s). Future requests reuse it without re-encoding.")
            if res["trimmed"]:
                st.info(f"Your {res['original_s']:.0f} s recording was trimmed to its first {res['duration_s']:.0f} s.")
        except AppError as e:
            st.error(f"{e.message}  \n_Category: `{e.category}`_")
if not consent and wav:
    st.caption("Tick the consent box to enable profile creation.")

st.subheader("3. Test the voice")
sentence = st.text_input("Test sentence", "Hello! This is an AI-generated voice based on my own recording.")
if st.button("Speak", disabled=not tts.profile_exists()):
    try:
        with st.spinner("Generating speech…"):
            t = time.perf_counter()
            out, secs = tts.synthesize(sentence)
            took = time.perf_counter() - t
        st.audio(out, format="audio/wav")
        st.caption(f"Generated {secs:.1f}s of audio in {took:.1f}s (real-time factor {took / secs:.2f}; lower is faster).")
    except AppError as e:
        st.error(f"{e.message}  \n_Category: `{e.category}`_")
if tts.last_timings:
    st.caption("Last timings: " + " · ".join(f"{k.replace('_s', '').replace('_', ' ')} {v:.1f}s" for k, v in tts.last_timings.items()))

st.subheader("Delete")
if st.button("Delete voice profile and recording", disabled=not info):
    tts.delete_profile()
    st.rerun()
