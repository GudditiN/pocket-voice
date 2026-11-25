"""Create, test and delete the voice profile (reference-audio conditioning, not fine-tuning)."""
import time
from datetime import datetime

import streamlit as st

import tts
import ui
from audio import SILENCE_RMS, to_dbfs, wav_info
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

PASSAGES = {
    "Everyday": "Good morning! I'm recording my own voice for a personal assistant. I'll speak clearly, at a natural "
                "pace, as if I were explaining something to a friend. Whether it's the weather, a quick recipe, or "
                "my calendar, I want the answer to sound just like me.",
    "Varied sounds": "The quick brown fox jumps over the lazy dog. Six big boats sailed through the misty harbor at "
                     "dawn, while seagulls circled above. Please call me when the package arrives, and remind me to "
                     "buy fresh bread, oranges, and a little honey.",
    "Conversational": "Honestly, I think the best way to learn something new is to try it yourself. Start small, "
                      "make a few mistakes, and keep going. By the end of the week, you'll be surprised how much you "
                      "remember, and how much fun it was.",
}

st.subheader("1. Record or upload your voice")
st.markdown("Use **your own voice** (or a voice you have explicit permission to use), in a quiet room.")
consent = st.checkbox("This is my own voice, or I have explicit permission to use it.")

choice = st.selectbox("Reading passage", list(PASSAGES))
with st.container(border=True):
    st.caption("READ THIS ALOUD")
    st.markdown(f"<p style='font-size:1.25rem; line-height:1.7; margin:0'>{PASSAGES[choice]}</p>",
                unsafe_allow_html=True)
st.caption(f"Click the mic, wait a second, read the passage at a natural pace (about 15–20 s), then click stop. "
           f"Minimum {tts.MIN_REF_S:.0f} s. Longer recordings are trimmed to the first {tts.USE_REF_S:.0f} s.")

tab_rec, tab_up = st.tabs(["🎤 Record in browser", "⬆️ Upload a WAV"])
with tab_rec:
    recorded = st.audio_input("Record")
with tab_up:
    uploaded = st.file_uploader("WAV file (16-bit PCM)", type=["wav"])
wav = uploaded.getvalue() if uploaded else (recorded.getvalue() if recorded else None)

if wav:
    st.audio(wav, format="audio/wav")  # preview; replace by uploading/recording again
    try:
        dur, rms = wav_info(wav)
        st.caption(f"{dur:.1f} s · speech level {to_dbfs(rms):.0f} dBFS (needs above {to_dbfs(SILENCE_RMS):.0f}; "
                   f"normal speech is about -35 to -15)")
        if rms < SILENCE_RMS:
            st.error(
                "No sound was captured. The browser is not getting audio from your microphone.\n\n"
                "- Open this app in a normal Chrome / Edge / Safari **tab**, not VS Code's built-in preview or Simple "
                "Browser (those block the microphone).\n"
                "- Click the lock icon in the address bar, allow the microphone, and pick the right input device.\n"
                "- Check the Mac input level in System Settings → Sound → Input.\n"
                "- Or record in Voice Memos, export as WAV, and use the **Upload a WAV** tab.")
        elif to_dbfs(SILENCE_RMS) < to_dbfs(rms) < -35:
            st.warning("This recording is quiet. It will be boosted automatically, but also check your microphone "
                       "input volume and that the right microphone is selected.")
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
