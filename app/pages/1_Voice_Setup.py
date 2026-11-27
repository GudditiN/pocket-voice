"""Manage voice profiles: create, test, choose the default and delete (reference conditioning, not fine-tuning)."""
import hashlib
import time
from datetime import datetime

import streamlit as st

import effects
import quality
import tts
import ui
from audio import background_level, decode_to_wav
from config import get_settings
from errors import AppError

ui.setup_page("Voice Setup", "🎤")
ui.hero("🎤 Voices", "Keep several voice profiles, test them, and choose which one speaks.")
ui.flow(["1 Record takes", "2 Build the voice", "3 Choose its sound", "4 Use it"])
st.caption("PocketTTS conditions on a short recording and saves a reusable *voice profile*. This is not model "
           "training. Recordings and profiles stay on this machine, are never served by URL, and are git-ignored.")

ss = st.session_state
if "flash" in ss:
    st.success(ss.pop("flash"))

manage = ui.voice_admin()
profiles = tts.list_profiles()
names = [p["name"] for p in profiles]
active = tts.active_profile()

def describe(tuning: dict) -> str:
    return (f"expression {tuning['temp']:.2f} · steps {tuning['steps']} · pitch {tuning['pitch']:+.1f} · "
            f"speed ×{tuning['speed']:.2f} · cleanup {effects.CLEAN_LEVELS[tuning['clean']]}")


# ---------------------------------------------------------------- your voices
st.subheader("Your voices")
if not profiles:
    st.info("No voice profile yet." + (" Add one below." if manage else ""))
else:
    sentence = st.text_input("Test sentence", "Hello! This is an AI-generated voice based on my own recording.")
    for p in profiles:
        name = p["name"]
        with st.container(border=True):
            info_col, test_col, default_col, delete_col = st.columns([4, 1, 2, 1])
            info_col.markdown(f"**{name}**" + ("  ·  ⭐ default" if name == active else ""))
            info_col.caption(f"created {datetime.fromtimestamp(p['created']):%Y-%m-%d %H:%M} · {p['kb']:.0f} KB")
            if tts.is_tuned(name):
                info_col.caption(describe(tts.get_tuning(name)))
            test = test_col.button("▶ Test", key=f"test_{name}")
            if manage:
                if default_col.button("⭐ Make default", key=f"default_{name}", disabled=name == active):
                    tts.set_active(name)
                    ss["flash"] = f"'{name}' is now the default voice."
                    st.rerun()
                with delete_col.popover("🗑"):
                    st.write(f"Delete **{name}** and its recording? This cannot be undone.")
                    if st.button("Yes, delete", key=f"delete_{name}", type="primary"):
                        tts.delete_profile(name)
                        ss["flash"] = f"Deleted '{name}'."
                        st.rerun()
            if test:
                try:
                    with st.spinner(f"Speaking in '{name}'…"):
                        t = time.perf_counter()
                        out, secs = tts.synthesize(sentence, profile=name)
                        took = time.perf_counter() - t
                    st.audio(out, format="audio/wav", autoplay=True)
                    st.caption(f"{secs:.1f}s of audio in {took:.1f}s (real-time factor {took / secs:.2f}; lower is faster).")
                except AppError as e:
                    st.error(f"{e.message}  \n_Category: `{e.category}`_")

if not manage:
    st.stop()  # visitors on the shared link can listen, but not change voices

# ---------------------------------------------------------------- add or replace
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

st.subheader("Add or replace a voice")
suggested = "default" if not names else next(f"voice-{i}" for i in range(2, len(names) + 3) if f"voice-{i}" not in names)
raw_name = st.text_input("Voice name", suggested, help="Letters, numbers, - and _. Use an existing name to replace it.")
try:
    new_name = tts.slug(raw_name)
except AppError as e:
    new_name = None
    st.error(e.message)
if new_name in names:
    st.warning(f"A voice called **{new_name}** exists and will be replaced.")
elif new_name and new_name != raw_name:
    st.caption(f"It will be saved as **{new_name}**.")
make_default = st.checkbox("Make this the default voice", value=True)

st.markdown("Use **your own voice** (or a voice you have explicit permission to use), in a quiet room.")
consent = st.checkbox("This is my own voice, or I have explicit permission to use it.")

choice = st.selectbox("Reading passage", list(PASSAGES))
with st.container(border=True):
    st.caption("READ THIS ALOUD")
    st.markdown(f"<p style='font-size:1.25rem; line-height:1.7; margin:0'>{PASSAGES[choice]}</p>",
                unsafe_allow_html=True)
st.caption(f"Click the mic, wait a second, read the passage at a natural pace (about 15–20 s), then click stop. "
           f"Try each of the three passages and keep each take: the best one is picked for you. Minimum "
           f"{tts.MIN_REF_S:.0f} s. Silence at the ends is cut, and the first {tts.USE_REF_S:.0f} s are used.")

tab_rec, tab_up = st.tabs(["🎤 Record in browser", "⬆️ Upload a file"])
with tab_rec:
    recorded = st.audio_input("Record")
with tab_up:
    uploaded = st.file_uploader("Audio file (WAV, MP3, FLAC or OGG)", type=["wav", "mp3", "flac", "ogg"],
                                help="M4A/AAC isn't supported. Convert it to MP3 or WAV first.")
wav = None
if uploaded:
    try:
        wav = decode_to_wav(uploaded.getvalue())
    except AppError as e:
        st.error(e.message)
elif recorded:
    wav = recorded.getvalue()

MAX_TAKES = 3
takes = ss.setdefault("takes", [])  # recordings kept for comparison: in memory only, gone when the session ends


def show_report(report: dict) -> None:
    st.progress(report["score"] / 100, text=f"{report['verdict']} · score {report['score']}/100")
    st.caption(f"{report['speech_s']:.0f} s of speech · your voice {report['speech_db']:.0f} dBFS · background "
               f"{report['noise_db']:.0f} dBFS")
    for tip in report["advice"]:
        st.markdown(f"- {tip}")


if wav:
    st.audio(wav, format="audio/wav")  # preview; replace by uploading/recording again
    try:
        report = quality.check(wav)
        if report["score"] == 0:
            st.error(
                "No sound was captured. The browser is not getting audio from your microphone.\n\n"
                "- Open this app in a normal Chrome / Edge / Safari **tab**, not VS Code's built-in preview or Simple "
                "Browser (those block the microphone).\n"
                "- Click the lock icon in the address bar, allow the microphone, and pick the right input device.\n"
                "- Check the Mac input level in System Settings → Sound → Input.\n"
                "- Or record in another app and use the **Upload** tab (WAV, MP3, FLAC or OGG).")
        else:
            show_report(report)
            digest = hashlib.sha1(wav).hexdigest()
            if digest not in [take["id"] for take in takes]:
                if st.button(f"➕ Keep this take ({len(takes)} of {MAX_TAKES} kept)", disabled=len(takes) >= MAX_TAKES):
                    takes.append({"id": digest, "wav": wav, "report": report})
                    st.rerun()
    except AppError as e:
        st.error(e.message)

if takes:
    st.markdown("**Your takes**")
    best = max(range(len(takes)), key=lambda i: takes[i]["report"]["score"])
    for i, take in enumerate(takes):
        info, player, remove = st.columns([2, 4, 1])
        info.markdown(f"Take {i + 1}: **{take['report']['score']}/100** {take['report']['verdict']}"
                      + ("  ·  ⭐ best" if i == best else ""))
        player.audio(take["wav"], format="audio/wav")
        if remove.button("Remove", key=f"remove_{take['id']}"):
            takes.pop(i)
            st.rerun()
    picked = st.radio("Build the voice from", range(len(takes)), index=best, horizontal=True,
                      format_func=lambda i: f"Take {i + 1}")
    wav = takes[picked]["wav"]
    st.caption("The voice is built from the take selected here. Keep a new recording to compare it.")

LEVELS = ["off", "light", "medium", "strong"]
denoise = st.radio("Reduce background noise", LEVELS, index=LEVELS.index(get_settings().denoise), horizontal=True,
                   help="For steady noise such as a fan, hiss or hum. It cannot remove other voices or TV.")
if wav and denoise != "off":
    try:
        before, _ = tts.prepare_reference(wav, "off")
        after, _ = tts.prepare_reference(wav, denoise)
        left, right = st.columns(2)
        left.caption(f"Original · background {background_level(before):.0f} dBFS")
        left.audio(before, format="audio/wav")
        right.caption(f"Cleaned ({denoise}) · background {background_level(after):.0f} dBFS")
        right.audio(after, format="audio/wav")
        st.caption("Lower background is quieter. Listen to both: if your voice sounds metallic or muffled, "
                   "choose a lighter level.")
    except AppError as e:
        st.error(e.message)

if st.button("Create voice", type="primary", disabled=not (wav and consent and new_name)):
    with st.spinner("Loading PocketTTS (first run downloads the model) and encoding the voice…"):
        t = time.perf_counter()
        try:
            res = tts.build_profile(wav, name=new_name, make_active=make_default, denoise=denoise)
            note = (f" The first {res['duration_s']:.0f}s of speech from the {res['original_s']:.0f}s recording were used."
                    if res["trimmed"] else "")
            ss.pop("takes", None)  # the kept recordings are no longer needed
            ss["tune_next"] = res["name"]
            ss.pop("compare", None)
            ss["flash"] = (f"Voice '{res['name']}' saved in {time.perf_counter() - t:.1f}s.{note} "
                           "Next: choose how it sounds at the bottom of this page.")
            st.rerun()
        except AppError as e:
            st.error(f"{e.message}  \n_Category: `{e.category}`_")
if not consent and wav:
    st.caption("Tick the consent box to enable voice creation.")

# ---------------------------------------------------------------- step 3: how the voice sounds
STYLE, MIX = "Speaking style", "Mix your own"


def speak_variations(group: str, text: str, voice: str, custom: dict | None) -> list[dict]:
    """One audio clip per choice. Pitch, speed and cleanup reuse one take, so only that setting differs."""
    if group == STYLE:
        choices = tts.style_choices(voice)
    elif group == MIX:
        choices = [("Your mix", tts.clean_tuning(custom))]
    else:
        choices = tts.variation_choices(group, voice)
        base, _ = tts.synthesize(text, profile=voice, tuning={**tts.get_tuning(voice), **effects.NEUTRAL})
    items = []
    for label, tuning in choices:
        if group in effects.VARIATIONS:
            out, secs = effects.process_wav(base, tuning)
        else:
            out, secs = tts.synthesize(text, profile=voice, tuning=tuning)
        items.append({"label": label, "tuning": tuning, "wav": out, "secs": secs})
    return items


if profiles:
    st.subheader("Choose how a voice sounds")
    st.caption("Hear one sentence in different ways and keep what sounds right. Each voice keeps its own choice.")
    if "tune_next" in ss:  # a voice that was just created
        ss["compare_voice"] = ss.pop("tune_next")
    if ss.get("compare_voice") not in names:
        ss["compare_voice"] = active
    voice_col, line_col = st.columns([1, 3])
    target = voice_col.selectbox("Voice", names, key="compare_voice")
    line = line_col.text_input("Sentence", "Honestly, I think this sounds a lot like me. What do you think?",
                               key="compare_line")
    saved = tts.get_tuning(target)
    now_col, reset_col = st.columns([5, 1])
    now_col.caption(f"**{target}** now: {describe(saved)}")
    if reset_col.button("↺ Reset", disabled=not tts.is_tuned(target)):
        tts.reset_tuning(target)
        ss.pop("compare", None)
        ss["flash"] = f"'{target}' is back to the default sound."
        st.rerun()
    group = st.radio("What to compare", [STYLE, *effects.VARIATIONS, MIX], horizontal=True, key="compare_group")
    custom = None
    if group == MIX:
        left, right = st.columns(2)
        custom = {
            **saved,
            "temp": left.slider("Expression", 0.05, 1.5, float(saved["temp"]), 0.05),
            "steps": left.slider("Quality steps (higher is slower)", 1, 8, saved["steps"]),
            "pitch": right.slider("Pitch (lower ← → higher)", -4.0, 4.0, float(saved["pitch"]), 0.5),
            "speed": right.slider("Speed", 0.75, 1.3, float(saved["speed"]), 0.01),
            "clean": effects.CLEAN_LEVELS.index(right.select_slider(
                "Noise cleanup", options=effects.CLEAN_LEVELS, value=effects.CLEAN_LEVELS[saved["clean"]])),
        }
    if st.button("🎚️ Preview" if group == MIX else "🎚️ Compare"):
        ss.pop("compare", None)
        try:
            with st.spinner("Speaking the sentence…"):
                ss["compare"] = {"voice": target, "group": group,
                                 "items": speak_variations(group, line, target, custom)}
        except AppError as e:
            st.error(f"{e.message}  \n_Category: `{e.category}`_")
    shown = ss.get("compare")
    if shown and shown["group"] == group and shown["voice"] == target:
        for i, (column, item) in enumerate(zip(st.columns(len(shown["items"])), shown["items"])):
            column.markdown(f"**{item['label']}**")
            column.audio(item["wav"], format="audio/wav")
            column.caption(describe(item["tuning"]))
            if item["label"] != "Current" and column.button("Use this", key=f"use_{i}"):
                tts.save_tuning(item["tuning"], voice=target)
                ss.pop("compare", None)
                ss["flash"] = f"Saved for '{target}': {describe(item['tuning'])}."
                st.rerun()
    st.caption("Pitch changes of more than 1.5 sound less like you. With pitch, speed or cleanup on, each sentence "
               "is finished before it plays, so replies start a little later.")
