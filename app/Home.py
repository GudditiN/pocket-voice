"""Voice assistant: type or speak in one chat box, hear the answer in your voice profile."""
import streamlit as st

import auth
import chat_style
import pipeline
import tts
import ui
from config import get_settings

ui.setup_page("Assistant", "🎙️")
chat_style.apply()
s = get_settings()

ui.hero("🎙️ Voice assistant", "Type a message or tap the mic, and hear the answer in your own voice.")
ui.chips([
    ("Voice profile ready", "ok") if tts.profile_exists() else ("No voice profile yet", "warn"),
    ("Groq key set", "ok") if s.groq_api_key else ("Groq key missing", "bad"),
    (f"ASR {s.asr_model_label}", ""), (f"LLM {s.llm_model}", ""), (f"TTS {tts.MODEL_NAME}", ""),
])

ss = st.session_state
ss.setdefault("history", [])  # [{'role','content'}] for the LLM
ss.setdefault("turns", [])  # display records incl. audio bytes
ss.setdefault("prompt", s.system_prompt)

voices = [p["name"] for p in tts.list_profiles()]
voice = None
if voices:
    # Each visitor picks a voice for their own session; it does not change the default for others.
    preferred = ss.get("chosen_voice") if ss.get("chosen_voice") in voices else tts.active_profile()
    voice = st.selectbox("🔊 Answer in this voice", voices, index=voices.index(preferred))
    ss["chosen_voice"] = voice
use_web = st.toggle("🌐 Search the web for fresh facts", value=ss.get("use_web", s.live_data),
                    help="Searches and reads web pages, news, market prices and weather. Your search words are sent "
                         "to those services. Answers take a little longer.")
ss["use_web"] = use_web

with st.sidebar:
    st.subheader("Settings")
    if auth.can_manage_voices(s, ss.get("admin", False)):  # visitors on a public link keep the default prompt
        ss["prompt"] = st.text_area("System prompt", ss["prompt"], height=140)
    if st.button("Clear conversation"):
        ss["history"], ss["turns"] = [], []
        st.rerun()

if not voices:
    st.warning("No voice profile yet. Create one on the **Voice Setup** page first.")
if not s.groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it as an environment variable or Codespaces secret, then restart.")

# Pinned to the bottom of the page: text box with the microphone on its right.
message = st.chat_input("Type a message, or tap the mic to speak…", accept_audio=True, key="composer")

for i, turn in enumerate(ss["turns"]):
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        newest = i == len(ss["turns"]) - 1
        st.audio(turn["wav"], format="audio/wav", autoplay=bool(turn.pop("fresh", False)) and newest)
        st.caption(f"AI-generated voice: {turn['voice']} · {turn['timing']}")
        if turn.get("sources"):
            links = " · ".join(f"[{x['name']}]({x['url'].replace('(', '%28').replace(')', '%29')})" for x in turn["sources"])
            st.caption(f"🌐 Sources: {links}")


def process(audio_bytes=None, text=None):
    """Run one turn with a live status box; on success add it to the conversation and redraw."""
    with st.status("Processing…", expanded=True) as status:
        result = pipeline.run_turn(
            audio_bytes, ss["history"], ss["prompt"], text=text, profile=voice, use_web=use_web,
            on_status=lambda label: status.update(label=label + ("…" if label not in ("Completed", "Failed") else "")),
        )
        if result.status != "success":
            status.update(label=f"Failed: {result.error_message}", state="error", expanded=True)
    if result.status != "success":
        st.error(f"{result.error_message}  \n_Category: `{result.error_category}`_")
        return
    t = result.timings_ms
    parts = ([f"ASR {t['asr'] / 1000:.1f}s"] if "asr" in t else []) + \
            [f"LLM {t['llm'] / 1000:.1f}s", f"TTS {t['tts'] / 1000:.1f}s"]
    ss["history"] += [{"role": "user", "content": result.question},
                      {"role": "assistant", "content": result.answer}]
    ss["turns"].append({"q": result.question, "a": result.answer, "wav": result.audio_wav, "voice": voice,
                        "timing": f"{t['total'] / 1000:.1f}s ({' · '.join(parts)})", "sources": result.sources,
                        "fresh": True})
    st.rerun()


if message:
    if message.audio is not None:  # a recording wins over any text typed with it
        process(audio_bytes=message.audio.getvalue())
    elif message.text.strip():
        process(text=message.text)
elif not ss["turns"]:
    st.info("**Ready.** Type below or tap the 🎙️ button. Your conversation will appear here.")
