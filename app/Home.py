"""Voice assistant: speak or type a question, hear the answer in your voice profile."""
import hashlib

import streamlit as st

import pipeline
import tts
import ui
from config import get_settings

ui.setup_page("Assistant", "🎙️")
s = get_settings()

ui.hero("🎙️ Voice assistant", "Speak or type a question and hear the answer in your own voice.")
ui.chips([
    ("Voice profile ready", "ok") if tts.profile_exists() else ("No voice profile yet", "warn"),
    ("Groq key set", "ok") if s.groq_api_key else ("Groq key missing", "bad"),
    (f"ASR {s.asr_model_label}", ""), (f"LLM {s.llm_model}", ""), (f"TTS {tts.MODEL_NAME}", ""),
])
ui.flow(["🎤 Listen or ⌨️ type", "📝 Transcribe", "🧠 Answer", "🔊 Speak"])

ss = st.session_state
ss.setdefault("history", [])  # [{'role','content'}] for the LLM
ss.setdefault("turns", [])  # display records incl. audio bytes
ss.setdefault("last_audio", None)
ss.setdefault("prompt", s.system_prompt)

with st.sidebar:
    st.subheader("Settings")
    ss["prompt"] = st.text_area("System prompt", ss["prompt"], height=140)
    if st.button("Clear conversation"):
        ss["history"], ss["turns"], ss["last_audio"] = [], [], None
        st.rerun()

if not tts.profile_exists():
    st.warning("No voice profile yet. Create one on the **Voice Setup** page first.")
if not s.groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it as an environment variable or Codespaces secret, then restart.")


def process(audio_bytes=None, text=None):
    """Run one turn with a live status box, then record it in the conversation."""
    with st.status("Processing…", expanded=True) as status:
        result = pipeline.run_turn(
            audio_bytes, ss["history"], ss["prompt"], text=text,
            on_status=lambda label: status.update(label=label + ("…" if label not in ("Completed", "Failed") else "")),
        )
        if result.status == "success":
            t = result.timings_ms
            parts = ([f"ASR {t['asr'] / 1000:.1f}s"] if "asr" in t else []) + \
                    [f"LLM {t['llm'] / 1000:.1f}s", f"TTS {t['tts'] / 1000:.1f}s"]
            status.update(label=f"Completed in {t['total'] / 1000:.1f}s ({' · '.join(parts)})",
                          state="complete", expanded=False)
        else:
            status.update(label=f"Failed: {result.error_message}", state="error", expanded=True)
    if result.status == "success":
        ss["history"] += [{"role": "user", "content": result.question},
                          {"role": "assistant", "content": result.answer}]
        ss["turns"].append({"q": result.question, "a": result.answer, "wav": result.audio_wav, "fresh": True})
    else:
        st.error(f"{result.error_message}  \n_Category: `{result.error_category}`_")


clip = st.audio_input("Click the mic, ask a question, click again to stop")
if clip is not None:
    data = clip.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    if digest != ss["last_audio"]:  # process each recording once
        ss["last_audio"] = digest
        process(audio_bytes=data)

typed = st.chat_input("…or type a question and hear the answer in your voice")
if typed:
    process(text=typed)

for i, turn in enumerate(ss["turns"]):
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        newest = i == len(ss["turns"]) - 1
        st.audio(turn["wav"], format="audio/wav", autoplay=bool(turn.pop("fresh", False)) and newest)
if not ss["turns"]:
    st.info("Status: **Ready**. Your conversation will appear here.")
