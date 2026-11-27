"""Voice assistant: type or speak in one chat box, hear the answer in your voice profile."""
import uuid

import streamlit as st

import auth
import cards
import chat_style
import pipeline
import player
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
fast = st.toggle("⚡ Fast reply: start talking while the answer is still being written", value=ss.get("fast", s.fast_reply))
ss["fast"] = fast
if fast:
    with st.container(key="pvsound-unlock"):
        player.prepare()

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

def show_sources(sources) -> None:
    if sources:
        links = " · ".join(f"[{x['name']}]({x['url'].replace('(', '%28').replace(')', '%29')})" for x in sources)
        st.caption(f"🌐 Sources: {links}")


def reply_area(turn_id: str):
    """The top of an assistant reply. Live and past replies share this layout, so nothing is left behind on screen."""
    text_slot = st.empty()
    sound = st.container(key=f"pvsound-{turn_id}")  # hidden: holds the audio pieces of a fast reply
    status = st.status("Thinking…", expanded=False)
    return text_slot, sound, status


def reply_end(turn: dict, autoplay: bool = False) -> None:
    cards.show(turn.get("cards"), turn.get("images"))  # links first, then the player
    st.audio(turn["wav"], format="audio/wav", autoplay=autoplay)
    st.caption(f"AI-generated voice: {turn['voice']}")
    show_sources(turn.get("sources"))


for turn in ss["turns"]:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        text_slot, _, status = reply_area(turn["id"])
        text_slot.write(turn["a"])
        status.update(label=turn["timing"], state="complete")
        reply_end(turn)


def timing_line(result) -> str:
    t = result.timings_ms
    parts = ([f"ASR {t['asr'] / 1000:.1f}s"] if "asr" in t else []) + \
            [f"LLM {t.get('llm', 0) / 1000:.1f}s", f"TTS {t.get('tts', 0) / 1000:.1f}s"]
    first = f"first words after {result.first_audio_ms / 1000:.1f}s · " if result.first_audio_ms else ""
    return f"{first}total {t['total'] / 1000:.1f}s ({' · '.join(parts)})"


def process(audio_bytes=None, text=None):
    """Run one turn, showing (and in fast mode speaking) the reply as it is produced."""
    turn_id = uuid.uuid4().hex[:10]
    pieces = {"n": 0}
    with st.chat_message("user"):
        question_slot = st.empty()
        question_slot.write(text if text else "🎙️ …")
    with st.chat_message("assistant"):
        answer_slot, sound, status = reply_area(turn_id)
        writing = {"started": False}

        def think(label: str):  # animated bubble until the first words of the answer appear
            answer_slot.markdown(chat_style.thinking(label), unsafe_allow_html=True)

        def show_text(so_far: str):
            writing["started"] = True
            answer_slot.write(so_far)

        def on_status(label: str):
            status.update(label=label + ("…" if label not in ("Completed", "Failed") else ""))
            waiting = {"Transcribing": "Listening to you…", "Generating speech": "Getting my voice ready…",
                       "Generating response": "Searching and thinking…" if use_web else "Thinking…"}
            if label in waiting and not writing["started"]:
                think(waiting[label])

        think("Listening to you…" if audio_bytes is not None else "Thinking…")

        def say(wav: bytes, kind: str = "speech"):
            pieces["n"] += 1
            with sound:
                player.push(wav, turn_id, pieces["n"], kind)

        if fast and s.fillers:  # heard only if the answer takes longer than a moment
            phrase = tts.filler("search" if use_web else "think", voice)
            if phrase:
                say(phrase, "filler")
        result = pipeline.run_turn(
            audio_bytes, ss["history"], ss["prompt"], text=text, profile=voice, use_web=use_web,
            on_status=on_status,
            on_audio=say if fast else None, on_text=show_text, on_question=question_slot.write,
        )
        if result.status != "success":
            if not writing["started"]:
                answer_slot.empty()  # stop the thinking animation
            status.update(label=f"Failed: {result.error_message}", state="error", expanded=True)
            st.error(f"{result.error_message}  \n_Category: `{result.error_category}`_")
            return
        timing = timing_line(result)
        status.update(label=timing, state="complete")
        answer_slot.write(result.answer)
        turn = {"id": turn_id, "q": result.question, "a": result.answer, "wav": result.audio_wav, "voice": voice,
                "timing": timing, "sources": result.sources, "cards": result.cards, "images": result.images}
        # In fast mode the reply has already been played piece by piece: the player is for replaying it.
        reply_end(turn, autoplay=not fast)
    ss["history"] += [{"role": "user", "content": result.question},
                      {"role": "assistant", "content": result.answer}]
    ss["turns"].append(turn)


if message:
    if message.audio is not None:  # a recording wins over any text typed with it
        process(audio_bytes=message.audio.getvalue())
    elif message.text.strip():
        process(text=message.text)
elif not ss["turns"]:
    st.info("**Ready.** Type below or tap the 🎙️ button. Your conversation will appear here.")
