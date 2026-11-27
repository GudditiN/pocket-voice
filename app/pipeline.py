"""One voice turn: audio -> ASR -> LLM -> TTS, timed per stage and recorded in analytics."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import asr
import audio
import live
import llm
import sniffy
import store
import tts
from config import get_settings
from errors import AppError
from sentences import SentenceStream


@dataclass
class TurnResult:
    request_id: str
    status: str = "failed"  # success | failed
    question: str = ""
    answer: str = ""
    audio_wav: bytes | None = None
    error_category: str | None = None
    error_message: str | None = None
    in_audio_s: float | None = None
    out_audio_s: float | None = None
    timings_ms: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)  # web pages used for the answer: [{'name','url'}]
    cards: list = field(default_factory=list)  # links to show: [{'title','url','snippet','image'}]
    images: list = field(default_factory=list)  # extra picture addresses to show
    shops: list = field(default_factory=list)  # "search elsewhere" links
    context: str = ""  # what the screen showed, remembered for follow-up questions
    first_audio_ms: float | None = None  # fast replies: time until the first spoken piece was ready


MAX_QUESTION_CHARS = 1000


def _reply_while_writing(r: TurnResult, usage: dict, history, system_prompt, use_web, profile, started,
                         on_status, on_text, on_audio) -> None:
    """Read the answer as it streams and speak it sentence by sentence."""
    splitter, said, spoken, rate = SentenceStream(), [], [], None
    waiting = {"llm": 0.0, "tts": 0.0}

    def speak(piece: str) -> None:
        nonlocal rate
        piece = llm.clean_for_speech(piece)
        if not any(ch.isalnum() for ch in piece):
            return
        if not said:
            on_status("Speaking")
        said.append(piece)
        r.answer = " ".join(said)
        on_text(r.answer)
        t = time.perf_counter()
        for samples, rate in tts.synthesize_stream(piece, profile=profile):
            if r.first_audio_ms is None:
                r.first_audio_ms = (time.perf_counter() - started) * 1000
            spoken.append(samples)
            on_audio(audio.float_to_wav(samples, rate))
        waiting["tts"] += time.perf_counter() - t

    stream = llm.answer_stream(history, r.question, system_prompt, use_web, usage)
    while True:
        t = time.perf_counter()
        text_piece = next(stream, None)
        waiting["llm"] += time.perf_counter() - t
        if text_piece is None:
            break
        for sentence in splitter.feed(text_piece):
            speak(sentence)
    for sentence in splitter.flush():
        speak(sentence)
    if not spoken:
        raise AppError("llm", "Groq returned an empty answer.")
    whole = np.concatenate(spoken)
    r.audio_wav, r.out_audio_s = audio.float_to_wav(whole, rate), len(whole) / rate
    r.timings_ms.update({k: v * 1000 for k, v in waiting.items()})


def _show_products(r: TurnResult, profile, started, on_status, on_text, on_audio) -> bool:
    """Answer a product search with cards and a short stock phrase: no model writing, no long speech.

    Returns False if the shop couldn't be searched, so the normal answer can take over.
    """
    on_status("Finding products")
    t = time.perf_counter()
    session = live.Session(r.question)
    try:
        data = session.lookup("sniffynose", r.question[:300])
    except Exception:  # not found or the site is down: let the model answer instead
        return False
    r.timings_ms["search"] = (time.perf_counter() - t) * 1000
    r.cards, r.shops, r.sources = session.cards, session.shops, session.sources
    products = data.get("products") or []
    t = time.perf_counter()
    r.answer, r.audio_wav = tts.phrase("products" if products else "no_products", profile)
    r.timings_ms["tts"] = (time.perf_counter() - t) * 1000
    r.out_audio_s = audio.wav_info(r.audio_wav)[0]
    on_text(r.answer)
    if on_audio is not None:
        r.first_audio_ms = (time.perf_counter() - started) * 1000
        on_audio(r.audio_wav)
    if products:
        r.context = "Products shown on screen from SniffyNose: " + "; ".join(
            f"{i}. {p['name']}" + (f" (Rs {p['price_inr']})" if p.get("price_inr") is not None else "")
            for i, p in enumerate(products, 1))
    return True


def run_turn(audio_bytes: bytes | None, history: list[dict], system_prompt: str | None = None,
             on_status: Callable[[str], None] = lambda s: None, text: str | None = None,
             profile: str | None = None, use_web: bool = False,
             on_audio: Callable[[bytes], None] | None = None,
             on_text: Callable[[str], None] = lambda s: None,
             on_question: Callable[[str], None] = lambda s: None) -> TurnResult:
    """One turn. Pass a recording (audio_bytes), or typed text to skip speech recognition.

    profile: name of the voice that speaks the answer (default: the active one).
    on_audio: if given, reply fast: speak each sentence while the answer is still being written, passing WAV
    pieces to on_audio and the text so far to on_text.
    """
    s = get_settings()
    r = TurnResult(request_id=uuid.uuid4().hex[:12])
    t0, ts = time.perf_counter(), time.time()
    usage: dict = {}
    try:
        if not tts.profile_exists(profile):
            raise AppError("no_profile", "No voice profile yet. Create one on the Voice Setup page.")
        if text is not None:  # typed question: no speech recognition
            r.question = text.strip()[:MAX_QUESTION_CHARS]
            if not r.question:
                raise AppError("empty_input", "Type a question first.")
        else:
            r.in_audio_s, rms = audio.wav_info(audio_bytes or b"")
            if r.in_audio_s < 0.3 or rms < audio.SILENCE_RMS:
                raise AppError("bad_audio", "The recording is empty or silent. Try again.")
            audio_bytes = audio.normalize_wav(audio_bytes)  # help Whisper with quiet microphones

            on_status("Transcribing")
            t = time.perf_counter()
            r.question = asr.transcribe(audio_bytes)
            r.timings_ms["asr"] = (time.perf_counter() - t) * 1000
            if not r.question:
                raise AppError("bad_audio", "No speech was recognized. Try again.")

        on_question(r.question)
        quick = bool(use_web and s.sniffynose and s.product_quick_reply and sniffy.is_shopping(r.question)
                     and _show_products(r, profile, t0, on_status, on_text, on_audio))
        if not quick:
            on_status("Generating response")
        if quick:
            pass
        elif on_audio is None:
            t = time.perf_counter()
            r.answer, usage = llm.answer(history, r.question, system_prompt, use_web=use_web)
            r.timings_ms["llm"] = (time.perf_counter() - t) * 1000

            on_status("Generating speech")
            t = time.perf_counter()
            r.audio_wav, r.out_audio_s = tts.synthesize(r.answer, profile=profile)
            r.timings_ms["tts"] = (time.perf_counter() - t) * 1000
        else:
            _reply_while_writing(r, usage, history, system_prompt, use_web, profile, t0, on_status, on_text, on_audio)
        if not quick:
            r.sources = usage.get("sources", [])
            r.cards, r.images, r.shops = (usage.get(k, []) for k in ("cards", "images", "shops"))
        r.status = "success"
        on_status("Completed")
    except AppError as e:
        r.error_category, r.error_message = e.category, e.message
    except Exception as e:  # keep the UI alive; the category makes it visible in analytics
        r.error_category, r.error_message = "unknown", f"Unexpected error: {type(e).__name__}"
    finally:
        r.timings_ms["total"] = (time.perf_counter() - t0) * 1000
        if r.status != "success":
            on_status("Failed")
            store.log_error(r.error_category or "unknown", r.error_message or "")
        store.insert_request({
            "id": r.request_id, "ts": ts, "asr_ms": r.timings_ms.get("asr"), "llm_ms": r.timings_ms.get("llm"),
            "tts_ms": r.timings_ms.get("tts"), "total_ms": r.timings_ms["total"], "first_audio_ms": r.first_audio_ms,
            "in_audio_s": r.in_audio_s, "out_audio_s": r.out_audio_s, "status": r.status,
            "error_cat": r.error_category, "asr_model": "typed" if text is not None else s.asr_model_label, "llm_model": "product lookup (no model)" if "search" in r.timings_ms else s.llm_model,
            "tts_model": tts.MODEL_NAME, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "question_chars": len(r.question), "answer_chars": len(r.answer),  # lengths only, never content
        })
    return r
