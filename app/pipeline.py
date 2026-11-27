"""One voice turn: audio -> ASR -> LLM -> TTS, timed per stage and recorded in analytics."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import asr
import audio
import llm
import store
import tts
from config import get_settings
from errors import AppError


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


MAX_QUESTION_CHARS = 1000


def run_turn(audio_bytes: bytes | None, history: list[dict], system_prompt: str | None = None,
             on_status: Callable[[str], None] = lambda s: None, text: str | None = None,
             profile: str | None = None, use_web: bool = False) -> TurnResult:
    """One turn. Pass a recording (audio_bytes), or typed text to skip speech recognition.

    profile: name of the voice that speaks the answer (default: the active one).
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

        on_status("Generating response")
        t = time.perf_counter()
        r.answer, usage = llm.answer(history, r.question, system_prompt, use_web=use_web)
        r.sources = usage.get("sources", [])
        r.timings_ms["llm"] = (time.perf_counter() - t) * 1000

        on_status("Generating speech")
        t = time.perf_counter()
        r.audio_wav, r.out_audio_s = tts.synthesize(r.answer, profile=profile)
        r.timings_ms["tts"] = (time.perf_counter() - t) * 1000
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
            "tts_ms": r.timings_ms.get("tts"), "total_ms": r.timings_ms["total"],
            "in_audio_s": r.in_audio_s, "out_audio_s": r.out_audio_s, "status": r.status,
            "error_cat": r.error_category, "asr_model": "typed" if text is not None else s.asr_model_label, "llm_model": s.llm_model,
            "tts_model": tts.MODEL_NAME, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "question_chars": len(r.question), "answer_chars": len(r.answer),  # lengths only, never content
        })
    return r
