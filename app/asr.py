"""Speech-to-text: Groq Whisper (default) or local faster-whisper (optional extra)."""
from __future__ import annotations

import io

from config import get_settings
from errors import AppError, from_groq

_local_model = None


def _groq_client(timeout: int):
    from groq import Groq

    key = get_settings().groq_api_key
    if not key:
        raise AppError("missing_key", "GROQ_API_KEY is not set. Add it as an environment variable or Codespaces secret.")
    return Groq(api_key=key, timeout=timeout, max_retries=1)


def transcribe(wav_bytes: bytes) -> str:
    s = get_settings()
    if s.asr_backend == "local":
        return _transcribe_local(wav_bytes)
    client = _groq_client(s.request_timeout_s)
    try:
        resp = client.audio.transcriptions.create(
            file=("question.wav", wav_bytes), model=s.asr_model, language="en", temperature=0.0
        )
    except AppError:
        raise
    except Exception as e:
        raise from_groq(e, "asr") from e
    return (getattr(resp, "text", "") or "").strip()


def _transcribe_local(wav_bytes: bytes) -> str:
    global _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise AppError("asr", "Local ASR needs `uv sync --extra local-asr`.") from None
    try:
        if _local_model is None:
            _local_model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = _local_model.transcribe(io.BytesIO(wav_bytes), language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as e:
        raise AppError("asr", f"Local transcription failed: {e}") from e
