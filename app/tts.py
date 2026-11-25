"""PocketTTS wrapper: one cached model, one reusable voice profile, strictly sequential synthesis.

The "voice profile" is reference-audio conditioning (a saved voice state), NOT fine-tuning.
pocket_tts is imported lazily so the rest of the app (and unit tests) work without it.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time

import numpy as np

from audio import SILENCE_RMS, float_to_wav, normalize_wav, to_dbfs, trim_wav, wav_info
from config import get_settings
from errors import AppError

log = logging.getLogger(__name__)

MODEL_NAME = "pocket-tts"
MIN_REF_S = 5.0  # shorter recordings are rejected
USE_REF_S = 20.0  # longer recordings are trimmed to their first USE_REF_S seconds
MAX_INPUT_S = 600.0  # recordings longer than this are rejected outright
MAX_TEXT_CHARS = 800

_lock = threading.Lock()  # PocketTTS is not assumed thread-safe: one synthesis at a time
_model = None
_state = None
_state_mtime: float | None = None
last_timings: dict[str, float] = {}


def validate_reference(wav_bytes: bytes) -> float:
    """Check a reference recording. Returns duration in seconds or raises AppError('bad_audio')."""
    duration, rms = wav_info(wav_bytes)
    if duration < MIN_REF_S:
        raise AppError("bad_audio", f"Recording is {duration:.1f}s; record at least {MIN_REF_S:.0f}s of clear speech.")
    if duration > MAX_INPUT_S:
        raise AppError("bad_audio", f"Recording is {duration / 60:.0f} min; please use a clip under {MAX_INPUT_S / 60:.0f} min (10-20s is ideal).")
    if rms < SILENCE_RMS:
        raise AppError("bad_audio", f"The recording is silent or too quiet (speech level {to_dbfs(rms):.0f} dBFS; "
                                    f"need above {to_dbfs(SILENCE_RMS):.0f}). Check the mic isn't muted or too far away.")
    return duration


def _get_model():
    global _model
    if _model is None:
        t = time.perf_counter()
        try:
            from pocket_tts import TTSModel

            _model = TTSModel.load_model()
        except ImportError:
            raise AppError("tts", "pocket-tts is not installed. Run `uv sync`.") from None
        except Exception as e:  # download failures, auth errors, out of disk, ...
            raise AppError("tts", f"Could not load the PocketTTS model: {e}") from e
        last_timings["model_load_s"] = time.perf_counter() - t
        log.info("PocketTTS loaded in %.1fs", last_timings["model_load_s"])
    return _model


def profile_exists() -> bool:
    return get_settings().profile_path.exists()


def profile_info() -> dict | None:
    s = get_settings()
    if not s.profile_path.exists():
        return None
    return {
        "created": s.profile_path.stat().st_mtime,
        "profile_kb": s.profile_path.stat().st_size / 1024,
        "has_reference": s.reference_path.exists(),
    }


def build_profile(wav_bytes: bytes) -> dict:
    """Validate the recording, store it privately, build the voice state and save it to disk."""
    global _state, _state_mtime
    s = get_settings()
    original = validate_reference(wav_bytes)
    trimmed = original > USE_REF_S
    if trimmed:  # a clip of 10-20 s is enough; keep the first USE_REF_S seconds
        wav_bytes = trim_wav(wav_bytes, USE_REF_S)
        validate_reference(wav_bytes)  # the kept part must contain speech
    wav_bytes = normalize_wav(wav_bytes)  # boost a quiet recording before encoding the voice
    duration = min(original, USE_REF_S)
    if shutil.disk_usage(s.data_dir if s.data_dir.exists() else s.data_dir.parent).free < 200 * 1024**2:
        raise AppError("disk", "Less than 200 MB of free disk space.")
    s.voice_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(s.voice_dir, 0o700)
    ref_tmp = s.reference_path.with_suffix(".tmp.wav")
    ref_tmp.write_bytes(wav_bytes)
    os.chmod(ref_tmp, 0o600)
    with _lock:
        model = _get_model()
        try:
            from pocket_tts import export_model_state

            t = time.perf_counter()
            state = model.get_state_for_audio_prompt(str(ref_tmp))
            last_timings["voice_state_s"] = time.perf_counter() - t
            prof_tmp = s.profile_path.with_suffix(".tmp")
            export_model_state(state, str(prof_tmp))
        except AppError:
            ref_tmp.unlink(missing_ok=True)
            raise
        except Exception as e:
            ref_tmp.unlink(missing_ok=True)
            raise AppError("tts", f"Voice profile creation failed: {e}") from e
        os.replace(ref_tmp, s.reference_path)
        os.replace(prof_tmp, s.profile_path)
        os.chmod(s.profile_path, 0o600)
        _state, _state_mtime = state, s.profile_path.stat().st_mtime
    return {"duration_s": duration, "original_s": original, "trimmed": trimmed,
            "voice_state_s": last_timings["voice_state_s"]}


def _load_state():
    """Load the saved voice state from disk (cached; reloaded if the file changes)."""
    global _state, _state_mtime
    s = get_settings()
    if not s.profile_path.exists():
        raise AppError("no_profile", "No voice profile yet. Create one on the Voice Setup page.")
    mtime = s.profile_path.stat().st_mtime
    if _state is None or _state_mtime != mtime:
        model = _get_model()
        t = time.perf_counter()
        try:
            _state = model.get_state_for_audio_prompt(str(s.profile_path))
        except Exception as e:
            raise AppError("tts", f"Could not load the saved voice profile (re-create it): {e}") from e
        _state_mtime = mtime
        last_timings["profile_load_s"] = time.perf_counter() - t
    return _state


def synthesize(text: str) -> tuple[bytes, float]:
    """Speak text with the saved profile. Returns (wav_bytes, audio_duration_s)."""
    text = (text or "").strip()
    if not text:
        raise AppError("tts", "Nothing to speak.")
    with _lock:
        state = _load_state()
        model = _get_model()
        try:
            audio = model.generate_audio(state, text[:MAX_TEXT_CHARS])
        except Exception as e:
            raise AppError("tts", f"Speech generation failed: {e}") from e
        samples = np.asarray(audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio, dtype=np.float32).reshape(-1)
        rate = int(model.sample_rate)
    return float_to_wav(samples, rate), len(samples) / rate


def delete_profile() -> None:
    """Delete the reference recording and the saved voice state."""
    global _state, _state_mtime
    s = get_settings()
    with _lock:
        for p in (s.reference_path, s.profile_path):
            p.unlink(missing_ok=True)
        _state, _state_mtime = None, None
