"""PocketTTS wrapper: one cached model, named reusable voice profiles, tunable generation, sequential synthesis.

A "voice profile" is reference-audio conditioning (a saved voice state), NOT fine-tuning.
pocket_tts is imported lazily so the rest of the app (and unit tests) work without it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import threading
import time

import numpy as np

from audio import SILENCE_RMS, float_to_wav, normalize_wav, reduce_noise, to_dbfs, trim_wav, wav_info
from config import get_settings
from errors import AppError

log = logging.getLogger(__name__)

MODEL_NAME = "pocket-tts"
MIN_REF_S = 5.0  # shorter recordings are rejected
USE_REF_S = 20.0  # longer recordings are trimmed to their first USE_REF_S seconds
MAX_INPUT_S = 600.0  # recordings longer than this are rejected outright
MAX_TEXT_CHARS = 800

# Generation settings. Defaults are PocketTTS 4.0's own (temp 0.3, 1 sampler step, eos threshold -4.0).
# They are plain attributes on the model that are read at generation time, so changing them needs no reload.
DEFAULT_TUNING = {"temp": 0.3, "steps": 1, "eos_threshold": -4.0}
_LIMITS = {"temp": (0.05, 1.5), "steps": (1, 8), "eos_threshold": (-10.0, 5.0)}

_lock = threading.Lock()  # PocketTTS is not assumed thread-safe: one synthesis at a time
_model = None
_states: dict[str, tuple[float, object]] = {}  # profile name -> (file mtime, loaded voice state)
last_timings: dict[str, float] = {}


# ---------------------------------------------------------------- recording validation
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


def _explain(prefix: str, e: Exception) -> AppError:
    """Turn a PocketTTS exception into an AppError, with setup steps for the gated voice-cloning weights."""
    text = str(e)
    if "voice cloning" in text.lower() and ("huggingface.co" in text or "hf auth" in text):
        return AppError(
            "tts",
            "Voice cloning needs access to a gated Hugging Face model. (1) Open https://huggingface.co/kyutai/pocket-tts "
            "while logged in and accept the terms. (2) Create a read token at https://huggingface.co/settings/tokens. "
            "(3) Provide it as the HF_TOKEN environment variable or Codespaces secret (or run `uv run hf auth login`), "
            "then restart the app.")
    return AppError("tts", f"{prefix}: {text}")


def _get_model():
    global _model
    if _model is None:
        t = time.perf_counter()
        try:
            from pocket_tts import TTSModel

            _model = TTSModel.load_model(quantize=get_settings().quantize)
        except ImportError:
            raise AppError("tts", "pocket-tts is not installed. Run `uv sync`.") from None
        except Exception as e:  # download failures, auth errors, out of disk, ...
            raise _explain("Could not load the PocketTTS model", e) from e
        last_timings["model_load_s"] = time.perf_counter() - t
        log.info("PocketTTS loaded in %.1fs", last_timings["model_load_s"])
    return _model


# ---------------------------------------------------------------- tuning
def clean_tuning(d: dict | None) -> dict:
    """Fill defaults and clamp every value into its allowed range."""
    out = dict(DEFAULT_TUNING)
    for key, (lo, hi) in _LIMITS.items():
        try:
            out[key] = min(max(float((d or {}).get(key, out[key])), lo), hi)
        except (TypeError, ValueError):
            pass
    out["steps"] = int(round(out["steps"]))
    return out


ENV_TUNING = {"temp": "POCKET_TEMP", "steps": "POCKET_STEPS", "eos_threshold": "POCKET_EOS_THRESHOLD"}


def get_tuning() -> dict:
    """Settings used for speech. Per setting: environment / .env wins, then the saved file, then the default."""
    try:
        values = json.loads(get_settings().tuning_path.read_text())
    except (OSError, ValueError):
        values = {}
    if not isinstance(values, dict):
        values = {}
    for key, var in ENV_TUNING.items():
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        try:
            number = float(raw)
        except ValueError:
            number = float("nan")
        if math.isfinite(number):
            values[key] = number
        else:
            log.warning("Ignoring %s: not a number", var)
    return clean_tuning(values)


def save_tuning(d: dict) -> dict:
    s = get_settings()
    cleaned = clean_tuning(d)
    s.voice_dir.mkdir(parents=True, exist_ok=True)
    tmp = s.tuning_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cleaned))
    os.replace(tmp, s.tuning_path)
    return cleaned


def reset_tuning() -> None:
    get_settings().tuning_path.unlink(missing_ok=True)


# ---------------------------------------------------------------- profiles
def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (name or "").lower().strip()).strip("-")[:32]
    if not s:
        raise AppError("bad_audio", "Give the profile a name using letters or numbers.")
    return s


def _paths(name: str):
    d = get_settings().profiles_dir / name
    return d, d / "reference.wav", d / "profile.safetensors"


def _migrate_legacy() -> None:
    """Move the single profile of earlier versions into profiles/default/."""
    s = get_settings()
    if s.legacy_profile_path.exists() and not (s.profiles_dir / "default").exists():
        d, ref, prof = _paths("default")
        d.mkdir(parents=True, exist_ok=True)
        os.replace(s.legacy_profile_path, prof)
        if s.legacy_reference_path.exists():
            os.replace(s.legacy_reference_path, ref)
        if not s.active_file.exists():
            s.active_file.write_text("default")


def list_profiles() -> list[dict]:
    _migrate_legacy()
    root = get_settings().profiles_dir
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            prof = d / "profile.safetensors"
            if d.is_dir() and prof.exists():
                out.append({"name": d.name, "created": prof.stat().st_mtime, "kb": prof.stat().st_size / 1024,
                            "has_reference": (d / "reference.wav").exists()})
    return out


def active_profile() -> str | None:
    names = [p["name"] for p in list_profiles()]
    try:
        chosen = get_settings().active_file.read_text().strip()
    except OSError:
        chosen = ""
    return chosen if chosen in names else (names[0] if names else None)


def set_active(name: str) -> None:
    if name not in [p["name"] for p in list_profiles()]:
        raise AppError("no_profile", f"There is no profile named {name!r}.")
    s = get_settings()
    s.voice_dir.mkdir(parents=True, exist_ok=True)
    s.active_file.write_text(name)


def profile_exists(name: str | None = None) -> bool:
    name = name or active_profile()
    return bool(name) and _paths(name)[2].exists()


def profile_info(name: str | None = None) -> dict | None:
    name = name or active_profile()
    if not name or not _paths(name)[2].exists():
        return None
    _, ref, prof = _paths(name)
    return {"name": name, "created": prof.stat().st_mtime, "profile_kb": prof.stat().st_size / 1024,
            "has_reference": ref.exists()}


def prepare_reference(wav_bytes: bytes, denoise: str = "off") -> tuple[bytes, dict]:
    """The exact audio a voice is built from: validated, trimmed, optionally de-noised, level-boosted."""
    original = validate_reference(wav_bytes)
    trimmed = original > USE_REF_S
    if trimmed:  # a clip of 10-20 s is enough; keep the first USE_REF_S seconds
        wav_bytes = trim_wav(wav_bytes, USE_REF_S)
        validate_reference(wav_bytes)  # the kept part must contain speech
    if denoise != "off":
        wav_bytes = reduce_noise(wav_bytes, denoise)
        validate_reference(wav_bytes)  # cleaning must not leave silence
    wav_bytes = normalize_wav(wav_bytes)  # boost a quiet recording before encoding the voice
    return wav_bytes, {"duration_s": min(original, USE_REF_S), "original_s": original, "trimmed": trimmed,
                       "denoise": denoise}


def build_profile(wav_bytes: bytes, name: str = "default", make_active: bool = True,
                  denoise: str | None = None) -> dict:
    """Validate the recording, store it privately, build the voice state and save it as a named profile.

    denoise: off, light, medium or strong (default: the POCKET_DENOISE setting).
    """
    s = get_settings()
    name = slug(name)
    wav_bytes, details = prepare_reference(wav_bytes, s.denoise if denoise is None else denoise)
    if shutil.disk_usage(s.data_dir if s.data_dir.exists() else s.data_dir.parent).free < 200 * 1024**2:
        raise AppError("disk", "Less than 200 MB of free disk space.")
    d, ref, prof = _paths(name)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(s.voice_dir, 0o700)
    ref_tmp = ref.with_suffix(".tmp.wav")
    prof_tmp = prof.with_suffix(".tmp")
    ref_tmp.write_bytes(wav_bytes)
    os.chmod(ref_tmp, 0o600)
    with _lock:
        model = _get_model()
        try:
            from pocket_tts import export_model_state

            t = time.perf_counter()
            state = model.get_state_for_audio_prompt(str(ref_tmp))
            last_timings["voice_state_s"] = time.perf_counter() - t
            export_model_state(state, str(prof_tmp))
        except AppError:
            ref_tmp.unlink(missing_ok=True)
            raise
        except Exception as e:
            ref_tmp.unlink(missing_ok=True)
            prof_tmp.unlink(missing_ok=True)
            raise _explain("Voice profile creation failed", e) from e
        os.replace(ref_tmp, ref)
        os.replace(prof_tmp, prof)
        shutil.rmtree(d / "fillers", ignore_errors=True)  # phrases made with the previous recording
        os.chmod(prof, 0o600)
        _states[name] = (prof.stat().st_mtime, state)
    if make_active:
        set_active(name)
    return {"name": name, **details, "voice_state_s": last_timings["voice_state_s"]}


def _load_state(name: str):
    """Load a saved voice state from disk (cached; reloaded if the file changes)."""
    prof = _paths(name)[2]
    if not prof.exists():
        raise AppError("no_profile", "No voice profile yet. Create one on the Voice Setup page.")
    mtime = prof.stat().st_mtime
    cached = _states.get(name)
    if cached is None or cached[0] != mtime:
        model = _get_model()
        t = time.perf_counter()
        try:
            state = model.get_state_for_audio_prompt(str(prof))
        except Exception as e:
            raise AppError("tts", f"Could not load the saved voice profile (re-create it): {e}") from e
        _states[name] = (mtime, state)
        last_timings["profile_load_s"] = time.perf_counter() - t
    return _states[name][1]


def delete_profile(name: str | None = None) -> None:
    """Delete a profile's reference recording and saved voice state (the active one by default)."""
    name = name or active_profile()
    if not name:
        return
    d = _paths(name)[0]
    with _lock:
        shutil.rmtree(d, ignore_errors=True)
        _states.pop(name, None)


# ---------------------------------------------------------------- synthesis
def synthesize_stream(text: str, profile: str | None = None, tuning: dict | None = None,
                      first_chunk_s: float = 0.4, chunk_s: float = 0.8):
    """Speak text piece by piece. Yields (samples float32, sample_rate) while the model is still generating."""
    text = (text or "").strip()
    if not text:
        raise AppError("tts", "Nothing to speak.")
    t = clean_tuning(tuning) if tuning is not None else get_tuning()
    name = profile or active_profile()
    if not name:
        raise AppError("no_profile", "No voice profile yet. Create one on the Voice Setup page.")
    with _lock:
        state = _load_state(name)
        model = _get_model()
        model.temp, model.sampler_decode_steps, model.eos_threshold = t["temp"], t["steps"], t["eos_threshold"]
        rate = int(model.sample_rate)
        held, size, wanted = [], 0, first_chunk_s * rate
        try:
            for piece in model.generate_audio_stream(state, text[:MAX_TEXT_CHARS]):
                samples = np.asarray(piece.detach().cpu().numpy() if hasattr(piece, "detach") else piece,
                                     dtype=np.float32).reshape(-1)
                held.append(samples)
                size += samples.size
                if size >= wanted:
                    yield np.concatenate(held), rate
                    held, size, wanted = [], 0, chunk_s * rate
        except Exception as e:
            raise AppError("tts", f"Speech generation failed: {e}") from e
        if held:
            yield np.concatenate(held), rate


# ---------------------------------------------------------------- fillers and warm-up
FILLERS = {"think": ["Hmm, let me think.", "Okay, one second.", "Right, so..."],
           "search": ["Let me check that for you.", "One moment, I'm looking it up."]}


def _filler_path(name: str, text: str):
    key = hashlib.sha1(f"{text}|{json.dumps(get_tuning(), sort_keys=True)}".encode()).hexdigest()[:16]
    return _paths(name)[0] / "fillers" / f"{key}.wav"


def filler(kind: str, profile: str | None = None) -> bytes | None:
    """A ready-made short phrase in this voice, or None. Never generates (so it never adds delay)."""
    name = profile or active_profile()
    if not name:
        return None
    ready = [p for p in (_filler_path(name, t) for t in FILLERS.get(kind, [])) if p.exists()]
    return random.choice(ready).read_bytes() if ready else None


def make_fillers(profile: str | None = None) -> int:
    """Generate any missing filler phrases for a voice. Returns how many were created."""
    name = profile or active_profile()
    if not name or not profile_exists(name):
        return 0
    made = 0
    for text in (t for phrases in FILLERS.values() for t in phrases):
        path = _filler_path(name, text)
        if path.exists():
            continue
        wav, _ = synthesize(text, profile=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav)
        os.chmod(path, 0o600)
        made += 1
    return made


def warm_up() -> None:
    """Load the model and the default voice ahead of the first question (run in the background at startup)."""
    try:
        name = active_profile()
        if not name:
            return
        with _lock:
            _load_state(name)
        if get_settings().fillers:
            make_fillers(name)
    except Exception as e:  # the first real request will report any problem to the user
        log.warning("voice warm-up skipped: %s", e)


def synthesize(text: str, profile: str | None = None, tuning: dict | None = None) -> tuple[bytes, float]:
    """Speak text. Returns (wav_bytes, audio_duration_s).

    profile: profile name (default: the active one). tuning: override of the saved generation settings.
    """
    text = (text or "").strip()
    if not text:
        raise AppError("tts", "Nothing to speak.")
    t = clean_tuning(tuning) if tuning is not None else get_tuning()
    name = profile or active_profile()
    if not name:
        raise AppError("no_profile", "No voice profile yet. Create one on the Voice Setup page.")
    with _lock:
        state = _load_state(name)
        model = _get_model()
        model.temp = t["temp"]
        model.sampler_decode_steps = t["steps"]
        model.eos_threshold = t["eos_threshold"]
        try:
            audio = model.generate_audio(state, text[:MAX_TEXT_CHARS])
        except Exception as e:
            raise AppError("tts", f"Speech generation failed: {e}") from e
        samples = np.asarray(audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio, dtype=np.float32).reshape(-1)
        rate = int(model.sample_rate)
    return float_to_wav(samples, rate), len(samples) / rate
