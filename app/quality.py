"""How good is a recording for copying a voice, and trimming the silence around it.

Plain Python (no numpy), so the rules are easy to read and test.
"""
from __future__ import annotations

import io
import math
import sys
import wave
from array import array

from errors import AppError

FRAME_S = 0.02
CHECK_S = 60.0  # only the first minute is scored
IDEAL_SPEECH_S = 12.0


def _read(data: bytes):
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
    except (wave.Error, EOFError):
        raise AppError("bad_audio", "Not a valid WAV file. Only PCM WAV is supported.") from None
    if params.sampwidth != 2 or not frames or params.framerate <= 0:
        raise AppError("bad_audio", "Use a 16-bit PCM WAV recording.")
    return params, frames


def _mono(frames: bytes, channels: int) -> array:
    samples = array("h")
    samples.frombytes(frames[: len(frames) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if channels > 1:
        usable = len(samples) // channels * channels
        samples = array("h", (sum(samples[i : i + channels]) // channels for i in range(0, usable, channels)))
    return samples


def _levels(samples: array, size: int) -> list[float]:
    """Loudness (0..1) of each short frame."""
    out = []
    for i in range(0, len(samples) - size + 1, size):
        frame = samples[i : i + size]
        out.append(math.sqrt(math.sumprod(frame, frame) / size) / 32768.0)
    return out or [0.0]


def _db(level: float) -> float:
    return 20 * math.log10(max(level, 1e-5))


def _speech_and_noise(levels: list[float]) -> tuple[float, float, float]:
    """(speech level, background level, the level above which a frame counts as speech)."""
    ordered = sorted(levels)
    speech, noise = ordered[int(0.9 * (len(ordered) - 1))], ordered[int(0.05 * (len(ordered) - 1))]
    return speech, noise, min(max(noise * 3, speech * 0.08), speech * 0.5)


def check(data: bytes) -> dict:
    """Score a recording from 0 to 100 and say what to improve."""
    params, frames = _read(data)
    samples = _mono(frames[: int(CHECK_S * params.framerate) * params.sampwidth * params.nchannels], params.nchannels)
    size = max(1, int(params.framerate * FRAME_S))
    levels = _levels(samples, size)
    speech, noise, threshold = _speech_and_noise(levels)
    speech_db, noise_db = _db(speech), _db(noise)
    snr = speech_db - noise_db
    speech_s = sum(level > threshold for level in levels) * size / params.framerate
    clipped = 100 * sum(1 for x in samples if x >= 32700 or x <= -32700) / max(len(samples), 1)

    score, advice = 100.0, []
    if speech_db < -60:
        return {"score": 0, "verdict": "No sound", "speech_s": 0.0, "speech_db": speech_db, "noise_db": noise_db,
                "snr_db": 0.0, "clipped_pct": 0.0, "advice": ["No sound was captured. Check the microphone."]}
    if speech_s < IDEAL_SPEECH_S:
        score -= min(40, (IDEAL_SPEECH_S - speech_s) * 4)
        advice.append(f"Speak longer: about {speech_s:.0f} s of speech was heard, aim for 12 to 20 s.")
    if speech_db < -35:
        score -= 30 if speech_db < -45 else 15
        advice.append("Too quiet: move closer to the microphone or raise its input level.")
    if snr < 30:
        score -= min(50, (30 - snr) * 2)
        advice.append("Background noise: turn off fans and TV, or choose noise reduction below. "
                      "(Leave a second of silence at the start so noise can be measured.)")
    if clipped > 0.1:
        score -= 30
        advice.append("Distorted: you are too loud or too close. Move back a little.")
    score = max(0, round(score))
    verdict = "Good" if score >= 75 else "Usable" if score >= 50 else "Record again"
    return {"score": score, "verdict": verdict, "speech_s": speech_s, "speech_db": speech_db, "noise_db": noise_db,
            "snr_db": snr, "clipped_pct": clipped, "advice": advice}


def trim_silence(data: bytes, margin_s: float = 0.15) -> bytes:
    """Cut the silence before the first and after the last words, so the part used for the voice is all speech."""
    params, frames = _read(data)
    samples = _mono(frames, params.nchannels)
    size = max(1, int(params.framerate * FRAME_S))
    levels = _levels(samples, size)
    _, _, threshold = _speech_and_noise(levels)
    spoken = [i for i, level in enumerate(levels) if level > threshold]
    if not spoken:
        return data
    margin = int(margin_s * params.framerate)
    start = max(0, spoken[0] * size - margin)
    end = min(len(samples), (spoken[-1] + 1) * size + margin)
    if (len(samples) - (end - start)) / params.framerate < 0.3:
        return data
    width = params.sampwidth * params.nchannels
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(frames[start * width : end * width])
    return out.getvalue()
