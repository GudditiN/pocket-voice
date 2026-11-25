"""WAV helpers using only the stdlib + numpy (no ffmpeg / soundfile needed)."""
from __future__ import annotations

import io
import wave

import numpy as np

from errors import AppError

SILENCE_RMS = 0.003  # ~ -50 dBFS


def wav_info(data: bytes) -> tuple[float, float]:
    """Return (duration_s, rms 0..1) for 16-bit PCM WAV bytes. Raises AppError('bad_audio')."""
    if not data:
        raise AppError("bad_audio", "The recording is empty.")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if width != 2:
                raise AppError("bad_audio", "Unsupported WAV encoding. Use 16-bit PCM WAV.")
            frames = w.readframes(n)
    except (wave.Error, EOFError):
        raise AppError("bad_audio", "Not a valid WAV file. Only PCM WAV is supported.") from None
    if rate <= 0 or n == 0:
        raise AppError("bad_audio", "The recording contains no audio.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        samples = samples[: len(samples) // ch * ch].reshape(-1, ch).mean(axis=1)
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    return n / rate, rms


def trim_wav(data: bytes, seconds: float) -> bytes:
    """Return the first `seconds` of a PCM WAV as a new WAV (unchanged if already shorter)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        p = w.getparams()
        frames = w.readframes(min(p.nframes, int(seconds * p.framerate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(p.nchannels)
        out.setsampwidth(p.sampwidth)
        out.setframerate(p.framerate)
        out.writeframes(frames)
    return buf.getvalue()


def float_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 [-1, 1] -> 16-bit PCM WAV bytes."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
