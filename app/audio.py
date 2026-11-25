"""WAV helpers using only the stdlib + numpy (no ffmpeg / soundfile needed)."""
from __future__ import annotations

import io
import wave

import numpy as np

from errors import AppError

SILENCE_RMS = 0.001  # -60 dBFS: below this the recording is treated as silent
TARGET_DBFS = -20.0  # quiet recordings are boosted so speech sits around here
MAX_GAIN_DB = 40.0


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
    # Speech level = 90th percentile of 20 ms window RMS values, so pauses don't drag it down.
    win = max(1, int(rate * 0.02))
    nw = samples.size // win
    if nw:
        levels = np.sqrt((samples[: nw * win].reshape(nw, win) ** 2).mean(axis=1))
        rms = float(np.percentile(levels, 90))
    else:
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    return n / rate, rms


def to_dbfs(level: float) -> float:
    """Linear level (0..1) to dBFS; silence maps to -100."""
    return 20 * float(np.log10(max(level, 1e-5)))


def normalize_wav(data: bytes, target_dbfs: float = TARGET_DBFS) -> bytes:
    """Boost quiet 16-bit PCM audio so speech sits near target_dbfs.

    Never attenuates, caps the gain at MAX_GAIN_DB and never clips. Returns the input unchanged
    when it is already loud enough.
    """
    _, level = wav_info(data)  # also validates the WAV
    gain_db = min(target_dbfs - to_dbfs(level), MAX_GAIN_DB)
    if gain_db <= 0.5:
        return data
    with wave.open(io.BytesIO(data), "rb") as w:
        p = w.getparams()
        frames = w.readframes(p.nframes)
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    peak = float(np.abs(x).max())
    gain = 10 ** (gain_db / 20)
    if peak > 0:
        gain = min(gain, 32000.0 / peak)
    if gain <= 1.05:
        return data
    y = np.clip(x * gain, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(p.nchannels)
        out.setsampwidth(p.sampwidth)
        out.setframerate(p.framerate)
        out.writeframes(y.tobytes())
    return buf.getvalue()


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
