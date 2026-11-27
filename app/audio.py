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


def decode_to_wav(data: bytes) -> bytes:
    """Return 16-bit PCM WAV bytes for an uploaded WAV, MP3, FLAC or OGG file (no ffmpeg needed).

    16-bit PCM WAV passes through untouched; anything else is decoded with soundfile, mixed to mono.
    """
    try:
        wav_info(data)
        return data
    except AppError:
        pass
    try:
        import soundfile as sf
    except ImportError:
        raise AppError("bad_audio", "Only 16-bit PCM WAV is supported here (install soundfile for MP3/FLAC/OGG).") from None
    try:
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise AppError("bad_audio", "Could not read this audio file. Supported: WAV, MP3, FLAC, OGG "
                                    "(not M4A/AAC: convert it to MP3 or WAV first).") from None
    if x.size == 0:
        raise AppError("bad_audio", "The audio file contains no audio.")
    return float_to_wav(x.mean(axis=1), int(rate))


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


# level -> (how strongly the noise estimate is subtracted, maximum attenuation in dB)
DENOISE_LEVELS = {"light": (1.5, 10.0), "medium": (2.0, 16.0), "strong": (3.0, 24.0)}


def _mono_samples(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        p = w.getparams()
        frames = w.readframes(p.nframes)
    x = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if p.nchannels > 1:
        x = x[: len(x) // p.nchannels * p.nchannels].reshape(-1, p.nchannels).mean(axis=1)
    return x, p.framerate


RUMBLE_HZ = 60  # below any speaking voice: handling bumps, mains hum, air-conditioner rumble


def background_level(data: bytes) -> float:
    """Level in dBFS of the quieter parts (10th percentile of 20 ms windows): roughly the background noise."""
    wav_info(data)
    x, rate = _mono_samples(data)
    win = max(1, int(rate * 0.02))
    n = x.size // win
    if not n:
        return to_dbfs(0.0)
    levels = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    return to_dbfs(float(np.percentile(levels, 10)))


def _smooth(a: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Moving average of odd length `size` along `axis`, keeping the shape."""
    pad = [(size // 2, size // 2) if i == axis else (0, 0) for i in range(a.ndim)]
    padded = np.pad(a, pad, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis).mean(axis=-1)


def reduce_noise(data: bytes, level: str = "medium") -> bytes:
    """Reduce steady background noise (fan, hiss, hum) by spectral gating. Returns mono 16-bit WAV.

    The noise spectrum is estimated from the quietest tenth of the recording, so it works best when there are
    short pauses between words. It does not remove other voices, TV or sudden sounds.
    """
    if level not in DENOISE_LEVELS:
        raise ValueError(f"Unknown noise reduction level: {level!r}")
    wav_info(data)  # validates the WAV
    x, rate = _mono_samples(data)
    n = 2 ** int(round(np.log2(rate * 0.032)))  # about 32 ms frames
    hop = n // 4
    if x.size < n * 8:
        return data
    over, max_cut_db = DENOISE_LEVELS[level]
    window = np.hanning(n + 1)[:-1]
    padded = np.pad(x, (n, n + (-x.size) % hop), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n)[::hop] * window
    spectrum = np.fft.rfft(frames, axis=1)
    power = np.abs(spectrum) ** 2

    energy = power.sum(axis=1)
    audible = np.flatnonzero(energy > 1e-7)  # ignore digital silence, which would hide the real noise
    if audible.size < 8:
        return data
    quiet = audible[np.argsort(energy[audible])[: max(4, audible.size // 10)]]
    noise = power[quiet].mean(axis=0)

    smoothed = _smooth(_smooth(power, 5, axis=0), 3, axis=1)
    gain = np.sqrt(np.clip(1.0 - over * noise / (smoothed + 1e-12), 0.0, 1.0))
    gain = np.maximum(gain, 10 ** (-max_cut_db / 20))
    gain = _smooth(_smooth(gain, 3, axis=0), 3, axis=1)  # avoids "musical" artefacts
    gain[:, np.fft.rfftfreq(n, 1 / rate) < RUMBLE_HZ] = 0.0

    cleaned = np.fft.irfft(spectrum * gain, n=n, axis=1) * window
    out = np.zeros(padded.size)
    weight = np.zeros(padded.size)
    for i, frame in enumerate(cleaned):
        start = i * hop
        out[start : start + n] += frame
        weight[start : start + n] += window**2
    y = (out / np.maximum(weight, 1e-8))[n : n + x.size]
    return float_to_wav(y.astype(np.float32), rate)


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
