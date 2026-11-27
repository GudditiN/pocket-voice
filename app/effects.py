"""Shape the generated speech: noise cleanup, pitch and speed. Plain numpy, no extra packages.

These change the finished audio, not the voice model.
"""
from __future__ import annotations

import numpy as np

from audio import _mono_samples, float_to_wav, reduce_noise

CLEAN_LEVELS = ("off", "light", "medium", "strong")
NEUTRAL = {"pitch": 0.0, "speed": 1.0, "clean": 0}

# what the Voice Setup page offers to compare: group -> (setting, [(label, value)])
VARIATIONS = {
    "Pitch": ("pitch", [("Much deeper", -3.0), ("Deeper", -1.5), ("Unchanged", 0.0), ("Higher", 1.5),
                        ("Much higher", 3.0)]),
    "Speed": ("speed", [("Slow", 0.85), ("A bit slower", 0.93), ("Unchanged", 1.0), ("A bit faster", 1.08),
                        ("Fast", 1.18)]),
    "Noise cleanup": ("clean", [("Off", 0), ("Light", 1), ("Medium", 2), ("Strong", 3)]),
}


def active(tuning: dict) -> bool:
    """Would these settings change the audio at all?"""
    return any(tuning.get(key, value) != value for key, value in NEUTRAL.items())


def time_stretch(x: np.ndarray, factor: float, rate: int) -> np.ndarray:
    """Make audio `factor` times as long without changing its pitch (overlap-add of best-matching pieces)."""
    frame = int(rate * 0.04) // 2 * 2
    hop, search = frame // 2, int(rate * 0.01)
    if abs(factor - 1.0) < 1e-3 or x.size < frame * 2:
        return x
    length = int(round(x.size * factor))
    padded = np.pad(x.astype(np.float64), (search, frame * 2 + search))
    window = np.hanning(frame + 1)[:-1]
    out, weight = np.zeros(length + frame), np.zeros(length + frame)
    previous = None
    for start in range(0, length, hop):
        target = min(int(start / factor), x.size - 1) + search
        if previous is None:
            position = target
        else:  # near the target, take the piece that best continues the one before: avoids warbling
            natural = padded[previous + hop : previous + hop + frame]
            region = padded[target - search : target + search + frame]
            position = target - search + int(np.argmax(np.correlate(region, natural, mode="valid")))
        out[start : start + frame] += padded[position : position + frame] * window
        weight[start : start + frame] += window
        previous = min(position, target + search)
    return (out[:length] / np.maximum(weight[:length], 1e-3)).astype(np.float32)


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Read the audio `ratio` times faster: pitch and speed both rise by that factor."""
    positions = np.arange(int(round(x.size / ratio))) * ratio
    return np.interp(positions, np.arange(x.size), x).astype(np.float32)


def apply(samples: np.ndarray, rate: int, tuning: dict) -> np.ndarray:
    """Noise cleanup, then pitch (in semitones) and speed (1.0 = unchanged). Mono float32 in and out."""
    out = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not out.size or not active(tuning):
        return out
    level = CLEAN_LEVELS[int(tuning.get("clean", 0))]
    if level != "off":
        out = _mono_samples(reduce_noise(float_to_wav(out, rate), level))[0].astype(np.float32)
    ratio = 2 ** (float(tuning.get("pitch", 0.0)) / 12)
    speed = float(tuning.get("speed", 1.0))
    if abs(ratio - 1) > 1e-3 or abs(speed - 1) > 1e-3:
        out = time_stretch(out, ratio / speed, rate)
        if abs(ratio - 1) > 1e-3:
            out = resample(out, ratio)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def process_wav(wav: bytes, tuning: dict) -> tuple[bytes, float]:
    """apply() for a WAV file. Returns (wav_bytes, duration_s)."""
    samples, rate = _mono_samples(wav)
    out = apply(samples, rate, tuning)
    return float_to_wav(out, rate), out.size / rate
