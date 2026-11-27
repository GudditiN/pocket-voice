"""How close does generated speech sound to the original recording? A light estimate in plain numpy.

It compares two things that stay roughly the same whatever words are said:
  * tone: the average shape of the sound spectrum (mel cepstrum), which reflects the mouth and throat
  * pitch: the typical height of the voice
It does not judge words, speaking speed or naturalness, and it is not a speaker-verification model. Use it to rank
versions of one voice, together with your ears. The 0-100 scale was set using computer-made system voices.
"""
from __future__ import annotations

import numpy as np

from audio import _mono_samples, reduce_noise, wav_info
from errors import AppError

BANDS, COEFFS = 32, 20
LOW_HZ, HIGH_HZ = 80, 5000  # above this, microphones and hiss differ more than voices do
MIN_SPEECH_S = 1.0
# measured difference -> points (interpolated)
TONE_SCALE = ([0.20, 0.30, 0.45, 0.60, 0.85, 1.30], [100, 88, 65, 45, 20, 0])
PITCH_SCALE = ([0.5, 1.0, 2.0, 3.0, 5.0, 8.0], [100, 90, 70, 45, 15, 0])  # difference in semitones
TONE_SHARE = 0.7


def _frames(x: np.ndarray, rate: int) -> np.ndarray:
    """The 32 ms pieces, every 10 ms, that are loud enough to be speech."""
    n, hop = int(rate * 0.032), int(rate * 0.010)
    if x.size < n:
        return np.empty((0, n))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    level = np.sqrt((frames**2).mean(axis=1))
    return frames[level > max(np.percentile(level, 95) * 0.25, 1e-4)]


def _filter_bank(rate: int, nfft: int) -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    edges = 700 * (10 ** (np.linspace(mel(LOW_HZ), mel(min(HIGH_HZ, rate / 2)), BANDS + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    return np.array([np.clip(np.minimum((freqs - a) / (b - a), (c - freqs) / (c - b)), 0, None)
                     for a, b, c in zip(edges, edges[1:], edges[2:])])


def _pitch(power: np.ndarray, rate: int, frame: int) -> float | None:
    """Typical pitch in Hz, from how each voiced piece repeats itself."""
    repeats = np.fft.irfft(power)[:, :frame]
    repeats = repeats / (repeats[:, :1] + 1e-12)
    low, high = int(rate / 400), min(int(rate / 60), frame - 1)  # voices: 60 to 400 Hz
    window = repeats[:, low:high]
    best = window.max(axis=1)
    voiced = best > 0.45
    if voiced.sum() < 5:
        return None
    first = np.argmax(window >= 0.9 * best[:, None], axis=1)  # shortest good repeat: avoids octave mistakes
    step = int(rate / 1000) + 1
    lags = [low + i + int(np.argmax(row[i : i + step])) for row, i in zip(window[voiced], first[voiced])]
    return float(rate / np.median(lags))


def profile(wav: bytes) -> dict:
    """What is measured about one recording. Raises AppError('bad_audio') if there is too little speech."""
    wav_info(wav)
    # the same cleanup for both sides, so background noise counts for less than the voice
    x, rate = _mono_samples(reduce_noise(wav, "medium"))
    frames = _frames(x, rate)
    if len(frames) * 0.010 < MIN_SPEECH_S:
        raise AppError("bad_audio", "Not enough speech to compare. Use a longer sentence.")
    nfft = 1 << int(np.ceil(np.log2(frames.shape[1] * 2)))
    power = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), nfft)) ** 2
    loudness = np.log(power @ _filter_bank(rate, nfft).T + 1e-8)
    k = np.arange(1, COEFFS + 1)[:, None]  # skipping coefficient 0 makes the result ignore volume
    cepstrum = loudness @ (np.sqrt(2 / BANDS) * np.cos(np.pi * k * (np.arange(BANDS) + 0.5) / BANDS)).T
    return {"mean": cepstrum.mean(axis=0), "spread": cepstrum.std(axis=0) + 1e-6,
            "pitch_hz": _pitch(power, rate, frames.shape[1])}


def compare(original: dict, other: dict) -> dict:
    """Score `other` against `original` (both from profile()). Returns 0-100 points and the pitch difference."""
    gap = float(np.sqrt(np.mean((original["mean"] - other["mean"]) ** 2 / (original["spread"] * other["spread"]))))
    tone = float(np.interp(gap, *TONE_SCALE))
    semitones, pitch = None, None
    if original["pitch_hz"] and other["pitch_hz"]:
        semitones = float(12 * np.log2(other["pitch_hz"] / original["pitch_hz"]))
        pitch = float(np.interp(abs(semitones), *PITCH_SCALE))
    score = tone if pitch is None else TONE_SHARE * tone + (1 - TONE_SHARE) * pitch
    verdict = ("Very close" if score >= 80 else "Close" if score >= 60 else "Somewhat alike" if score >= 40
               else "Different")
    return {"score": round(score), "verdict": verdict, "tone": round(tone),
            "pitch": None if pitch is None else round(pitch), "semitones": semitones}
