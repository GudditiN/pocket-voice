"""The voice match estimate, on made-up "voices": harmonics shaped by resonances that stand in for a speaker."""
import numpy as np
import pytest

import effects
import similarity
import tts
from audio import float_to_wav
from errors import AppError

RATE = 24000
VOWELS = [(1.0, 1.0), (1.35, 0.8), (0.75, 1.2), (1.1, 1.35), (0.85, 0.75)]  # shifts of the two main resonances


def voice(pitch=120.0, resonances=(600, 1500, 2600), seed=1, seconds=6.0, amp=0.3, noise=0.0005):
    """A speaker is a pitch and three resonances; the seed picks which "words" (vowel order) are said."""
    rng, pieces = np.random.default_rng(seed), []
    for _ in range(int(seconds / 0.3)):
        a, b = VOWELS[rng.integers(len(VOWELS))]
        centres = (resonances[0] * a, resonances[1] * b, resonances[2])
        f0 = pitch * rng.uniform(0.97, 1.03)
        t = np.arange(int(0.25 * RATE)) / RATE
        tone = sum(np.sin(2 * np.pi * f0 * k * t) * sum(1 / (1 + ((f0 * k - c) / 120) ** 2) for c in centres)
                   for k in range(1, int(5000 / f0)))
        pieces += [tone / np.abs(tone).max() * np.hanning(t.size) ** 0.3, np.zeros(int(0.05 * RATE))]
    x = amp * np.concatenate(pieces) + noise * rng.standard_normal(sum(p.size for p in pieces))
    return float_to_wav(x.astype(np.float32), RATE)


def match(a, b):
    return similarity.compare(similarity.profile(a), similarity.profile(b))


def test_the_same_recording_matches_fully():
    result = match(voice(), voice())
    assert result["score"] == 100 and result["verdict"] == "Very close" and abs(result["semitones"]) < 0.05


def test_same_voice_saying_other_words_beats_another_voice():
    mine = voice(seed=1)
    same = match(mine, voice(seed=7))
    other = match(mine, voice(pitch=210, resonances=(800, 2100, 3300), seed=7))
    assert same["score"] >= 75 and other["score"] < 45


def test_pitch_is_measured_and_lowers_the_score():
    mine = voice()
    higher = effects.process_wav(voice(seed=3), {**effects.NEUTRAL, "pitch": 3.0})[0]
    result = match(mine, higher)
    assert result["semitones"] == pytest.approx(3.0, abs=0.6) and result["score"] < match(mine, voice(seed=3))["score"]


def test_volume_does_not_matter():
    assert match(voice(), voice(amp=0.05, noise=0.00008))["score"] >= 90


def test_silence_cannot_be_compared():
    with pytest.raises(AppError):
        similarity.profile(float_to_wav(np.zeros(RATE * 3, dtype=np.float32), RATE))


def test_original_recording_is_available_only_for_real_voices():
    from config import get_settings

    assert tts.reference_wav("missing") is None
    d = get_settings().profiles_dir / "a"
    d.mkdir(parents=True)
    (d / "reference.wav").write_bytes(b"RIFF")
    assert tts.reference_wav("a") == b"RIFF"
