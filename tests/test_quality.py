"""Recording score and silence trimming, on made-up audio: tone bursts stand in for speech."""
import io
import math
import random
import wave
from array import array

import pytest

import quality
from errors import AppError

RATE = 8000


def recording(seconds=18.0, amp=0.3, noise=0.0005, lead=1.0, tail=1.0):
    rng, samples = random.Random(1), array("h")
    for i in range(int((lead + seconds + tail) * RATE)):
        t = i / RATE
        speaking = lead <= t < lead + seconds and (t - lead) % 1.0 < 0.8
        x = (amp * math.sin(2 * math.pi * 220 * t) if speaking else 0.0) + noise * rng.gauss(0, 1)
        samples.append(int(max(-1.0, min(1.0, x)) * 32767))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())
    return out.getvalue()


def seconds_of(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_clean_recording_scores_well():
    report = quality.check(recording())
    assert report["verdict"] == "Good" and report["advice"] == [] and 13 < report["speech_s"] < 16


@pytest.mark.parametrize("kwargs,word", [
    ({"noise": 0.05}, "noise"), ({"amp": 0.008, "noise": 0.00002}, "quiet"), ({"seconds": 6.0}, "longer"),
    ({"amp": 3.0}, "Distorted"),
])
def test_problems_are_explained(kwargs, word):
    report = quality.check(recording(**kwargs))
    assert report["score"] < 100 and any(word in tip for tip in report["advice"])


def test_silence_scores_zero():
    assert quality.check(recording(amp=0.0, noise=0.0))["score"] == 0


def test_silence_around_the_words_is_cut():
    trimmed = quality.trim_silence(recording(seconds=8.0, lead=4.0, tail=3.0))
    assert 7.7 < seconds_of(trimmed) < 8.4


def test_recording_without_silence_is_unchanged():
    data = recording(seconds=8.0, lead=0.0, tail=0.0)
    assert quality.trim_silence(data) == data


def test_not_a_wav():
    with pytest.raises(AppError):
        quality.check(b"hello")
