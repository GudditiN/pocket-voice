"""Pitch, speed and noise cleanup on made-up audio. No model."""
import numpy as np
import pytest

import effects
import tts
from audio import background_level, float_to_wav, wav_info

RATE = 24000


def tone(seconds=3.0, hz=220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def loudest_hz(x):
    middle = x[x.size // 4 : -x.size // 4]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(middle.size)))
    return np.fft.rfftfreq(middle.size, 1 / RATE)[spectrum.argmax()]


def settings(**change):
    return {**effects.NEUTRAL, **change}


def test_neutral_settings_change_nothing():
    x = tone()
    assert not effects.active(settings()) and np.array_equal(effects.apply(x, RATE, settings()), x)


@pytest.mark.parametrize("semitones", [-3.0, -1.5, 1.5, 3.0])
def test_pitch_moves_but_length_stays(semitones):
    out = effects.apply(tone(), RATE, settings(pitch=semitones))
    assert out.size == pytest.approx(3 * RATE, rel=0.01)
    assert loudest_hz(out) == pytest.approx(220 * 2 ** (semitones / 12), rel=0.02)
    assert np.sqrt(np.mean(out[RATE:-RATE] ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)  # no dropouts


@pytest.mark.parametrize("speed", [0.85, 1.18])
def test_speed_changes_length_but_not_pitch(speed):
    out = effects.apply(tone(), RATE, settings(speed=speed))
    assert out.size == pytest.approx(3 * RATE / speed, rel=0.01)
    assert loudest_hz(out) == pytest.approx(220, rel=0.02)


def test_cleanup_lowers_background_noise():
    t = np.arange(8 * RATE) / RATE
    noisy = (tone(8.0) * ((t % 1.0) < 0.6) + 0.01 * np.random.default_rng(1).standard_normal(t.size)).astype(np.float32)
    wav = float_to_wav(noisy, RATE)
    cleaned, seconds = effects.process_wav(wav, settings(clean=2))
    assert seconds == pytest.approx(8.0, abs=0.01) and wav_info(cleaned)[0] == pytest.approx(8.0, abs=0.01)
    assert background_level(cleaned) < background_level(wav) - 8


def test_very_short_audio_is_safe():
    for length in (1, 50, 1500):
        assert effects.apply(tone()[:length], RATE, settings(pitch=2, speed=1.1, clean=3)).size > 0


class FakeModel:
    sample_rate = RATE

    def generate_audio(self, state, text):
        return tone(1.0)

    def generate_audio_stream(self, state, text):
        yield from np.split(tone(3.0), 30)


def fake_voice(monkeypatch):
    monkeypatch.setattr(tts, "_get_model", FakeModel)
    monkeypatch.setattr(tts, "_load_state", lambda name: "state")


def test_speech_uses_the_effects(monkeypatch):
    fake_voice(monkeypatch)
    _, seconds = tts.synthesize("hi", profile="a", tuning={"speed": 1.25})
    assert seconds == pytest.approx(0.8, abs=0.01)


def test_streaming_shapes_each_sentence_as_a_whole(monkeypatch):
    fake_voice(monkeypatch)
    plain = list(tts.synthesize_stream("hi", profile="a", tuning={}))
    shaped = list(tts.synthesize_stream("hi", profile="a", tuning={"pitch": 2}))
    assert len(plain) > 1 and len(shaped) == 1  # processing small pieces separately would click at the joins
    assert shaped[0][0].size == pytest.approx(3 * RATE, rel=0.01)
