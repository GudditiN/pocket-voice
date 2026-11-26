"""Noise reduction on synthetic audio: tone bursts (stand-in for speech) with pauses, plus steady noise."""
import numpy as np
import pytest

import tts
from audio import DENOISE_LEVELS, _mono_samples, background_level, float_to_wav, reduce_noise, wav_info
from config import get_settings

RATE = 24000


def signal(seconds=8.0, noise=0.01, seed=1):
    t = np.arange(int(seconds * RATE)) / RATE
    speaking = (t % 1.0) < 0.6  # 0.6 s on, 0.4 s pause
    clean = 0.25 * speaking * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t))
    mixed = clean + noise * np.random.default_rng(seed).standard_normal(t.size)
    return clean, float_to_wav(mixed.astype(np.float32), RATE)


def pauses(x):
    t = np.arange(x.size) / RATE
    return x[((t % 1.0) > 0.7) & ((t % 1.0) < 0.9)]


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)


@pytest.mark.parametrize("level,min_drop", [("light", 6), ("medium", 10), ("strong", 14)])
def test_noise_in_pauses_drops_and_speech_stays(level, min_drop):
    clean, noisy = signal()
    before, _ = _mono_samples(noisy)
    after, rate = _mono_samples(reduce_noise(noisy, level))
    assert rate == RATE and after.size == before.size
    assert db(pauses(before)) - db(pauses(after)) >= min_drop
    t = np.arange(clean.size) / RATE
    speech = ((t % 1.0) > 0.1) & ((t % 1.0) < 0.5)
    assert abs(db(after[speech]) - db(clean[speech])) < 1.0  # loudness of "speech" kept
    assert np.corrcoef(after[speech], clean[speech])[0, 1] > 0.99


def test_stronger_levels_remove_more():
    _, noisy = signal()
    levels = [background_level(reduce_noise(noisy, name)) for name in ("light", "medium", "strong")]
    assert levels[0] > levels[1] > levels[2] and background_level(noisy) > levels[0]


def test_clean_audio_is_barely_changed():
    clean, wav = signal(noise=0.0)
    out, _ = _mono_samples(reduce_noise(wav, "medium"))
    assert np.corrcoef(out, clean)[0, 1] > 0.995


def test_very_short_audio_is_left_alone(wav):
    short = wav(0.2)
    assert reduce_noise(short, "medium") == short


def test_unknown_level_rejected(wav):
    with pytest.raises(ValueError):
        reduce_noise(wav(2.0), "extreme")


def test_prepare_reference_trims_and_cleans():
    _, noisy = signal(seconds=30.0)
    off, info_off = tts.prepare_reference(noisy, "off")
    cleaned, info = tts.prepare_reference(noisy, "medium")
    assert info["trimmed"] and info["denoise"] == "medium" and info_off["denoise"] == "off"
    assert wav_info(cleaned)[0] == pytest.approx(tts.USE_REF_S, abs=0.01)
    assert background_level(cleaned) < background_level(off) - 8


@pytest.mark.parametrize("value,expected", [("", "off"), ("Medium", "medium"), ("loud", "off"), ("strong", "strong")])
def test_denoise_setting(monkeypatch, value, expected):
    monkeypatch.setenv("POCKET_DENOISE", value)
    assert get_settings().denoise == expected
    assert expected == "off" or expected in DENOISE_LEVELS
