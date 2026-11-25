import os

import pytest

import tts
from audio import trim_wav, wav_info
from errors import AppError


def test_wav_info_duration(wav):
    dur, rms = wav_info(wav(2.0))
    assert dur == pytest.approx(2.0, abs=0.01) and rms > 0.1


@pytest.mark.parametrize("data", [b"", b"not a wav file"])
def test_invalid_audio_rejected(data):
    with pytest.raises(AppError) as e:
        wav_info(data)
    assert e.value.category == "bad_audio"


def test_reference_too_short(wav):
    with pytest.raises(AppError, match="at least"):
        tts.validate_reference(wav(1.0))


def test_long_reference_is_valid_but_trimmed(wav):
    assert tts.validate_reference(wav(65.0, rate=8000)) == pytest.approx(65.0, abs=0.01)
    trimmed = trim_wav(wav(65.0, rate=8000), tts.USE_REF_S)
    assert wav_info(trimmed)[0] == pytest.approx(tts.USE_REF_S, abs=0.01)


def test_trim_leaves_short_audio_unchanged(wav):
    assert wav_info(trim_wav(wav(6.0), 20.0))[0] == pytest.approx(6.0, abs=0.01)


def test_absurdly_long_reference_rejected(wav):
    with pytest.raises(AppError, match="min"):
        tts.validate_reference(wav(700.0, rate=4000))


def test_reference_silent(wav):
    with pytest.raises(AppError, match="silent"):
        tts.validate_reference(wav(6.0, amp=0.0))


def test_reference_ok(wav):
    assert tts.validate_reference(wav(6.0)) == pytest.approx(6.0, abs=0.01)


def test_synthesize_without_profile():
    with pytest.raises(AppError) as e:
        tts.synthesize("hello")
    assert e.value.category == "no_profile"


def test_delete_profile_removes_files():
    from config import get_settings

    d = get_settings().profiles_dir / "a"
    d.mkdir(parents=True)
    (d / "reference.wav").write_bytes(b"x")
    (d / "profile.safetensors").write_bytes(b"x")
    assert tts.profile_exists("a")
    tts.delete_profile("a")
    assert not d.exists() and not tts.profile_exists("a") and tts.list_profiles() == []


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("REF_WAV"), reason="set REF_WAV=/path/to/your_voice.wav (run in Codespaces)")
def test_real_profile_roundtrip():
    """Real PocketTTS: build profile, drop in-memory state, reload from disk, synthesize."""
    tts.build_profile(open(os.environ["REF_WAV"], "rb").read())
    assert tts.profile_exists()
    tts._states.clear()  # simulate a restart: must load from the saved file
    out, secs = tts.synthesize("This is a test.")
    assert secs > 0.5 and wav_info(out)[0] == pytest.approx(secs, abs=0.1)
