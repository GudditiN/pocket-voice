import os

import pytest

import tts
from audio import wav_info
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


def test_reference_too_long(wav):
    with pytest.raises(AppError, match="under"):
        tts.validate_reference(wav(70.0, rate=8000))


def test_reference_silent(wav):
    with pytest.raises(AppError, match="silent"):
        tts.validate_reference(wav(6.0, amp=0.0))


def test_reference_ok(wav):
    assert tts.validate_reference(wav(6.0)) == pytest.approx(6.0, abs=0.01)


def test_synthesize_without_profile():
    with pytest.raises(AppError) as e:
        tts.synthesize("hello")
    assert e.value.category == "no_profile"


def test_delete_profile_removes_files(tmp_path):
    from config import get_settings

    s = get_settings()
    s.voice_dir.mkdir(parents=True)
    s.reference_path.write_bytes(b"x")
    s.profile_path.write_bytes(b"x")
    tts.delete_profile()
    assert not s.reference_path.exists() and not tts.profile_exists()


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("REF_WAV"), reason="set REF_WAV=/path/to/your_voice.wav (run in Codespaces)")
def test_real_profile_roundtrip():
    """Real PocketTTS: build profile, drop in-memory state, reload from disk, synthesize."""
    tts.build_profile(open(os.environ["REF_WAV"], "rb").read())
    assert tts.profile_exists()
    tts._state = None  # simulate a restart: must load from the saved file
    out, secs = tts.synthesize("This is a test.")
    assert secs > 0.5 and wav_info(out)[0] == pytest.approx(secs, abs=0.1)
