import pytest

from audio import normalize_wav, to_dbfs, wav_info
import tts
from errors import AppError


def level_db(data):
    return to_dbfs(wav_info(data)[1])


def test_quiet_recording_is_accepted_and_boosted(wav):
    quiet = wav(6.0, amp=0.004)  # about -49 dBFS: the old check would have rejected this
    assert tts.validate_reference(quiet) == pytest.approx(6.0, abs=0.01)
    boosted = normalize_wav(quiet)
    assert level_db(boosted) == pytest.approx(-20.0, abs=1.5)
    assert wav_info(boosted)[0] == pytest.approx(6.0, abs=0.01)


def test_loud_recording_is_unchanged(wav):
    loud = wav(6.0, amp=0.3)
    assert normalize_wav(loud) == loud


def test_boost_never_clips(wav):
    import numpy as np
    from audio import float_to_wav

    # quiet tone with one loud spike: gain must be limited by the peak
    t = np.arange(96000) / 16000
    x = (0.003 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x[1000] = 0.9
    out = normalize_wav(float_to_wav(x, 16000))
    import io
    import wave

    with wave.open(io.BytesIO(out)) as w:
        s = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert np.abs(s).max() <= 32767


def test_truly_silent_recording_still_rejected(wav):
    with pytest.raises(AppError, match="silent or too quiet"):
        tts.validate_reference(wav(6.0, amp=0.0))
    with pytest.raises(AppError, match="dBFS"):
        tts.validate_reference(wav(6.0, amp=0.0005))  # about -69 dBFS: below the floor
