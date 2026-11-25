import io

import numpy as np
import pytest

from audio import decode_to_wav, wav_info
from errors import AppError

sf = pytest.importorskip("soundfile")


def encode(fmt, seconds=6.0, rate=22050, channels=1):
    t = np.arange(int(seconds * rate)) / rate
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, rate, format=fmt)
    return buf.getvalue()


def test_pcm_wav_passes_through_unchanged(wav):
    data = wav(3.0)
    assert decode_to_wav(data) == data


@pytest.mark.parametrize("fmt", ["FLAC", "OGG", "MP3"])
def test_other_formats_become_pcm_wav(fmt):
    if fmt not in sf.available_formats():
        pytest.skip(f"this libsndfile can't write {fmt}")
    out = decode_to_wav(encode(fmt))
    dur, rms = wav_info(out)  # valid 16-bit PCM WAV now
    assert dur == pytest.approx(6.0, abs=0.2) and rms > 0.1


def test_stereo_is_mixed_to_mono():
    out = decode_to_wav(encode("FLAC", channels=2))
    import wave

    with wave.open(io.BytesIO(out)) as w:
        assert w.getnchannels() == 1


def test_float_wav_is_converted():
    out = decode_to_wav(encode("WAV"))  # soundfile writes 16-bit by default; force float below
    buf = io.BytesIO()
    sf.write(buf, np.zeros(22050, dtype=np.float32) + 0.2, 22050, format="WAV", subtype="FLOAT")
    assert wav_info(decode_to_wav(buf.getvalue()))[0] == pytest.approx(1.0, abs=0.05)
    assert wav_info(out)[0] > 0


def test_garbage_is_rejected_with_clear_message():
    with pytest.raises(AppError, match="Supported"):
        decode_to_wav(b"this is not audio at all")
