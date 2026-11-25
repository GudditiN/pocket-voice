import os

os.environ["POCKET_VOICE_NO_DOTENV"] = "1"  # before app modules import config: a real .env must not affect tests

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from audio import float_to_wav  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("RETENTION_DAYS", "1")


def make_wav(seconds=6.0, amp=0.3, rate=16000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return float_to_wav((amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


@pytest.fixture
def wav():
    return make_wav
