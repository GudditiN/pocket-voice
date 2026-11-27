import numpy as np
import pytest

import tts
from config import get_settings
from errors import AppError


def make_profile(name):
    d = get_settings().profiles_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.safetensors").write_bytes(b"x")
    (d / "reference.wav").write_bytes(b"x")


def test_slug_sanitises_names():
    assert tts.slug("  My Voice #1! ") == "my-voice-1"
    with pytest.raises(AppError):
        tts.slug("!!!")


def test_list_and_active_profile():
    assert tts.list_profiles() == [] and tts.active_profile() is None
    make_profile("a")
    make_profile("b")
    assert [p["name"] for p in tts.list_profiles()] == ["a", "b"]
    assert tts.active_profile() == "a"  # falls back to the first one
    tts.set_active("b")
    assert tts.active_profile() == "b"
    with pytest.raises(AppError):
        tts.set_active("missing")


def test_deleting_active_profile_falls_back():
    make_profile("a")
    make_profile("b")
    tts.set_active("b")
    tts.delete_profile()  # deletes the active one
    assert tts.active_profile() == "a"


def test_legacy_single_profile_is_migrated():
    s = get_settings()
    s.voice_dir.mkdir(parents=True)
    s.legacy_profile_path.write_bytes(b"x")
    s.legacy_reference_path.write_bytes(b"x")
    assert tts.active_profile() == "default"
    assert tts.profile_exists("default") and not s.legacy_profile_path.exists()


def test_tuning_defaults_clamp_save_reset():
    assert tts.get_tuning() == tts.DEFAULT_TUNING
    saved = tts.save_tuning({"temp": 9, "steps": 2.6, "eos_threshold": -100})
    assert saved == {**tts.DEFAULT_TUNING, "temp": 1.5, "steps": 3, "eos_threshold": -10.0}
    assert tts.get_tuning() == saved
    assert tts.clean_tuning({"temp": "bad"})["temp"] == tts.DEFAULT_TUNING["temp"]
    tts.reset_tuning()
    assert tts.get_tuning() == tts.DEFAULT_TUNING


def test_env_overrides_saved_tuning_per_setting(monkeypatch):
    tts.save_tuning({"temp": 0.9, "steps": 4})
    monkeypatch.setenv("POCKET_TEMP", "0.5")
    monkeypatch.setenv("POCKET_EOS_THRESHOLD", "-3")
    assert tts.get_tuning() == {**tts.DEFAULT_TUNING, "temp": 0.5, "steps": 4, "eos_threshold": -3.0}
    monkeypatch.setenv("POCKET_STEPS", "99")  # clamped to the allowed range
    assert tts.get_tuning()["steps"] == 8


@pytest.mark.parametrize("bad", ["", "  ", "loud", "nan", "inf"])
def test_invalid_env_tuning_is_ignored(monkeypatch, bad):
    tts.save_tuning({"temp": 0.9})
    monkeypatch.setenv("POCKET_TEMP", bad)
    assert tts.get_tuning()["temp"] == 0.9


def test_synthesize_applies_tuning_and_profile(monkeypatch):
    class FakeModel:
        sample_rate = 24000

        def generate_audio(self, state, text):
            self.seen = (state, text, self.temp, self.sampler_decode_steps, self.eos_threshold)
            return np.zeros(24000, dtype=np.float32)

    fake = FakeModel()
    make_profile("a")
    monkeypatch.setattr(tts, "_get_model", lambda: fake)
    monkeypatch.setattr(tts, "_load_state", lambda name: f"state-{name}")
    wav, secs = tts.synthesize("hello", tuning={"temp": 0.7, "steps": 2, "eos_threshold": -3})
    assert secs == pytest.approx(1.0) and wav[:4] == b"RIFF"
    assert fake.seen == ("state-a", "hello", 0.7, 2, -3.0)
    tts.save_tuning({"temp": 0.5})
    tts.synthesize("hi")  # uses the saved tuning when none is given
    assert fake.seen[2:] == (0.5, 1, -4.0)


def test_each_voice_keeps_its_own_choice(monkeypatch):
    make_profile("a")
    make_profile("b")
    monkeypatch.setenv("POCKET_TEMP", "0.9")
    assert tts.get_tuning("a")["temp"] == 0.9 and not tts.is_tuned("a")  # .env is the starting point
    tts.save_tuning({"temp": 0.4, "pitch": 1.5, "clean": 2}, voice="a")
    chosen = tts.get_tuning("a")
    assert (chosen["temp"], chosen["pitch"], chosen["clean"]) == (0.4, 1.5, 2)  # the app's choice beats .env
    assert tts.get_tuning("b")["pitch"] == 0 and tts.get_tuning()["pitch"] == 0
    tts.reset_tuning("a")
    assert tts.get_tuning("a")["temp"] == 0.9
    with pytest.raises(AppError):
        tts.save_tuning({}, voice="missing")


def test_deleting_a_voice_removes_its_choice():
    make_profile("a")
    tts.save_tuning({"speed": 1.2}, voice="a")
    tts.delete_profile("a")
    make_profile("a")
    assert tts.get_tuning("a")["speed"] == 1.0


def test_comparisons_start_from_the_voice_settings():
    make_profile("a")
    tts.save_tuning({"temp": 0.6, "speed": 1.08}, voice="a")
    styles = tts.style_choices("a")
    assert [name for name, _ in styles] == ["Current", "Steady", "Natural", "Expressive"]
    assert styles[0][1]["temp"] == 0.6 and all(t["speed"] == 1.08 for _, t in styles)
    pitches = tts.variation_choices("Pitch", "a")
    assert [t["pitch"] for _, t in pitches] == [-3.0, -1.5, 0.0, 1.5, 3.0] and pitches[0][1]["temp"] == 0.6


def test_speech_uses_the_voice_choice(monkeypatch):
    import numpy as np

    class FakeModel:
        sample_rate = 24000

        def generate_audio(self, state, text):
            return np.zeros(24000, dtype=np.float32)

    make_profile("a")
    monkeypatch.setattr(tts, "_get_model", FakeModel)
    monkeypatch.setattr(tts, "_load_state", lambda name: "state")
    before = tts._filler_path("a", "Okay.")
    tts.save_tuning({"speed": 1.25}, voice="a")
    assert tts.synthesize("hi", profile="a")[1] == pytest.approx(0.8, abs=0.01)
    assert tts._filler_path("a", "Okay.") != before  # waiting phrases are remade with the new sound
