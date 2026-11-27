"""Run the real Streamlit pages headlessly (stand-in profile files, no model, no network)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import tts  # noqa: E402
import ui  # noqa: E402,F401
from config import get_settings  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app"
HOME = str(APP / "Home.py")
VOICES = str(APP / "pages" / "1_Voice_Setup.py")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for key in ("APP_PASSWORD", "ADMIN_PASSWORD", "POCKET_EXPOSED"):
        monkeypatch.delenv(key, raising=False)
    import monitor

    monkeypatch.setattr(monitor, "start_sampler", lambda: None)
    ui._throttle.__init__()
    ui._boot.clear()


def add_voice(name):
    d = get_settings().profiles_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.safetensors").write_bytes(b"x")


def public(monkeypatch):
    monkeypatch.setenv("POCKET_EXPOSED", "1")
    monkeypatch.setenv("APP_PASSWORD", "user-password-1")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-password-1")


def page(path):
    at = AppTest.from_file(path, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def labels(at):
    return [b.label for b in at.button]


def login(at, password):
    at.text_input[0].input(password)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def test_assistant_lets_you_choose_a_voice():
    add_voice("alpha")
    add_voice("beta")
    tts.set_active("beta")
    at = page(HOME)
    box = at.selectbox[0]
    assert list(box.options) == ["alpha", "beta"] and box.value == "beta"
    box.select("alpha").run()
    assert at.selectbox[0].value == "alpha" and tts.active_profile() == "beta"  # session only


def test_typed_message_is_answered_in_the_chosen_voice(monkeypatch):
    import pipeline

    seen = {}

    def fake_turn(audio, history, prompt, **kw):
        seen.update(audio=audio, text=kw["text"], voice=kw["profile"])
        return pipeline.TurnResult("id1", status="success", question=kw["text"], answer="Paris.",
                                   audio_wav=b"RIFFfake", timings_ms={"llm": 100, "tts": 200, "total": 300})

    monkeypatch.setattr(pipeline, "run_turn", fake_turn)
    add_voice("alpha")
    at = page(HOME)
    assert len(at.chat_input) == 1
    at.chat_input[0].set_value("Capital of France?").run()
    assert not at.exception, at.exception
    assert seen == {"audio": None, "text": "Capital of France?", "voice": "alpha"}
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert at.session_state["turns"][0]["a"] == "Paris."


def test_failed_turn_removes_the_thinking_animation(monkeypatch):
    import pipeline

    def fail(audio, history, prompt, **kw):
        kw["on_status"]("Generating response")
        return pipeline.TurnResult("id2", error_category="network", error_message="Could not reach Groq.")

    monkeypatch.setattr(pipeline, "run_turn", fail)
    add_voice("alpha")
    at = page(HOME)
    at.chat_input[0].set_value("hello").run()
    assert not at.exception, at.exception
    assert any("Could not reach Groq" in e.value for e in at.error)
    assert not any("<div class='pv-think'" in m.value for m in at.markdown)


def test_assistant_without_voices_warns():
    at = page(HOME)
    assert not at.selectbox and any("No voice profile yet" in w.value for w in at.warning)


def test_voice_page_lists_voices_and_changes_default():
    add_voice("alpha")
    add_voice("beta")
    tts.set_active("alpha")
    at = page(VOICES)
    assert labels(at).count("▶ Test") == 2 and "Create voice" in labels(at)
    at.button(key="default_beta").click().run()
    assert not at.exception and tts.active_profile() == "beta"


def test_voice_page_delete():
    add_voice("alpha")
    add_voice("beta")
    at = page(VOICES)
    at.button(key="delete_beta").click().run()
    assert not at.exception and [p["name"] for p in tts.list_profiles()] == ["alpha"]


def test_public_link_requires_a_password(monkeypatch):
    public(monkeypatch)
    add_voice("alpha")
    at = page(HOME)
    assert not at.selectbox and labels(at) == ["Unlock"]
    login(at, "nope")
    assert any("Wrong password" in e.value for e in at.error) and not at.selectbox


def test_public_visitor_cannot_manage_voices_or_edit_prompt(monkeypatch):
    public(monkeypatch)
    add_voice("alpha")
    home = login(page(HOME), "user-password-1")
    assert home.selectbox[0].value == "alpha" and not home.text_area
    voices = login(page(VOICES), "user-password-1")
    got = labels(voices)
    assert "▶ Test" in got and "Create voice" not in got
    assert not [b for b in got if "default" in b.lower() or "delete" in b.lower()]


def test_public_admin_can_manage(monkeypatch):
    public(monkeypatch)
    add_voice("alpha")
    voices = login(page(VOICES), "admin-password-1")
    assert "Create voice" in labels(voices) and "Yes, delete" in labels(voices)


def test_public_without_password_serves_nothing(monkeypatch):
    monkeypatch.setenv("POCKET_EXPOSED", "1")
    add_voice("alpha")
    at = page(HOME)
    assert not at.selectbox and not at.button and any("APP_PASSWORD is not set" in e.value for e in at.error)
