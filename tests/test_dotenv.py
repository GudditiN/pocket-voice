import os

import pytest

from config import get_settings, load_dotenv

KEYS = ["PV_TEST_A", "PV_TEST_B", "PV_TEST_C", "PV_TEST_D", "PV_TEST_E", "PV_TEST_EMPTY"]


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
    for k in KEYS:  # load_dotenv writes os.environ directly, so remove what it added
        os.environ.pop(k, None)


def write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text)
    return [p]


def test_loads_plain_quoted_export_and_skips_comments(tmp_path):
    paths = write(tmp_path, '# comment\n\nPV_TEST_A=one\nPV_TEST_B = "two words"\nexport PV_TEST_C=\'three\'\nnot a pair\n')
    assert sorted(load_dotenv(paths)) == ["PV_TEST_A", "PV_TEST_B", "PV_TEST_C"]
    assert (os.environ["PV_TEST_A"], os.environ["PV_TEST_B"], os.environ["PV_TEST_C"]) == ("one", "two words", "three")


def test_existing_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_TEST_A", "from-secret")
    assert load_dotenv(write(tmp_path, "PV_TEST_A=from-file\n")) == []
    assert os.environ["PV_TEST_A"] == "from-secret"


def test_blank_values_are_skipped_so_defaults_still_apply(tmp_path, monkeypatch):
    load_dotenv(write(tmp_path, "PV_TEST_EMPTY=\nGROQ_LLM_MODEL=\n"))
    assert "PV_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
    assert get_settings().llm_model == "openai/gpt-oss-20b"


def test_value_may_contain_equals_and_hash(tmp_path):
    load_dotenv(write(tmp_path, "PV_TEST_D=a=b # not a comment\n"))
    assert os.environ["PV_TEST_D"] == "a=b # not a comment"


def test_missing_file_is_fine(tmp_path):
    assert load_dotenv([tmp_path / "nope.env"]) == []


def test_settings_pick_up_dotenv_values(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_LLM_MODEL", raising=False)
    try:
        load_dotenv(write(tmp_path, "GROQ_LLM_MODEL=openai/gpt-oss-120b\n"))
        assert get_settings().llm_model == "openai/gpt-oss-120b"
    finally:
        os.environ.pop("GROQ_LLM_MODEL", None)
