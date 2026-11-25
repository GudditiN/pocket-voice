from types import SimpleNamespace

import groq
import httpx
import pytest

import asr
import llm
from errors import AppError, from_groq

REQ = httpx.Request("POST", "https://api.groq.com/x")


def resp(code):
    return httpx.Response(code, request=REQ)


class FakeClient:
    def __init__(self, exc=None, text="hello", content="Hi there.", usage=(5, 3)):
        self.exc = exc
        outer = self

        def _t(**kw):
            if outer.exc:
                raise outer.exc
            return SimpleNamespace(text=text)

        def _c(**kw):
            if outer.exc:
                raise outer.exc
            outer.kwargs = kw
            msg = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                                   usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))

        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=_t))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_c))


def patch(monkeypatch, client):
    monkeypatch.setattr(asr, "_groq_client", lambda timeout: client)
    monkeypatch.setattr(llm, "_groq_client", lambda timeout: client)


def test_transcribe_ok(monkeypatch, wav):
    patch(monkeypatch, FakeClient(text="  what time is it "))
    assert asr.transcribe(wav(1)) == "what time is it"


def test_answer_returns_usage_and_uses_history(monkeypatch):
    c = FakeClient()
    patch(monkeypatch, c)
    text, usage = llm.answer([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "c")
    assert text == "Hi there." and usage == {"prompt_tokens": 5, "completion_tokens": 3}
    roles = [m["role"] for m in c.kwargs["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize("exc,cat", [
    (groq.AuthenticationError("x", response=resp(401), body=None), "auth"),
    (groq.RateLimitError("x", response=resp(429), body=None), "rate_limit"),
    (groq.APITimeoutError(request=REQ), "timeout"),
    (groq.APIConnectionError(request=REQ), "network"),
    (groq.InternalServerError("x", response=resp(500), body=None), "llm"),
])
def test_llm_error_mapping(monkeypatch, exc, cat):
    patch(monkeypatch, FakeClient(exc=exc))
    with pytest.raises(AppError) as e:
        llm.answer([], "hi")
    assert e.value.category == cat


def test_asr_bad_request_is_bad_audio(monkeypatch, wav):
    patch(monkeypatch, FakeClient(exc=groq.BadRequestError("x", response=resp(400), body=None)))
    with pytest.raises(AppError) as e:
        asr.transcribe(wav(1))
    assert e.value.category == "bad_audio"


def test_missing_key(monkeypatch, wav):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(AppError) as e:
        asr.transcribe(wav(1))
    assert e.value.category == "missing_key"


def test_empty_llm_answer(monkeypatch):
    patch(monkeypatch, FakeClient(content="  "))
    with pytest.raises(AppError):
        llm.answer([], "hi")


def test_from_groq_unknown_exception():
    assert from_groq(ValueError("boom"), "llm").category == "llm"
