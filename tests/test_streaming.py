"""Fast replies: sentence splitting, streaming answers, streaming speech and fillers. No model, no network."""
import sqlite3
from types import SimpleNamespace as NS

import numpy as np
import pytest

import live
import llm
import pipeline
import store
import tts
from config import get_settings
from errors import AppError
from sentences import SentenceStream


def split(text, size=3):
    s, out = SentenceStream(), []
    for i in range(0, len(text), size):  # arrives a few characters at a time
        out += s.feed(text[i : i + size])
    return out + s.flush()


def test_sentences_come_out_whole_and_in_order():
    text = "Sure, happy to help. The market is down today. Anything else you need?"
    got = split(text)
    assert " ".join(got) == text and len(got) >= 3


@pytest.mark.parametrize("text", [
    "Dr. Rao and N. Chandrababu Naidu met in Amaravati on Friday.",
    "Nifty is at 25,704.1 right now, down 0.7 percent.",
])
def test_names_and_numbers_are_not_cut(text):
    got = split(text)
    assert " ".join(got) == text
    assert not any(p.endswith(("Dr.", "N.", "25,704.", "0.")) for p in got)


def test_first_piece_starts_early_at_a_comma():
    s = SentenceStream()
    assert s.feed("Well, that is a really good question, and ") == ["Well, that is a really good question,"]


def test_very_long_sentence_is_cut():
    pieces = split("word " * 120)
    assert len(pieces) > 1 and max(len(p) for p in pieces) <= 221


def test_citation_marks_are_not_split():
    got = split("It is a pet app 【2†L6. 10】 in India. Done.")
    assert all(p.count("【") == p.count("】") for p in got)


# ---------------------------------------------------------------- streaming answers
def chunk(content=None, calls=None, usage=None):
    delta = NS(content=content, tool_calls=calls)
    return NS(choices=[NS(delta=delta)], usage=None, x_groq={"usage": usage} if usage else None)


def part(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def stream_client(monkeypatch, rounds):
    sent = []

    def create(**kw):
        sent.append({**kw, "messages": list(kw["messages"])})
        return iter(rounds[len(sent) - 1])

    monkeypatch.setattr(llm, "_groq_client", lambda t: NS(chat=NS(completions=NS(create=create))))
    return sent


def test_answer_stream_yields_text_and_usage(monkeypatch):
    sent = stream_client(monkeypatch, [[chunk("Hi "), chunk("there."),
                                        chunk(usage={"prompt_tokens": 9, "completion_tokens": 3})]])
    usage = {}
    assert "".join(llm.answer_stream([], "hello", None, False, usage)) == "Hi there."
    assert usage == {"prompt_tokens": 9, "completion_tokens": 3}
    assert sent[0]["stream"] is True and "tools" not in sent[0]


def test_answer_stream_assembles_tool_calls_then_answers(monkeypatch):
    seen = []
    monkeypatch.setattr(live.Session, "run", lambda self, n, a: seen.append((n, a)) or '{"ok": 1}')
    first = [chunk(calls=[part(0, "c1", "market_quote", '{"na')]), chunk(calls=[part(0, args='me": "Nifty"}')])]
    sent = stream_client(monkeypatch, [first, [chunk("Nifty is down.")]])
    assert "".join(llm.answer_stream([], "nifty?", None, True, {})) == "Nifty is down."
    assert seen == [("market_quote", '{"name": "Nifty"}')]
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "tool"]


def test_answer_stream_empty_reply(monkeypatch):
    stream_client(monkeypatch, [[chunk()]])
    with pytest.raises(AppError, match="empty"):
        list(llm.answer_stream([], "q", None, False, {}))


# ---------------------------------------------------------------- the whole fast turn
def test_speech_starts_before_the_answer_is_finished(monkeypatch):
    store.init()
    events = []

    def fake_stream(history, question, prompt, use_web, usage):
        for text in ["Sure, happy to help you. ", "The weather ", "looks fine today. ", "Bye now."]:
            events.append("llm:" + text.strip())
            yield text
        usage.update(prompt_tokens=5, completion_tokens=7)

    def fake_tts(text, profile=None):
        events.append("tts:" + text)
        for _ in range(2):
            yield np.zeros(2400, dtype=np.float32), 24000

    monkeypatch.setattr(llm, "answer_stream", fake_stream)
    monkeypatch.setattr(tts, "synthesize_stream", fake_tts)
    monkeypatch.setattr(tts, "profile_exists", lambda *a: True)
    audio, texts, statuses = [], [], []
    r = pipeline.run_turn(None, [], text="hi", on_audio=audio.append, on_text=texts.append, on_status=statuses.append)
    assert r.status == "success" and r.answer == "Sure, happy to help you. The weather looks fine today. Bye now."
    assert events.index("tts:Sure, happy to help you.") < events.index("llm:Bye now.")  # overlapped
    assert len(audio) == 6 and all(a[:4] == b"RIFF" for a in audio) and r.out_audio_s == pytest.approx(0.6)
    assert texts[0] == "Sure, happy to help you." and statuses == ["Generating response", "Speaking", "Completed"]
    row = store.query_requests(0)[0]
    assert 0 <= row["first_audio_ms"] <= row["total_ms"] and row["prompt_tokens"] == 5
    assert store.summarize([row])["avg_first_audio_ms"] == row["first_audio_ms"]


def test_old_database_gets_the_new_column():
    s = get_settings()
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    old = store.SCHEMA.replace(", first_audio_ms REAL", "")
    with sqlite3.connect(s.db_path) as c:
        c.executescript(old)
    assert store.init()
    store.insert_request({"id": "x", "ts": 1.0, "status": "success", "first_audio_ms": 12.0})
    assert store.query_requests(0)[0]["first_audio_ms"] == 12.0


# ---------------------------------------------------------------- streaming speech and fillers
class FakeModel:
    sample_rate = 1000

    def generate_audio_stream(self, state, text):
        for _ in range(25):
            yield np.zeros(100, dtype=np.float32)  # 0.1 s pieces


def add_voice(name="a"):
    d = get_settings().profiles_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile.safetensors").write_bytes(b"x")


def test_stream_groups_small_pieces(monkeypatch):
    add_voice()
    monkeypatch.setattr(tts, "_get_model", lambda: FakeModel())
    monkeypatch.setattr(tts, "_load_state", lambda name: "state")
    sizes = [x.size for x, _ in tts.synthesize_stream("hello", first_chunk_s=0.4, chunk_s=0.8)]
    assert sizes == [400, 800, 800, 500]  # quick first piece, then larger ones; nothing lost
    assert not tts._lock.locked()


def test_fillers_are_made_once_and_never_on_demand(monkeypatch):
    add_voice()
    made = []
    monkeypatch.setattr(tts, "synthesize", lambda text, profile=None: made.append(text) or (b"RIFF" + text.encode(), 1.0))
    assert tts.filler("search", "a") is None and made == []  # asking never generates
    count = tts.make_fillers("a")
    assert count == sum(len(v) for v in tts.FILLERS.values()) == len(made)
    assert tts.make_fillers("a") == 0
    assert tts.filler("search", "a").startswith(b"RIFF")


def test_warm_up_never_raises(monkeypatch):
    tts.warm_up()  # no voice yet
    add_voice()
    monkeypatch.setattr(tts, "_load_state", lambda n: (_ for _ in ()).throw(AppError("tts", "no model")))
    tts.warm_up()
