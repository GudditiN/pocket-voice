import asr
import llm
import pipeline
import store
import tts
from errors import AppError


def setup_ok(monkeypatch):
    store.init()
    monkeypatch.setattr(tts, "profile_exists", lambda *a: True)
    monkeypatch.setattr(asr, "transcribe", lambda b: "what is the capital of France")
    monkeypatch.setattr(llm, "answer",
                        lambda h, q, p=None, **k: ("Paris.", {"prompt_tokens": 12, "completion_tokens": 2}))
    monkeypatch.setattr(tts, "synthesize", lambda t, **k: (b"RIFFfake", 1.5))


def test_success_records_metrics_without_content(monkeypatch, wav):
    setup_ok(monkeypatch)
    seen = []
    r = pipeline.run_turn(wav(2.0), [], on_status=seen.append)
    assert r.status == "success" and r.answer == "Paris." and r.audio_wav
    assert seen == ["Transcribing", "Generating response", "Generating speech", "Completed"]
    (row,) = store.query_requests(0)
    assert row["status"] == "success" and row["prompt_tokens"] == 12 and row["out_audio_s"] == 1.5
    assert row["in_audio_s"] > 1.9 and row["total_ms"] >= row["asr_ms"]
    assert "Paris" not in str(row) and "France" not in str(row)  # lengths only, never content


def test_typed_question_skips_speech_recognition(monkeypatch):
    setup_ok(monkeypatch)

    def no_asr(b):
        raise AssertionError("ASR must not run for typed questions")

    monkeypatch.setattr(asr, "transcribe", no_asr)
    seen = []
    r = pipeline.run_turn(None, [], text="  What is 2+2?  ", on_status=seen.append)
    assert r.status == "success" and r.question == "What is 2+2?" and r.audio_wav
    assert seen == ["Generating response", "Generating speech", "Completed"]
    (row,) = store.query_requests(0)
    assert row["asr_ms"] is None and row["in_audio_s"] is None and row["asr_model"] == "typed"
    assert row["llm_ms"] is not None and row["tts_ms"] is not None


def test_empty_typed_question(monkeypatch):
    setup_ok(monkeypatch)
    r = pipeline.run_turn(None, [], text="   ")
    assert r.status == "failed" and r.error_category == "empty_input"


def test_no_profile(monkeypatch, wav):
    store.init()
    monkeypatch.setattr(tts, "profile_exists", lambda *a: False)
    r = pipeline.run_turn(wav(2.0), [])
    assert r.status == "failed" and r.error_category == "no_profile"
    assert store.query_requests(0)[0]["error_cat"] == "no_profile"
    assert store.query_errors(0)[0]["category"] == "no_profile"


def test_empty_and_silent_audio(monkeypatch, wav):
    setup_ok(monkeypatch)
    assert pipeline.run_turn(b"", []).error_category == "bad_audio"
    assert pipeline.run_turn(wav(2.0, amp=0.0), []).error_category == "bad_audio"


def test_empty_transcript(monkeypatch, wav):
    setup_ok(monkeypatch)
    monkeypatch.setattr(asr, "transcribe", lambda b: "")
    assert pipeline.run_turn(wav(2.0), []).error_category == "bad_audio"


def test_stage_failure_keeps_earlier_timings(monkeypatch, wav):
    setup_ok(monkeypatch)

    def boom(t, **k):
        raise AppError("tts", "synthesis failed")

    monkeypatch.setattr(tts, "synthesize", boom)
    r = pipeline.run_turn(wav(2.0), [])
    assert r.status == "failed" and r.error_category == "tts"
    row = store.query_requests(0)[0]
    assert row["asr_ms"] is not None and row["llm_ms"] is not None and row["tts_ms"] is None


def test_unexpected_exception_is_contained(monkeypatch, wav):
    setup_ok(monkeypatch)
    monkeypatch.setattr(llm, "answer", lambda *a, **k: 1 / 0)
    assert pipeline.run_turn(wav(2.0), []).error_category == "unknown"


def test_product_search_shows_cards_and_skips_the_model(monkeypatch, wav):
    import live

    store.init()
    setup_ok(monkeypatch)

    def no_model(*a, **k):
        raise AssertionError("the model must not be called for a product search")

    def fake_lookup(session, tool, value):
        session.cards.append({"title": "Dog Food", "url": "https://sniffynose.com/products/x", "snippet": "", "image": ""})
        return {"products": [{"name": "Dog Food", "price_inr": 510}]}

    monkeypatch.setattr(llm, "answer", no_model)
    monkeypatch.setattr(llm, "answer_stream", no_model)
    monkeypatch.setattr(live.Session, "lookup", fake_lookup)
    monkeypatch.setattr(tts, "phrase", lambda kind, profile=None: (f"phrase:{kind}", wav(1.0)))
    played = []
    r = pipeline.run_turn(None, [], text="show me dog food", use_web=True, on_audio=played.append)
    assert r.status == "success" and r.answer == "phrase:products" and len(played) == 1
    assert r.cards[0]["title"] == "Dog Food" and "1. Dog Food (Rs 510)" in r.context
    row = store.query_requests(0)[0]
    assert row["llm_ms"] is None and row["llm_model"] == "product lookup (no model)"


def test_chosen_voice_is_used(monkeypatch, wav):
    setup_ok(monkeypatch)
    seen = {}
    monkeypatch.setattr(tts, "profile_exists", lambda name=None: seen.setdefault("checked", name) or True)
    monkeypatch.setattr(tts, "synthesize", lambda t, profile=None: (seen.update(spoke=profile) or (b"RIFFfake", 1.0)))
    assert pipeline.run_turn(wav(2.0), [], profile="studio").status == "success"
    assert seen == {"checked": "studio", "spoke": "studio"}
