import asr
import llm
import pipeline
import store
import tts
from errors import AppError


def setup_ok(monkeypatch):
    store.init()
    monkeypatch.setattr(tts, "profile_exists", lambda: True)
    monkeypatch.setattr(asr, "transcribe", lambda b: "what is the capital of France")
    monkeypatch.setattr(llm, "answer", lambda h, q, p=None: ("Paris.", {"prompt_tokens": 12, "completion_tokens": 2}))
    monkeypatch.setattr(tts, "synthesize", lambda t: (b"RIFFfake", 1.5))


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
    monkeypatch.setattr(tts, "profile_exists", lambda: False)
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

    def boom(t):
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
