import time

import pytest

import monitor
import store


def req(i, status="success", ts=None, total=1000.0):
    return {"id": f"r{i}", "ts": ts or time.time(), "asr_ms": 100, "llm_ms": 300, "tts_ms": 600, "total_ms": total,
            "in_audio_s": 2.0, "out_audio_s": 3.0, "status": status, "error_cat": None if status == "success" else "tts",
            "asr_model": "a", "llm_model": "l", "tts_model": "t", "prompt_tokens": 10, "completion_tokens": 5}


def test_insert_query_and_dedupe():
    assert store.init()
    store.insert_request(req(1))
    store.insert_request(req(1))  # duplicate id ignored
    assert len(store.query_requests(0)) == 1


def test_time_range_filter():
    store.init()
    store.insert_request(req(1, ts=time.time() - 7200))
    store.insert_request(req(2))
    assert [r["id"] for r in store.query_requests(time.time() - 3600)] == ["r2"]


def test_summarize_empty_has_no_placeholders():
    s = store.summarize([])
    assert s["total"] == 0 and s["avg_total_ms"] is None and s["p95_total_ms"] is None and s["total_tokens"] is None


def test_summarize_values_and_p95_threshold():
    rows = [req(i, total=float(i * 100)) for i in range(1, 11)] + [req(99, status="failed")]
    s = store.summarize(rows)
    assert (s["total"], s["success"], s["failed"]) == (11, 10, 1)
    assert s["avg_total_ms"] == pytest.approx(550) and s["median_total_ms"] == pytest.approx(550)
    assert s["p95_total_ms"] is None  # fewer than 20 samples
    assert s["total_tokens"] == 11 * 15 and s["total_audio_out_s"] == pytest.approx(30)
    many = [req(i, total=float(i)) for i in range(1, 101)]
    assert store.summarize(many)["p95_total_ms"] == 95.0


def test_retention_cleanup():
    store.init()
    now = time.time()
    store.insert_sample({"ts": now - 3 * 86400, "cpu": 1})
    store.insert_sample({"ts": now, "cpu": 2})
    store.cleanup()  # RETENTION_DAYS=1
    assert len(store.query_samples(0)) == 1


def test_store_failure_does_not_raise(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("DATA_DIR", str(blocker / "sub"))  # parent is a file: cannot create DB
    assert store.init() is False
    store.insert_request(req(1))
    assert store.query_requests(0) == [] and store.last_error()


def test_live_sample_shape_and_ranges():
    monitor.sample()
    s = monitor.sample()
    assert 0 <= s["cpu"] <= 100 and 0 <= s["mem_pct"] <= 100 and s["proc_rss"] > 0
    assert s["net_sent"] >= 0 and s["net_recv"] >= 0


def test_snapshot_sections():
    monitor._history.append(monitor.sample())
    d = monitor.snapshot()
    assert d["proc"]["pid"] > 0 and d["proc"]["threads"] >= 1 and d["proc"]["uptime_s"] >= 0
    assert d["mem"]["total"] >= d["mem"]["used"] > 0 and d["cpu"]["cores"] >= 1
    assert d["disk"]["total"] >= d["disk"]["used"]
    assert d["mem"]["peak_rss"] >= d["proc"]["rss"] * 0.5
    assert d["gpu"] is None or "util_pct" in d["gpu"]


def test_dir_size(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    assert monitor.dir_size(tmp_path, ttl=0) == 1000
    assert monitor.dir_size(tmp_path / "missing") is None


def test_history_in_memory():
    monitor._history.clear()
    monitor._history.append(monitor.sample())
    assert len(monitor.history(900)) == 1
