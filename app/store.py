"""SQLite metrics store. Never raises into the app: failures are logged and return defaults."""
from __future__ import annotations

import logging
import sqlite3
import statistics
import threading
import time

from config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()
_last_error: str | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id TEXT PRIMARY KEY, ts REAL NOT NULL,
  asr_ms REAL, llm_ms REAL, tts_ms REAL, total_ms REAL,
  in_audio_s REAL, out_audio_s REAL,
  status TEXT NOT NULL, error_cat TEXT,
  asr_model TEXT, llm_model TEXT, tts_model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER,
  question_chars INTEGER, answer_chars INTEGER, first_audio_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
CREATE TABLE IF NOT EXISTS resource_samples (
  ts REAL PRIMARY KEY, cpu REAL, proc_cpu REAL, mem_pct REAL, proc_mem_pct REAL,
  proc_rss REAL, net_sent REAL, net_recv REAL, net_up_bps REAL, net_down_bps REAL
);
CREATE TABLE IF NOT EXISTS errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, category TEXT, message TEXT
);
CREATE INDEX IF NOT EXISTS idx_errors_ts ON errors(ts);
"""

REQUEST_COLS = ["id", "ts", "asr_ms", "llm_ms", "tts_ms", "total_ms", "in_audio_s", "out_audio_s", "status",
                "error_cat", "asr_model", "llm_model", "tts_model", "prompt_tokens", "completion_tokens",
                "question_chars", "answer_chars", "first_audio_ms"]
SAMPLE_COLS = ["ts", "cpu", "proc_cpu", "mem_pct", "proc_mem_pct", "proc_rss", "net_sent", "net_recv",
               "net_up_bps", "net_down_bps"]


def last_error() -> str | None:
    return _last_error


def _run(fn, default=None, write=False):
    global _last_error
    try:
        with _lock:
            path = get_settings().db_path
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path, timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                result = fn(conn)
                if write:
                    conn.commit()
                _last_error = None
                return result
            finally:
                conn.close()
    except Exception as e:  # sqlite3.Error, OSError (disk full / read-only)
        _last_error = f"{type(e).__name__}: {e}"
        log.warning("metrics store unavailable: %s", _last_error)
        return default


def init() -> bool:
    def fn(c):
        c.executescript(SCHEMA)
        have = {row[1] for row in c.execute("PRAGMA table_info(requests)")}
        if "first_audio_ms" not in have:  # databases created by earlier versions
            c.execute("ALTER TABLE requests ADD COLUMN first_audio_ms REAL")
        return True

    return _run(fn, False, write=True)


def _insert(table: str, cols: list[str], row: dict) -> None:
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
    _run(lambda c: c.execute(sql, [row.get(k) for k in cols]), write=True)


def insert_request(row: dict) -> None:
    _insert("requests", REQUEST_COLS, row)


def insert_sample(row: dict) -> None:
    _insert("resource_samples", SAMPLE_COLS, row)


def log_error(category: str, message: str) -> None:
    _run(lambda c: c.execute("INSERT INTO errors (ts, category, message) VALUES (?,?,?)",
                             (time.time(), category, message[:300])), write=True)


def _query(table: str, since: float, limit: int | None = None) -> list[dict]:
    sql = f"SELECT * FROM {table} WHERE ts >= ? ORDER BY ts" + (f" LIMIT {int(limit)}" if limit else "")
    return _run(lambda c: [dict(r) for r in c.execute(sql, (since,))], [])


def query_requests(since: float) -> list[dict]:
    return _query("requests", since)


def query_samples(since: float) -> list[dict]:
    return _query("resource_samples", since)


def query_errors(since: float, limit: int = 50) -> list[dict]:
    return _run(lambda c: [dict(r) for r in c.execute(
        "SELECT * FROM errors WHERE ts >= ? ORDER BY ts DESC LIMIT ?", (since, limit))], [])


def cleanup() -> int:
    """Delete resource samples/errors older than RETENTION_DAYS. Request rows are kept (tiny)."""
    cutoff = time.time() - get_settings().retention_days * 86400

    def fn(c):
        n = c.execute("DELETE FROM resource_samples WHERE ts < ?", (cutoff,)).rowcount
        c.execute("DELETE FROM errors WHERE ts < ?", (cutoff,))
        return n

    return _run(fn, 0, write=True)


def _avg(vals):
    return statistics.fmean(vals) if vals else None


def _p95(vals):
    """Nearest-rank P95; only reported with at least 20 samples."""
    if len(vals) < 20:
        return None
    s = sorted(vals)
    return s[max(0, int(round(0.95 * len(s))) - 1)]


def summarize(rows: list[dict]) -> dict:
    """Aggregate request rows. Values are None when there is no real data (never placeholders)."""
    ok = [r for r in rows if r["status"] == "success"]

    def col(name, src=ok):
        return [r[name] for r in src if r.get(name) is not None]

    totals = col("total_ms")
    tokens = [(r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rows
              if r.get("prompt_tokens") is not None or r.get("completion_tokens") is not None]
    return {
        "total": len(rows), "success": len(ok), "failed": len(rows) - len(ok),
        "avg_total_ms": _avg(totals),
        "median_total_ms": statistics.median(totals) if totals else None,
        "p95_total_ms": _p95(totals),
        "avg_first_audio_ms": _avg(col("first_audio_ms")),
        "avg_asr_ms": _avg(col("asr_ms")), "avg_llm_ms": _avg(col("llm_ms")), "avg_tts_ms": _avg(col("tts_ms")),
        "total_audio_out_s": sum(col("out_audio_s")) if col("out_audio_s") else None,
        "total_tokens": sum(tokens) if tokens else None,
    }
