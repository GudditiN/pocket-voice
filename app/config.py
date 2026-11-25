"""Environment-based settings. Read fresh on each call so tests can monkeypatch env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPT = (
    "You are a friendly voice assistant. Answer in one to three short spoken sentences. "
    "Plain text only: no markdown, lists, or emojis."
)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    groq_api_key: str
    asr_model: str
    llm_model: str
    asr_backend: str
    system_prompt: str
    data_dir: Path
    retention_days: int
    request_timeout_s: int
    history_turns: int
    sample_interval_s: int
    app_password: str

    @property
    def voice_dir(self) -> Path:
        return self.data_dir / "voice"

    @property
    def reference_path(self) -> Path:
        return self.voice_dir / "reference.wav"

    @property
    def profile_path(self) -> Path:
        return self.voice_dir / "profile.safetensors"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "metrics.db"

    @property
    def asr_model_label(self) -> str:
        return self.asr_model if self.asr_backend == "groq" else "faster-whisper-base"


def get_settings() -> Settings:
    e = os.environ.get
    return Settings(
        groq_api_key=e("GROQ_API_KEY", "").strip(),
        asr_model=e("GROQ_ASR_MODEL", "whisper-large-v3-turbo"),
        llm_model=e("GROQ_LLM_MODEL", "llama-3.3-70b-versatile"),
        asr_backend=e("ASR_BACKEND", "groq").lower(),
        system_prompt=e("SYSTEM_PROMPT") or DEFAULT_PROMPT,
        data_dir=Path(e("DATA_DIR", "./data")).resolve(),
        retention_days=_int("RETENTION_DAYS", 14),
        request_timeout_s=_int("REQUEST_TIMEOUT_S", 30),
        history_turns=_int("HISTORY_TURNS", 6),
        sample_interval_s=max(1, _int("SAMPLE_INTERVAL_S", 5)),
        app_password=e("APP_PASSWORD", ""),
    )
