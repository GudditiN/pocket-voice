"""Environment-based settings. Read fresh on each call so tests can monkeypatch env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

def load_dotenv(paths: list[Path] | None = None) -> list[str]:
    """Load KEY=VALUE lines from .env into os.environ. Returns the names of the variables it set.

    Variables that are already set (for example Codespaces secrets) are never overridden, and blank values are
    skipped so a bare `KEY=` doesn't hide a default. Lines starting with # are comments; `export KEY=...` and
    quoted values work. Values are never logged.
    """
    if paths is None:
        paths = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    loaded: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, sep, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if not sep or not key.isidentifier():
                continue
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if not val or os.environ.get(key):
                continue
            os.environ[key] = val
            loaded.append(key)
    return loaded


if not os.environ.get("POCKET_VOICE_NO_DOTENV"):  # tests set this so a real .env can't change their behaviour
    load_dotenv()

DEFAULT_PROMPT = (
    "You are a warm, friendly person talking on a phone call. Reply the way people speak: start with a few natural "
    "words, use contractions, and keep it to one to three short sentences. React to how the person sounds, and when "
    "it helps, end with one short follow-up question, like a real conversation. Do not repeat the question or "
    "announce what you are going to do. Plain text only: no markdown, lists, or emojis."
)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _choice(name: str, allowed: tuple[str, ...], default: str) -> str:
    value = os.environ.get(name, "").strip().lower()
    return value if value in allowed else default


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
    admin_password: str  # on a public link, required to manage voices
    exposed: bool  # set by `./run.sh --expose`
    quantize: bool  # int8 quantization of the TTS model (applied at load, so it needs an app restart)
    denoise: str  # default noise reduction for new voice samples: off, light, medium or strong
    live_data: bool  # default for the "search the web" switch
    tavily_api_key: str  # optional: more dependable web search
    live_region: str  # two-letter country code for news, e.g. IN or US
    app_timezone: str  # used to tell the model today's date and time
    fast_reply: bool  # speak while the answer is still being written
    fillers: bool  # short "let me check" phrases while waiting
    warm_up: bool  # load the voice model at startup
    sniffynose: bool  # answer SniffyNose and pet-shopping questions from sniffynose.com
    nest_allowed: bool  # set only if you own SniffyNest and allow it to be read

    @property
    def voice_dir(self) -> Path:
        return self.data_dir / "voice"

    @property
    def profiles_dir(self) -> Path:
        return self.voice_dir / "profiles"

    @property
    def active_file(self) -> Path:
        return self.voice_dir / "active.txt"

    @property
    def tuning_path(self) -> Path:
        return self.voice_dir / "tuning.json"

    @property
    def legacy_reference_path(self) -> Path:  # single-profile layout used before named profiles
        return self.voice_dir / "reference.wav"

    @property
    def legacy_profile_path(self) -> Path:
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
        llm_model=e("GROQ_LLM_MODEL", "openai/gpt-oss-20b"),
        asr_backend=e("ASR_BACKEND", "groq").lower(),
        system_prompt=e("SYSTEM_PROMPT") or DEFAULT_PROMPT,
        data_dir=Path(e("DATA_DIR", "./data")).resolve(),
        retention_days=_int("RETENTION_DAYS", 14),
        request_timeout_s=_int("REQUEST_TIMEOUT_S", 30),
        history_turns=_int("HISTORY_TURNS", 6),
        sample_interval_s=max(1, _int("SAMPLE_INTERVAL_S", 5)),
        app_password=e("APP_PASSWORD", ""),
        admin_password=e("ADMIN_PASSWORD", ""),
        exposed=e("POCKET_EXPOSED", "0").lower() in ("1", "true", "yes"),
        quantize=e("POCKET_QUANTIZE", "0").lower() in ("1", "true", "yes"),
        denoise=_choice("POCKET_DENOISE", ("off", "light", "medium", "strong"), "off"),
        live_data=e("LIVE_DATA", "1").lower() in ("1", "true", "yes"),
        tavily_api_key=e("TAVILY_API_KEY", "").strip(),
        live_region=(e("LIVE_REGION", "") or "IN").strip().upper()[:2],
        app_timezone=e("APP_TIMEZONE", "") or "Asia/Kolkata",
        fast_reply=e("FAST_REPLY", "1").lower() in ("1", "true", "yes"),
        fillers=e("FILLERS", "1").lower() in ("1", "true", "yes"),
        warm_up=e("POCKET_WARMUP", "1").lower() in ("1", "true", "yes"),
        sniffynose=e("SNIFFYNOSE", "1").lower() in ("1", "true", "yes"),
        nest_allowed=e("SNIFFYNEST_ALLOWED", "0").lower() in ("1", "true", "yes"),
    )
