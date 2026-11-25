"""Groq chat completion with conversation history."""
from __future__ import annotations

from asr import _groq_client
from config import get_settings
from errors import AppError, from_groq


_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "safeguard")


def _chat_models(client) -> str:
    """IDs of the chat models available to this API key (used to help fix a wrong GROQ_LLM_MODEL)."""
    try:
        ids = [m.id for m in client.models.list().data if getattr(m, "active", True)]
    except Exception:
        return ""
    return ", ".join(sorted(i for i in ids if not any(k in i.lower() for k in _NOT_CHAT))[:12])


def answer(history: list[dict], question: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """history: [{'role': 'user'|'assistant', 'content': str}, ...]. Returns (answer, usage dict)."""
    s = get_settings()
    client = _groq_client(s.request_timeout_s)
    recent = history[-2 * s.history_turns :] if s.history_turns > 0 else []
    messages = [{"role": "system", "content": system_prompt or s.system_prompt}, *recent,
                {"role": "user", "content": question}]
    try:
        kwargs: dict = {"model": s.llm_model, "messages": messages, "temperature": 0.6, "max_completion_tokens": 400}
        if s.llm_model.startswith("openai/gpt-oss"):
            # GPT-OSS models reason before answering, and reasoning tokens count toward the limit. Use low effort,
            # leave room for the answer, and keep the reasoning out of the reply. extra_body works on any SDK version.
            kwargs["max_completion_tokens"] = 1500
            kwargs["extra_body"] = {"reasoning_effort": "low", "include_reasoning": False}
        resp = client.chat.completions.create(**kwargs)
    except AppError:
        raise
    except Exception as e:
        hint = _chat_models(client) if type(e).__name__ == "NotFoundError" else None
        raise from_groq(e, "llm", s.llm_model, hint) from e
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise AppError("llm", "Groq returned an empty answer.")
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None)}
    return text, usage
