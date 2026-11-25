"""Groq chat completion with conversation history."""
from __future__ import annotations

from asr import _groq_client
from config import get_settings
from errors import AppError, from_groq


def answer(history: list[dict], question: str, system_prompt: str | None = None) -> tuple[str, dict]:
    """history: [{'role': 'user'|'assistant', 'content': str}, ...]. Returns (answer, usage dict)."""
    s = get_settings()
    client = _groq_client(s.request_timeout_s)
    recent = history[-2 * s.history_turns :] if s.history_turns > 0 else []
    messages = [{"role": "system", "content": system_prompt or s.system_prompt}, *recent,
                {"role": "user", "content": question}]
    try:
        resp = client.chat.completions.create(model=s.llm_model, messages=messages, max_tokens=300, temperature=0.6)
    except AppError:
        raise
    except Exception as e:
        raise from_groq(e, "llm") from e
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise AppError("llm", "Groq returned an empty answer.")
    u = getattr(resp, "usage", None)
    usage = {"prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None)}
    return text, usage
