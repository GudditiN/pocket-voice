"""One exception type carrying a category that is stored in analytics and shown in the UI."""
from __future__ import annotations


class AppError(Exception):
    """category: missing_key, auth, rate_limit, timeout, network, bad_audio, asr, llm, tts, no_profile, disk, unknown"""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


def from_groq(exc: Exception, stage: str, model: str | None = None, hint: str | None = None) -> AppError:
    """Translate a groq SDK exception into an AppError. stage is 'asr' or 'llm'.

    model: the model ID that was requested. hint: comma-separated model IDs available to the key (for 404s).
    """
    import groq

    if isinstance(exc, groq.NotFoundError):
        var = "GROQ_ASR_MODEL" if stage == "asr" else "GROQ_LLM_MODEL"
        msg = f"Groq could not find the model {model!r}." if model else "Groq returned 404 (not found)."
        if hint:
            msg += f" Models available to your key: {hint}."
        return AppError("model_not_found", f"{msg} Set {var} to a valid model ID and restart the app.")

    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return AppError("auth", "Groq rejected the API key. Check GROQ_API_KEY.")
    if isinstance(exc, groq.RateLimitError):
        return AppError("rate_limit", "Groq rate limit reached. Wait a moment and try again.")
    if isinstance(exc, groq.APITimeoutError):  # subclass of APIConnectionError: check first
        return AppError("timeout", f"Groq {stage.upper()} request timed out.")
    if isinstance(exc, groq.APIConnectionError):
        return AppError("network", "Could not reach Groq. Check the network connection.")
    if isinstance(exc, groq.BadRequestError) and stage == "asr":
        return AppError("bad_audio", "Groq could not read this audio. Try recording again.")
    return AppError(stage, f"Groq {stage.upper()} request failed: {type(exc).__name__}")
