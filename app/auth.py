"""Password checks and permissions. Pure logic (no Streamlit) so it can be tested."""
from __future__ import annotations

import hmac
import time

from config import Settings


class Throttle:
    """Lock password entry for a while after too many wrong guesses (shared by all visitors)."""

    def __init__(self, max_fails: int = 8, lock_s: int = 60):
        self.max_fails, self.lock_s = max_fails, lock_s
        self.fails, self.until = 0, 0.0

    def locked_for(self, now: float | None = None) -> int:
        """Seconds left on the lock; 0 if entry is allowed."""
        now = time.time() if now is None else now
        return max(0, int(self.until - now + 0.999))

    def failure(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.fails += 1
        if self.fails >= self.max_fails:
            self.until, self.fails = now + self.lock_s, 0

    def success(self) -> None:
        self.fails = 0


def _same(a: str, b: str) -> bool:
    return bool(b) and hmac.compare_digest(a.encode(), b.encode())


def role_for(entered: str, s: Settings) -> str | None:
    """'admin', 'user' or None for a wrong password."""
    is_admin, is_user = _same(entered, s.admin_password), _same(entered, s.app_password)
    return "admin" if is_admin else ("user" if is_user else None)


def can_manage_voices(s: Settings, is_admin: bool) -> bool:
    """On a public link only the admin may create, replace, delete or set the default voice."""
    return is_admin or not s.exposed
