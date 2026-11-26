import hashlib
import socket
import sys
from pathlib import Path

import pytest

import auth
from config import get_settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import expose  # noqa: E402


def settings(monkeypatch, app="", admin="", exposed="0"):
    monkeypatch.setenv("APP_PASSWORD", app)
    monkeypatch.setenv("ADMIN_PASSWORD", admin)
    monkeypatch.setenv("POCKET_EXPOSED", exposed)
    return get_settings()


def test_roles(monkeypatch):
    s = settings(monkeypatch, app="user-password-1", admin="admin-password-1")
    assert auth.role_for("user-password-1", s) == "user"
    assert auth.role_for("admin-password-1", s) == "admin"
    assert auth.role_for("wrong", s) is None


def test_empty_password_never_matches(monkeypatch):
    s = settings(monkeypatch)
    assert auth.role_for("", s) is None


def test_voice_management_is_locked_only_on_public_link(monkeypatch):
    assert auth.can_manage_voices(settings(monkeypatch), is_admin=False)
    public = settings(monkeypatch, exposed="1")
    assert not auth.can_manage_voices(public, is_admin=False)
    assert auth.can_manage_voices(public, is_admin=True)


def test_throttle_locks_then_releases():
    t = auth.Throttle(max_fails=3, lock_s=60)
    for _ in range(2):
        t.failure(now=100)
    assert t.locked_for(now=100) == 0
    t.failure(now=100)
    assert t.locked_for(now=101) == 59 and t.locked_for(now=161) == 0


def test_passwords_generated_when_missing():
    app, admin, generated = expose.resolve_passwords("", "")
    assert generated == ["APP_PASSWORD", "ADMIN_PASSWORD"]
    assert len(app) >= expose.MIN_PASSWORD and len(admin) >= expose.MIN_PASSWORD and app != admin


@pytest.mark.parametrize("app,admin", [("short", ""), ("", "short"), ("same-password-123", "same-password-123")])
def test_weak_or_equal_passwords_refused(app, admin):
    with pytest.raises(SystemExit):
        expose.resolve_passwords(app, admin)


def test_banner_hides_passwords_you_set_yourself():
    text = expose.banner("https://x.trycloudflare.com", "my-own-secret-1", "generated-admin-1", ["ADMIN_PASSWORD"])
    assert "my-own-secret-1" not in text and "generated-admin-1" in text and "(your APP_PASSWORD)" in text


def test_port_in_use_detects_a_listener():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert expose.port_in_use(port)
    assert not expose.port_in_use(port)


def test_public_url_pattern():
    line = "2026 INF |  https://quiet-blue-fox-12.trycloudflare.com   |"
    assert expose.URL_RE.search(line).group(0) == "https://quiet-blue-fox-12.trycloudflare.com"
    assert expose.URL_RE.search('Post "https://api.trycloudflare.com/tunnel": timeout') is None


def test_download_keeps_file_only_with_matching_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"pretend binary")
    good = hashlib.sha256(b"pretend binary").hexdigest()
    expose.download(src.as_uri(), tmp_path / "ok.bin", good)
    assert (tmp_path / "ok.bin").read_bytes() == b"pretend binary"
    with pytest.raises(SystemExit):
        expose.download(src.as_uri(), tmp_path / "bad.bin", "0" * 64)
    assert not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.part").exists()


def test_checksums_are_well_formed():
    for _, sha in expose.ASSETS.values():
        assert len(sha) == 64 and int(sha, 16) >= 0
