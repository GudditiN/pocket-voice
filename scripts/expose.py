"""Start the app and share it through a temporary public link. Run with `./run.sh --expose`.

Uses a Cloudflare "quick tunnel" (no account needed). Safety rules:
  * the app always requires a password; managing voices needs a second, admin password
  * missing passwords are generated for this run and printed in your terminal
  * it refuses to start if something else already uses the port (so an unprotected app is never published)
  * the link stops working when you press Ctrl+C
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from config import get_settings  # noqa: E402  (also loads .env)

MIN_PASSWORD = 12
URL_RE = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")

# Official release assets and their published SHA-256 digests (github.com/cloudflare/cloudflared/releases).
VERSION = "2026.11.2"
ASSETS = {
    ("Linux", "x86_64"): ("cloudflared-linux-amd64", "56f13335143ec0e22772963290683a84a04114a35ec64cbc794a4d39ff9bb8cf"),
    ("Linux", "aarch64"): ("cloudflared-linux-arm64", "04a7b7f08bce85d56b26ffe1fdbfbb21a074495a363d16f0783c40e25b5c5281"),
    ("Darwin", "x86_64"): ("cloudflared-darwin-amd64.tgz", "6153d9ec629174882a3adbff2f7dc5a24b4f3bdd7ccc38eaa84fba26eb890311"),
    ("Darwin", "arm64"): ("cloudflared-darwin-arm64.tgz", "f6d58eeec91ad1ec5fd1f7e4a587303772929b2625be5d24068033851c36ff55"),
}


def fail(message: str):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def resolve_passwords(app_pw: str, admin_pw: str) -> tuple[str, str, list[str]]:
    """Return (app_password, admin_password, names_that_were_generated). Exits on weak or equal passwords."""
    generated = []
    values = {"APP_PASSWORD": app_pw, "ADMIN_PASSWORD": admin_pw}
    for name, value in values.items():
        if not value:
            values[name] = secrets.token_urlsafe(12)
            generated.append(name)
        elif len(value) < MIN_PASSWORD:
            fail(f"{name} is too short for a public link: use at least {MIN_PASSWORD} characters, "
                 "or leave it unset to have one generated.")
    if values["APP_PASSWORD"] == values["ADMIN_PASSWORD"]:
        fail("APP_PASSWORD and ADMIN_PASSWORD must be different.")
    return values["APP_PASSWORD"], values["ADMIN_PASSWORD"], generated


def port_in_use(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def download(url: str, dest: Path, sha256: str) -> None:
    """Download url to dest, keeping the file only if its SHA-256 matches."""
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        fail(f"Could not download {url}: {e}")
    if digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        fail("Downloaded cloudflared does not match the published checksum. Nothing was run.")
    os.replace(tmp, dest)


def find_cloudflared() -> str:
    """An installed cloudflared, or the pinned release downloaded once into .cache/bin/."""
    installed = shutil.which("cloudflared")
    if installed:
        return installed
    bin_dir = ROOT / ".cache" / "bin"
    binary = bin_dir / f"cloudflared-{VERSION}"
    if binary.exists():
        return str(binary)
    key = (platform.system(), platform.machine())
    if key not in ASSETS:
        fail(f"No cloudflared build is configured for {key}. Install cloudflared yourself and run again.")
    asset, sha = ASSETS[key]
    bin_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/{asset}"
    print(f"Downloading cloudflared {VERSION} from {url}\n(one time, 20-40 MB, checksum verified)…", flush=True)
    target = bin_dir / asset
    download(url, target, sha)
    if asset.endswith(".tgz"):
        with tarfile.open(target) as archive:
            source = archive.extractfile("cloudflared")  # only this one file is read from the archive
            if source is None:
                fail("cloudflared was not found inside the downloaded archive.")
            binary.write_bytes(source.read())
        target.unlink()
    else:
        os.replace(target, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return str(binary)


def wait_until_ready(app: subprocess.Popen, port: int, timeout: float = 120) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if app.poll() is not None:
            fail("The app stopped while starting. See the messages above.")
        try:
            with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    time.sleep(0.5)
                    if app.poll() is None:  # make sure it is our app that is answering
                        return
        except OSError:
            pass
        time.sleep(1)
    fail("The app did not become ready in time.")


def banner(url: str, app_pw: str, admin_pw: str, generated: list[str]) -> str:
    def show(name: str, value: str) -> str:
        return value if name in generated else f"(your {name})"

    return "\n".join([
        "", "=" * 68,
        "  PUBLIC LINK (anyone with it can reach the login page):",
        f"  {url}",
        "",
        f"  Password to share        : {show('APP_PASSWORD', app_pw)}",
        f"  Admin password (keep it) : {show('ADMIN_PASSWORD', admin_pw)}",
        "",
        "  People with the password can make your voice say things and use your Groq quota.",
        "  Share only with people you trust. Press Ctrl+C to close the link.",
        "=" * 68, "",
    ])


def stop(processes: list[subprocess.Popen]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Pocket Voice behind a temporary public link.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8501)))
    args = ap.parse_args(argv)

    s = get_settings()
    app_pw, admin_pw, generated = resolve_passwords(s.app_password, s.admin_password)
    if port_in_use(args.port):
        fail(f"Port {args.port} is already in use (is ./run.sh still running?). Stop it, or set PORT to another number.")
    cloudflared = find_cloudflared()

    env = dict(os.environ, APP_PASSWORD=app_pw, ADMIN_PASSWORD=admin_pw, POCKET_EXPOSED="1")
    processes: list[subprocess.Popen] = []

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    try:
        app = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py"),
             "--server.port", str(args.port), "--server.address", "127.0.0.1", "--server.headless", "true",
             "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env)
        processes.append(app)
        wait_until_ready(app, args.port)
        # The tunnel process gets no passwords or API keys.
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("APP_PASSWORD", "ADMIN_PASSWORD", "GROQ_API_KEY", "HF_TOKEN")}
        tunnel = subprocess.Popen(
            [cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{args.port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=clean)
        processes.append(tunnel)

        found = threading.Event()
        recent: list[str] = []

        def watch():  # keep draining the tunnel's output so it never blocks
            for line in tunnel.stdout:
                recent[:] = [*recent[-14:], line.rstrip()]
                match = URL_RE.search(line)
                if match and not found.is_set():
                    found.set()
                    print(banner(match.group(0), app_pw, admin_pw, generated), flush=True)

        threading.Thread(target=watch, daemon=True).start()
        started = time.time()
        while app.poll() is None and tunnel.poll() is None:
            if not found.is_set() and time.time() - started > 60:
                print("\n".join(recent), file=sys.stderr)
                fail("No public link after 60 s. Check the network and try again.")
            time.sleep(0.5)
        if tunnel.poll() is not None:
            print("\n".join(recent), file=sys.stderr)
            fail("The tunnel stopped.")
        fail("The app stopped.")
    except KeyboardInterrupt:
        print("\nClosing the public link…")
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    sys.exit(main())
