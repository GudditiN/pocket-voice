"""Live resource monitoring with psutil. Independent of the database (history is also kept in memory)."""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import psutil

import store
from config import get_settings

log = logging.getLogger(__name__)

_proc = psutil.Process()
_history: deque[dict] = deque(maxlen=720)  # 1 hour at 5 s
_peak = {"cpu": 0.0, "proc_cpu": 0.0, "proc_rss": 0, "mem_pct": 0.0}
_prev_net: tuple[float, int, int] | None = None
_started = False
_start_lock = threading.Lock()
_dir_cache: dict[str, tuple[float, int]] = {}


def environment_label() -> str:
    if os.environ.get("CODESPACES") == "true":
        return "GitHub Codespaces: figures describe this container/VM, not your laptop"
    if Path("/.dockerenv").exists():
        return "Docker container: figures describe the container/host VM"
    return f"Local machine ({platform.system()})"


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def container_limits() -> dict:
    """cgroup memory/CPU limits when running in a container (psutil reports host-wide values there)."""
    out: dict = {"mem_limit": None, "mem_used": None, "cpu_limit": None}
    lim = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read("/sys/fs/cgroup/memory.current") or _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if lim and lim.isdigit() and int(lim) < 1 << 60:
        out["mem_limit"] = int(lim)
    if used and used.isdigit():
        out["mem_used"] = int(used)
    cpu = _read("/sys/fs/cgroup/cpu.max")
    if cpu:
        q, _, p = cpu.partition(" ")
        if q.isdigit() and p.isdigit() and int(p) > 0:
            out["cpu_limit"] = int(q) / int(p)
    return out


def dir_size(path: Path, ttl: float = 30.0) -> int | None:
    """Total bytes under path, cached for ttl seconds. None if the path does not exist."""
    key = str(path)
    now = time.time()
    if key in _dir_cache and now - _dir_cache[key][0] < ttl:
        return _dir_cache[key][1]
    if not path.exists():
        return None
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    _dir_cache[key] = (now, total)
    return total


def gpu_info() -> dict | None:
    """Optional GPU stats via nvidia-smi; None when unavailable (never fake values)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        name, util, used, total = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
        return {"name": name, "util_pct": float(util), "mem_used_mb": float(used), "mem_total_mb": float(total)}
    except Exception:
        return None


def sample() -> dict:
    """Collect one sample (cheap, non-blocking). Called by the sampler thread."""
    global _prev_net
    now = time.time()
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    with _proc.oneshot():
        proc_cpu = _proc.cpu_percent(interval=None) / (psutil.cpu_count() or 1)  # normalise to 0-100 of all cores
        rss = _proc.memory_info().rss
    net = psutil.net_io_counters()
    up = down = None
    if _prev_net:
        dt = now - _prev_net[0]
        if dt > 0:
            up, down = (net.bytes_sent - _prev_net[1]) / dt, (net.bytes_recv - _prev_net[2]) / dt
    _prev_net = (now, net.bytes_sent, net.bytes_recv)
    s = {"ts": now, "cpu": cpu, "proc_cpu": proc_cpu, "mem_pct": vm.percent, "proc_mem_pct": rss / vm.total * 100,
         "proc_rss": rss, "net_sent": net.bytes_sent, "net_recv": net.bytes_recv, "net_up_bps": up, "net_down_bps": down}
    _peak["cpu"] = max(_peak["cpu"], cpu)
    _peak["proc_cpu"] = max(_peak["proc_cpu"], proc_cpu)
    _peak["proc_rss"] = max(_peak["proc_rss"], rss)
    _peak["mem_pct"] = max(_peak["mem_pct"], vm.percent)
    return s


def snapshot() -> dict:
    """Everything the System tab shows, collected live."""
    s = get_settings()
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    with _proc.oneshot():
        mi = _proc.memory_info()
        info = {"pid": _proc.pid, "status": _proc.status(), "threads": _proc.num_threads(),
                "uptime_s": time.time() - _proc.create_time(), "rss": mi.rss, "vms": mi.vms}
    try:
        disk = shutil.disk_usage(s.data_dir if s.data_dir.exists() else Path.cwd())
    except OSError:
        disk = None
    net = psutil.net_io_counters()
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    last = _history[-1] if _history else {}
    return {
        "env": environment_label(),
        "cpu": {"pct": last.get("cpu"), "proc_pct": last.get("proc_cpu"), "cores": psutil.cpu_count(),
                "per_core": psutil.cpu_percent(interval=None, percpu=True), "peak": _peak["cpu"]},
        "mem": {"total": vm.total, "available": vm.available, "used": vm.total - vm.available, "pct": vm.percent,
                "swap_pct": sw.percent, "peak_pct": _peak["mem_pct"], "peak_rss": _peak["proc_rss"]},
        "container": container_limits(),
        "disk": None if disk is None else {"total": disk.total, "used": disk.used, "free": disk.free,
                                           "pct": disk.used / disk.total * 100},
        "data_dir_bytes": dir_size(s.data_dir), "model_cache_bytes": dir_size(hf),
        "net": {"sent": net.bytes_sent, "recv": net.bytes_recv, "up_bps": last.get("net_up_bps"),
                "down_bps": last.get("net_down_bps")},
        "proc": info,
        "gpu": gpu_info(),
    }


def history(range_s: int) -> list[dict]:
    """Samples for the last range_s seconds: in-memory when it covers the range, else from SQLite."""
    since = time.time() - range_s
    mem = [x for x in _history if x["ts"] >= since]
    covered = bool(mem) and mem[0]["ts"] <= since + max(60, get_settings().sample_interval_s * 3)
    if covered or range_s <= 900:
        return mem
    db = store.query_samples(since)
    return db if len(db) > len(mem) else mem


def _loop():
    interval = get_settings().sample_interval_s
    last_db, last_clean = 0.0, time.time()
    sample()  # prime cpu_percent counters
    while True:
        time.sleep(interval)
        try:
            s = sample()
            _history.append(s)
            if s["ts"] - last_db >= 30:  # bounded DB writes: one row per 30 s
                store.insert_sample(s)
                last_db = s["ts"]
            if s["ts"] - last_clean >= 6 * 3600:
                store.cleanup()
                last_clean = s["ts"]
        except Exception:
            log.exception("sampler iteration failed")


def start_sampler() -> None:
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
        threading.Thread(target=_loop, name="resource-sampler", daemon=True).start()
