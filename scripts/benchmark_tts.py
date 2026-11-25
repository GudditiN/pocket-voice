"""Measure real PocketTTS performance. Run in Codespaces:

    uv run python scripts/benchmark_tts.py --ref path/to/my_voice.wav

Reports model load time, voice-state time, profile reload time, per-sentence latency and
real-time factor (RTF = generation time / audio duration; < 1 is faster than real time),
time to first audio chunk (if streaming is available), RSS before/after, and CPU use.
Results are printed and saved to $DATA_DIR/benchmark.json (git-ignored).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import psutil  # noqa: E402

import tts  # noqa: E402
from config import get_settings  # noqa: E402

SENTENCES = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog, and then takes a short nap in the sun.",
    "Photosynthesis is the process plants use to turn light, water and carbon dioxide into sugar and oxygen.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference WAV (your own voice, 10-20 s)")
    ap.add_argument("--runs", type=int, default=2, help="repeats per sentence")
    args = ap.parse_args()

    proc = psutil.Process()
    mb = lambda b: round(b / 1024**2)  # noqa: E731
    out = {"cores": psutil.cpu_count(), "total_ram_mb": mb(psutil.virtual_memory().total)}
    out["rss_before_mb"] = mb(proc.memory_info().rss)

    tts._get_model()
    out["model_load_s"] = round(tts.last_timings["model_load_s"], 2)
    out["rss_after_load_mb"] = mb(proc.memory_info().rss)

    info = tts.build_profile(Path(args.ref).read_bytes())
    out["voice_state_s"] = round(info["voice_state_s"], 2)
    tts._state = None  # force a reload from the saved file, as after an app restart
    t = time.perf_counter()
    tts._load_state()
    out["profile_reload_s"] = round(time.perf_counter() - t, 2)

    rows = []
    cpu0 = proc.cpu_times()
    wall0 = time.perf_counter()
    for text in SENTENCES:
        for _ in range(args.runs):
            t = time.perf_counter()
            _, secs = tts.synthesize(text)
            took = time.perf_counter() - t
            rows.append({"chars": len(text), "audio_s": round(secs, 2), "gen_s": round(took, 2), "rtf": round(took / secs, 2)})
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    out["synthesis"] = rows
    out["cpu_utilisation_during_tts_pct_of_one_core"] = round(
        ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / wall * 100)
    out["rss_peak_after_tts_mb"] = mb(proc.memory_info().rss)

    # Time to first audio chunk (streaming API), best effort
    try:
        model, state = tts._get_model(), tts._load_state()
        t = time.perf_counter()
        next(iter(model.generate_audio_stream(state, SENTENCES[1])))
        out["time_to_first_audio_s"] = round(time.perf_counter() - t, 2)
    except Exception as e:  # API differences across versions
        out["time_to_first_audio_s"] = f"unavailable ({type(e).__name__})"

    # Idle CPU for 5 s after synthesis
    psutil.cpu_percent(interval=None)
    proc.cpu_percent(interval=None)
    time.sleep(5)
    out["idle_process_cpu_pct"] = proc.cpu_percent(interval=None)

    text = json.dumps(out, indent=2)
    print(text)
    dest = get_settings().data_dir / "benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"\nSaved to {dest}", file=sys.stderr)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
