"""System + voice analytics. Live values come from psutil (no DB); history from memory/SQLite."""
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import monitor
import store
import ui

ui.setup_page("Dashboard", "📊")
st.title("📊 Dashboard")

RANGES = {"15 min": 900, "1 hour": 3600, "24 hours": 86400}
# Fixed colour per series (identity never changes with filtering)
C = {"sys": "#2a6fdb", "proc": "#e8833a", "up": "#2a9d8f", "down": "#7b5ea7",
     "asr": "#2a6fdb", "llm": "#e8833a", "tts": "#2a9d8f", "total": "#7b5ea7"}


def mb(b):
    return "n/a" if b is None else (f"{b / 1024**3:.2f} GB" if b >= 1024**3 else f"{b / 1024**2:.0f} MB")


def rate(bps):
    return "n/a" if bps is None else f"{bps / 1024:.1f} KB/s"


def pct(v):
    return "n/a" if v is None else f"{v:.0f}%"


def secs(ms):
    return "n/a" if ms is None else f"{ms / 1000:.2f} s"


def to_df(rows):
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s")  # UTC
    return df


def line_chart(df, series, ytitle, yfmt=None):
    """series: {column: (label, colour)}. Single y-axis, thin lines, legend for 2+ series."""
    fig = go.Figure()
    for col, (label, colour) in series.items():
        if col in df and df[col].notna().any():
            fig.add_scatter(x=df["time"], y=df[col], name=label, mode="lines+markers" if len(df) < 40 else "lines",
                            line=dict(width=2, color=colour), marker=dict(size=6),
                            hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                      showlegend=len(series) > 1, legend=dict(orientation="h", y=1.15),
                      yaxis=dict(title=ytitle, rangemode="tozero"), xaxis=dict(title="time (UTC)"))
    st.plotly_chart(fig, use_container_width=True)


def system_tab(rng_s):
    d = monitor.snapshot()
    st.caption(f"ℹ️ {d['env']}")
    c = st.columns(6)
    c[0].metric("CPU (system)", pct(d["cpu"]["pct"]))
    c[1].metric("CPU (this app)", pct(d["cpu"]["proc_pct"]), help="Share of all cores used by this process")
    c[2].metric("Memory (system)", pct(d["mem"]["pct"]), f"{mb(d['mem']['used'])} of {mb(d['mem']['total'])}", delta_color="off")
    c[3].metric("App RSS", mb(d["proc"]["rss"]), f"peak {mb(d['mem']['peak_rss'])}", delta_color="off")
    c[4].metric("Disk", "n/a" if not d["disk"] else pct(d["disk"]["pct"]),
                None if not d["disk"] else f"{mb(d['disk']['free'])} free", delta_color="off")
    up = d["proc"]["uptime_s"]
    c[5].metric("Uptime", f"{int(up // 3600)}h {int(up % 3600 // 60)}m")

    hist = to_df(monitor.history(rng_s))
    if hist.empty:
        st.info("Collecting samples… charts appear after a few seconds.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("**CPU utilisation (%)**")
            line_chart(hist, {"cpu": ("System", C["sys"]), "proc_cpu": ("This app", C["proc"])}, "% of all cores")
            st.caption(f"In range: avg {hist['cpu'].mean():.0f}% · peak {hist['cpu'].max():.0f}% (system); "
                       f"peak {hist['proc_cpu'].max():.0f}% (app)")
        with b:
            st.markdown("**Memory utilisation (% of total RAM)**")
            line_chart(hist, {"mem_pct": ("System", C["sys"]), "proc_mem_pct": ("This app", C["proc"])}, "% of RAM")
        st.markdown("**Network throughput, system-wide (KB/s)**")
        net = hist.assign(up=hist["net_up_bps"] / 1024, down=hist["net_down_bps"] / 1024)
        line_chart(net, {"up": ("Sent", C["up"]), "down": ("Received", C["down"])}, "KB/s")
        st.caption("Network figures cover the whole machine/container; per-process network isn't available.")

    with st.expander("Details"):
        a, b, c3 = st.columns(3)
        with a:
            st.markdown("**CPU**")
            st.write(f"Cores visible: {d['cpu']['cores']}")
            if d["container"]["cpu_limit"]:
                st.write(f"Container CPU limit: {d['container']['cpu_limit']:.1f} cores")
            st.write("Per core: " + ", ".join(f"{x:.0f}%" for x in d["cpu"]["per_core"]))
            st.markdown("**Process**")
            p = d["proc"]
            st.write(f"PID {p['pid']} · {p['status']} · {p['threads']} threads")
            st.write(f"RSS {mb(p['rss'])} · VMS {mb(p['vms'])}")
        with b:
            st.markdown("**Memory**")
            m = d["mem"]
            st.write(f"Available {mb(m['available'])} · swap {m['swap_pct']:.0f}%")
            st.write(f"Peak system {m['peak_pct']:.0f}% · peak app RSS {mb(m['peak_rss'])}")
            if d["container"]["mem_limit"]:
                st.write(f"Container limit {mb(d['container']['mem_limit'])}, in use {mb(d['container']['mem_used'])}")
            st.markdown("**GPU**")
            g = d["gpu"]
            st.write("Unavailable (none detected)" if not g else
                     f"{g['name']} · {g['util_pct']:.0f}% · {g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f} MB")
        with c3:
            st.markdown("**Disk**")
            dk = d["disk"]
            if dk:
                st.write(f"{mb(dk['used'])} used of {mb(dk['total'])} ({dk['pct']:.0f}%)")
            st.write(f"App data dir: {mb(d['data_dir_bytes'])}")
            st.write(f"Model cache: {mb(d['model_cache_bytes'])}")
            st.markdown("**Network totals since boot (system-wide)**")
            st.write(f"Sent {mb(d['net']['sent'])} · received {mb(d['net']['recv'])}")


def voice_tab(rng_s):
    since = time.time() - rng_s
    rows = store.query_requests(since)
    if store.last_error():
        st.warning(f"Analytics storage unavailable ({store.last_error()}). Live system metrics still work.")
    if not rows:
        st.info("No voice requests in this time range yet. Ask the assistant a question on the Assistant page.")
        return
    s = store.summarize(rows)
    c = st.columns(5)
    c[0].metric("Questions", s["total"])
    c[1].metric("Successful", s["success"])
    c[2].metric("Failed", s["failed"])
    c[3].metric("Audio generated", "n/a" if s["total_audio_out_s"] is None else f"{s['total_audio_out_s']:.0f} s")
    c[4].metric("Tokens used", "n/a" if s["total_tokens"] is None else f"{s['total_tokens']:,}")
    c = st.columns(6)
    c[0].metric("Avg response", secs(s["avg_total_ms"]))
    c[1].metric("Median response", secs(s["median_total_ms"]))
    c[2].metric("P95 response", secs(s["p95_total_ms"]), help="Shown once there are at least 20 successful requests")
    c[3].metric("Avg transcription", secs(s["avg_asr_ms"]))
    c[4].metric("Avg Groq LLM", secs(s["avg_llm_ms"]))
    c[5].metric("Avg TTS", secs(s["avg_tts_ms"]))

    df = to_df(rows)
    for k in ("asr_ms", "llm_ms", "tts_ms", "total_ms"):
        df[k] = df[k] / 1000
    ok = df[df["status"] == "success"]
    a, b = st.columns(2)
    with a:
        st.markdown("**Latency per request (s)**")
        if ok.empty:
            st.caption("No successful requests yet.")
        else:
            line_chart(ok, {"asr_ms": ("Transcription", C["asr"]), "llm_ms": ("Groq LLM", C["llm"]),
                            "tts_ms": ("TTS", C["tts"]), "total_ms": ("Total", C["total"])}, "seconds")
    with b:
        st.markdown("**Requests over time**")
        bucket = "1min" if rng_s <= 900 else ("5min" if rng_s <= 3600 else "1h")
        vol = df.set_index("time").groupby([pd.Grouper(freq=bucket), "status"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for status, colour in (("success", C["tts"]), ("failed", C["llm"])):
            if status in vol:
                fig.add_bar(x=vol.index, y=vol[status], name=status.capitalize(), marker_color=colour)
        fig.update_layout(height=260, barmode="stack", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15),
                          yaxis=dict(title="requests", dtick=1), xaxis=dict(title="time (UTC)"))
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.markdown("**Model usage**")
        usage = df.groupby(["asr_model", "llm_model", "tts_model"]).size().reset_index(name="requests")
        st.dataframe(usage, hide_index=True, use_container_width=True)
    with b:
        st.markdown("**Recent errors**")
        errs = store.query_errors(since, 10)
        if errs:
            e = pd.DataFrame(errs)
            e["time"] = pd.to_datetime(e["ts"], unit="s")
            st.dataframe(e[["time", "category", "message"]], hide_index=True, use_container_width=True)
        else:
            st.caption("No errors in this range.")

    st.markdown("**Recent requests**")
    recent = df.sort_values("ts", ascending=False).head(15)[
        ["time", "id", "status", "error_cat", "asr_ms", "llm_ms", "tts_ms", "total_ms", "in_audio_s", "out_audio_s"]]
    st.dataframe(recent.round(2), hide_index=True, use_container_width=True)


@st.fragment(run_every=5)
def page():
    label = st.radio("Time range", list(RANGES), horizontal=True, key="range")
    rng_s = RANGES[label]
    t1, t2 = st.tabs(["System", "Voice & usage"])
    with t1:
        system_tab(rng_s)
    with t2:
        voice_tab(rng_s)


page()
