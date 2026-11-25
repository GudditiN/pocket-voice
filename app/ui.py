"""Shared page setup, theme CSS and small layout helpers."""
from __future__ import annotations

import hmac
import html

import streamlit as st

import monitor
import store
from config import get_settings

# Neutral translucent greys keep this readable in both light and dark themes.
_CSS = """
<style>
.block-container { padding-top: 1.6rem; max-width: 1100px; }
footer, [data-testid="stDecoration"] { visibility: hidden; }
.pv-hero { padding: 1.3rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
  background: linear-gradient(135deg, rgba(79,70,229,.20), rgba(20,184,166,.15));
  border: 1px solid rgba(127,127,127,.22); }
.pv-hero h1 { margin: 0; padding: 0; font-size: 2rem; line-height: 1.2; }
.pv-hero p { margin: .35rem 0 0 0; opacity: .8; font-size: 1.02rem; }
.pv-chip { display: inline-block; padding: .22rem .75rem; margin: 0 .45rem .45rem 0; border-radius: 999px;
  font-size: .85rem; border: 1px solid rgba(127,127,127,.35); background: rgba(127,127,127,.10); }
.pv-ok { background: rgba(34,197,94,.16); border-color: rgba(34,197,94,.55); }
.pv-warn { background: rgba(245,158,11,.16); border-color: rgba(245,158,11,.6); }
.pv-bad { background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.55); }
.pv-flow { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: .2rem 0 1rem 0; opacity: .9; }
.pv-flow span.step { padding: .25rem .7rem; border-radius: 10px; background: rgba(127,127,127,.10);
  border: 1px solid rgba(127,127,127,.25); font-size: .9rem; }
.pv-flow span.arrow { opacity: .5; }
[data-testid="stMetric"] { background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.22);
  border-radius: 12px; padding: .7rem .9rem; }
[data-testid="stChatMessage"] { background: rgba(127,127,127,.07); border-radius: 14px; padding: .8rem 1rem; }
[data-testid="stSidebar"] .pv-brand { font-weight: 700; font-size: 1.15rem; margin-bottom: .2rem; }
.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 600; }
</style>
"""


@st.cache_resource
def _boot() -> bool:
    store.init()
    store.cleanup()
    monitor.start_sampler()
    return True


def hero(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='pv-hero'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
                unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> None:
    """items: [(text, kind)] where kind is 'ok', 'warn', 'bad' or ''."""
    st.markdown("".join(f"<span class='pv-chip pv-{k}'>{html.escape(t)}</span>" for t, k in items),
                unsafe_allow_html=True)


def flow(steps: list[str]) -> None:
    body = "<span class='arrow'>→</span>".join(f"<span class='step'>{html.escape(s)}</span>" for s in steps)
    st.markdown(f"<div class='pv-flow'>{body}</div>", unsafe_allow_html=True)


def setup_page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · Pocket Voice", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    pw = get_settings().app_password
    if pw and not st.session_state.get("authed"):
        hero("🎙️ Pocket Voice", "Enter the password to continue.")
        entered = st.text_input("Password", type="password")
        if entered and hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["authed"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password.")
        st.stop()
    _boot()
    st.sidebar.markdown("<div class='pv-brand'>🎙️ Pocket Voice</div>", unsafe_allow_html=True)
    st.sidebar.caption("Speech is **AI-generated** in a voice profile made from a recording you provided.")
    if store.last_error():
        st.sidebar.warning("Analytics storage unavailable; live features still work.")
