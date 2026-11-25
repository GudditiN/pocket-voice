"""Shared page setup: page config, optional password gate, one-time background start."""
from __future__ import annotations

import hmac

import streamlit as st

import monitor
import store
from config import get_settings


@st.cache_resource
def _boot() -> bool:
    store.init()
    store.cleanup()
    monitor.start_sampler()
    return True


def setup_page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · Pocket Voice", page_icon=icon, layout="wide")
    pw = get_settings().app_password
    if pw and not st.session_state.get("authed"):
        st.title("Pocket Voice")
        entered = st.text_input("Password", type="password")
        if entered and hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["authed"] = True
            st.rerun()
        elif entered:
            st.error("Wrong password.")
        st.stop()
    _boot()
    st.sidebar.caption("🔊 Speech is **AI-generated** in a voice profile created from a recording you provided.")
    if store.last_error():
        st.sidebar.warning("Analytics storage unavailable; live features still work.")
