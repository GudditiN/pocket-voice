"""Play audio pieces in the browser as they arrive, one after another without gaps (Web Audio).

Each piece is sent as a tiny script. A short filler ("let me check") is held back briefly and dropped if the
real answer arrives first. Starting a new turn stops whatever was still playing.
"""
from __future__ import annotations

import base64
import json

import streamlit as st

FILLER_DELAY_S = 0.7  # a filler is only heard if the answer isn't ready within this time

_JS = """
(() => {
  const m = __MESSAGE__;
  const w = window;
  const P = w.pvPlayer || (w.pvPlayer = {ctx: null, turn: null, nodes: [], next: 0, seen: new Set(),
                                          chain: Promise.resolve(), filler: null, played: []});
  if (!P.ctx) P.ctx = new (w.AudioContext || w.webkitAudioContext)();
  if (P.turn !== m.turn) {
    P.nodes.forEach(n => { try { n.stop(); } catch (e) {} });
    Object.assign(P, {turn: m.turn, nodes: [], next: 0, filler: null, played: [], chain: Promise.resolve()});
  }
  if (P.seen.has(m.id)) return;
  P.seen.add(m.id);
  const bytes = Uint8Array.from(atob(m.audio), c => c.charCodeAt(0)).buffer;
  P.chain = P.chain.then(async () => {
    if (m.turn !== P.turn) return;
    await P.ctx.resume();
    const buffer = await P.ctx.decodeAudioData(bytes);
    const source = P.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(P.ctx.destination);
    const now = P.ctx.currentTime;
    if (m.kind !== "filler" && P.filler) {
      if (P.filler.at > now) { try { P.filler.source.stop(); } catch (e) {} P.next = 0; }
      P.filler = null;
    }
    const at = Math.max(now + (m.kind === "filler" ? m.delay : 0.03), P.next);
    source.start(at);
    P.next = at + buffer.duration;
    P.nodes.push(source);
    P.played.push({kind: m.kind, at: at, seconds: buffer.duration});
    if (m.kind === "filler") P.filler = {source: source, at: at};
  }).catch(e => console.warn("pocket-voice audio", e));
})();
"""


_UNLOCK = """
(() => {
  const w = window;
  if (w.pvUnlock) return;
  // Browsers (Safari especially) only allow sound after a tap or key press: switch audio on at the first one.
  w.pvUnlock = () => {
    const P = w.pvPlayer || (w.pvPlayer = {ctx: null, turn: null, nodes: [], next: 0, seen: new Set(),
                                           chain: Promise.resolve(), filler: null, played: []});
    if (!P.ctx) P.ctx = new (w.AudioContext || w.webkitAudioContext)();
    if (P.ctx.state !== "running") P.ctx.resume();
  };
  ["pointerdown", "keydown", "touchstart"].forEach(e => document.addEventListener(e, w.pvUnlock, true));
})();
"""


def prepare() -> None:
    """Call once per page render, before any audio is pushed."""
    st.html(f"<script>{_UNLOCK}</script>", unsafe_allow_javascript=True)


def push(wav: bytes, turn: str, number: int, kind: str = "speech") -> None:
    """Queue one WAV piece for playback. Call it while the answer is still being produced."""
    message = {"turn": turn, "id": f"{turn}-{kind}-{number}", "kind": kind, "delay": FILLER_DELAY_S,
               "audio": base64.b64encode(wav).decode("ascii")}  # base64 and a hex turn id: safe inside the script
    st.html(f"<script>{_JS.replace('__MESSAGE__', json.dumps(message))}</script>", unsafe_allow_javascript=True)
