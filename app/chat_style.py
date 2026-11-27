"""Look of the Assistant's chat box: rounded composer, large mic on the right, 'Listening' animation.

Targets Streamlit's data-testid attributes (checked against Streamlit 1.67). If a future version renames them,
the chat box still works and only falls back to the default look.
"""
import html

import streamlit as st

MIC_SIZE_PX = 52  # change this to make the microphone button bigger or smaller

_CSS = """
<style>
:root { --pv-mic: __MIC__px; --pv-accent: #4f46e5; --pv-live: #ef4444; }

/* audio pieces of fast replies are scripts only: take no space */
[class*="st-key-pvsound"] { display: none !important; }

/* link cards: the whole card is clickable */
.pv-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: .75rem; margin: .4rem 0 .6rem; }
a.pv-card { display: flex; flex-direction: column; gap: .35rem; padding: .7rem; border-radius: 14px; color: inherit !important;
  text-decoration: none !important; border: 1px solid rgba(127,127,127,.28); background: rgba(127,127,127,.05);
  transition: transform .15s, box-shadow .15s, border-color .15s; }
a.pv-card:hover, a.pv-card:focus-visible { transform: translateY(-3px); border-color: var(--pv-accent);
  box-shadow: 0 10px 24px rgba(79,70,229,.20); }
a.pv-card img, a.pv-card .pv-noimg { width: 100%; height: 150px; object-fit: contain; border-radius: 10px; background: #fff; }
a.pv-card .pv-noimg { display: flex; align-items: center; justify-content: center; font-size: 2rem; }
a.pv-card .pv-title { font-weight: 600; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
  -webkit-box-orient: vertical; overflow: hidden; }
a.pv-card .pv-sub { font-size: .86rem; opacity: .75; }
a.pv-card .pv-go { margin-top: auto; font-weight: 600; font-size: .86rem; color: var(--pv-accent); }

/* composer */
[data-testid="stChatInput"] > div {
  border-radius: 999px; border: 1.5px solid rgba(127,127,127,.30); padding: .35rem .45rem .35rem .9rem;
  box-shadow: 0 8px 28px rgba(17,24,39,.12); transition: border-color .2s, box-shadow .2s;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--pv-accent); box-shadow: 0 0 0 4px rgba(79,70,229,.18), 0 8px 28px rgba(17,24,39,.12);
}
/* one row: text on the left, buttons on the right */
[data-testid="stChatInput"] > div > div { background: transparent; flex-direction: row !important;
  flex-wrap: nowrap !important; align-items: center !important; gap: .6rem; }
[data-testid="stChatInput"] > div > div > div:first-child { flex: 1 1 auto; min-width: 0; }
[data-testid="stChatInput"] > div > div > div:last-child { flex: 0 0 auto; width: auto !important; align-items: center; }
[data-testid="stChatInputTextArea"] { font-size: 1.05rem; }

/* microphone */
[data-testid="stChatInputMicButton"] {
  width: var(--pv-mic) !important; height: var(--pv-mic) !important; min-width: var(--pv-mic);
  border-radius: 50% !important; color: #fff !important;
  background: linear-gradient(135deg, #4f46e5, #14b8a6) !important;
  box-shadow: 0 4px 14px rgba(79,70,229,.40); transition: transform .15s, box-shadow .15s;
}
[data-testid="stChatInputMicButton"]:hover { transform: scale(1.07); box-shadow: 0 6px 20px rgba(79,70,229,.55); }
[data-testid="stChatInputMicButton"] svg { width: calc(var(--pv-mic) * .48) !important;
  height: calc(var(--pv-mic) * .48) !important; color: #fff !important; fill: currentColor; }
[data-testid="stChatInputSubmitButton"] { width: 40px !important; height: 40px !important; border-radius: 50% !important; }

/* listening: the ✓ (approve) button exists only while recording; the waveform element is always in the page */
[data-testid="stChatInput"]:has([data-testid="stChatInputApproveButton"]) > div {
  border-color: var(--pv-live); animation: pv-glow 1.4s ease-in-out infinite;
}
[data-testid="stChatInput"]:has([data-testid="stChatInputApproveButton"])::before {
  content: "● Listening…  tap ✓ when you finish, ✕ to cancel";
  display: block; width: fit-content; margin: 0 auto .55rem auto; padding: .3rem .95rem; border-radius: 999px;
  font-size: .92rem; font-weight: 600; color: #fff; background: var(--pv-live);
  animation: pv-blink 1.4s ease-in-out infinite;
}
[data-testid="stChatInputApproveButton"], [data-testid="stChatInputCancelButton"] {
  width: 44px !important; height: 44px !important; border-radius: 50% !important;
}
[data-testid="stChatInputApproveButton"] { background: #16a34a !important; color: #fff !important; }
[data-testid="stChatInputApproveButton"] svg { color: #fff !important; fill: currentColor; }
[data-testid="stChatInputCancelButton"] { background: rgba(127,127,127,.18) !important; }

@keyframes pv-glow { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.45); } 50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); } }
@keyframes pv-blink { 0%,100% { opacity: 1; } 50% { opacity: .65; } }
@media (prefers-reduced-motion: reduce) { [data-testid="stChatInput"] *, [data-testid="stChatInput"]::before { animation: none !important; } }

/* thinking: shown in the reply while the answer is on its way */
.pv-think { display: inline-flex; align-items: center; gap: .7rem; padding: .55rem 1rem; border-radius: 18px 18px 18px 4px;
  background: linear-gradient(110deg, rgba(79,70,229,.12) 30%, rgba(20,184,166,.28) 50%, rgba(79,70,229,.12) 70%);
  background-size: 250% 100%; animation: pv-shimmer 1.8s linear infinite; font-weight: 600; }
.pv-dots { display: inline-flex; gap: 5px; align-items: flex-end; height: 16px; }
.pv-dots i { width: 9px; height: 9px; border-radius: 50%; background: linear-gradient(135deg, #4f46e5, #14b8a6);
  animation: pv-bounce 1.1s ease-in-out infinite; }
.pv-dots i:nth-child(2) { animation-delay: .15s; }
.pv-dots i:nth-child(3) { animation-delay: .3s; }
@keyframes pv-bounce { 0%, 60%, 100% { transform: translateY(0); opacity: .55; } 30% { transform: translateY(-7px); opacity: 1; } }
@keyframes pv-shimmer { from { background-position: 120% 0; } to { background-position: -120% 0; } }
@media (prefers-reduced-motion: reduce) { .pv-think, .pv-dots i { animation: none !important; } }
</style>
"""


def thinking(label: str) -> str:
    """HTML for the animated 'thinking' bubble."""
    return (f"<div class='pv-think' role='status'><span class='pv-dots' aria-hidden='true'><i></i><i></i><i></i></span>"
            f"<span>{html.escape(label)}</span></div>")


def apply() -> None:
    st.markdown(_CSS.replace("__MIC__", str(MIC_SIZE_PX)), unsafe_allow_html=True)
