"""Look of the Assistant's chat box: rounded composer, large mic on the right, 'Listening' animation.

Targets Streamlit's data-testid attributes (checked against Streamlit 1.67). If a future version renames them,
the chat box still works and only falls back to the default look.
"""
import streamlit as st

MIC_SIZE_PX = 52  # change this to make the microphone button bigger or smaller

_CSS = """
<style>
:root { --pv-mic: __MIC__px; --pv-accent: #4f46e5; --pv-live: #ef4444; }

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
</style>
"""


def apply() -> None:
    st.markdown(_CSS.replace("__MIC__", str(MIC_SIZE_PX)), unsafe_allow_html=True)
