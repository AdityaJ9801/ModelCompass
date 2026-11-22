"""Warm editorial presentation, independent of calculator behavior."""
import streamlit as st

CSS = """
<style>
.stApp {background: #f7f5f0; color:#302d29;}
.block-container {max-width:1160px; padding:2rem 2.5rem 3rem;}
header[data-testid="stHeader"] {background:rgba(247,245,240,.94);}
.mc-nav {display:flex;align-items:center;justify-content:space-between;padding:8px 0 26px;border-bottom:1px solid #ded8ce;gap:16px;}
.mc-brand {font-family:Georgia,serif;font-size:24px;letter-spacing:-.7px;display:flex;align-items:center;gap:12px;}
.mc-mark {display:inline-grid;place-items:center;background:#b45e42;color:#fffaf3;width:35px;height:35px;border-radius:11px;font-size:25px;}
.mc-nav-note {font-size:11px;letter-spacing:1.6px;color:#746d62;text-transform:uppercase;}
.mc-hero {padding:45px 0 30px;text-align:center;}
.mc-eyebrow {color:#9d5038;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:13px;}
.mc-hero h1 {font-family:Georgia,'Times New Roman',serif;font-weight:400;font-size:clamp(36px,5vw,57px);line-height:1.12;letter-spacing:-1.8px;margin:0 auto 18px;padding:0;color:#302d29;}
.mc-hero h1 em {color:#ae5b41;font-weight:400;}
.mc-hero p {font-size:16px;line-height:1.65;max-width:590px;margin:auto;color:#6c655c;}
.mc-steps {display:flex;justify-content:center;gap:10px;flex-wrap:wrap;padding:0 0 32px;}
.mc-step {font-size:12px;color:#6c655c;padding:8px 13px;border:1px solid #ded8ce;border-radius:100px;background:#fbfaf7;}
.mc-step b {color:#a9583e;margin-right:7px;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:18px;}
.st-key-model_panel, .st-key-results_panel, .st-key-hardware_panel, .st-key-alternatives_panel {background:#fdfcf9;border:1px solid #e2ddd3;border-radius:20px;padding:24px 26px;margin-bottom:22px;box-shadow:0 3px 14px rgba(55,43,28,.025);}
h2,h3 {font-family:Georgia,'Times New Roman',serif !important;font-weight:400 !important;letter-spacing:-.5px;}
h3 {font-size:25px !important;}
[data-testid="stCaptionContainer"] {color:#756e63;line-height:1.5;}
[data-testid="stMetric"] {background:#f2efe8;border:1px solid #e5dfd4;padding:18px 17px;border-radius:14px;min-height:112px;}
[data-testid="stMetricLabel"] {color:#756e63;font-size:12px;}
[data-testid="stMetricValue"] {font-family:Georgia,serif;font-size:clamp(23px,2.4vw,32px);letter-spacing:-.6px;color:#40382f;}
.st-key-results_panel [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"]:first-child [data-testid="stMetric"] {background:#f2e1d6;border-color:#e2bda8;}
[data-testid="stExpander"] {border-color:#e3ddd2;border-radius:12px;background:#faf8f3;}
[data-testid="stExpander"] summary {font-size:14px;}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {font-variant-numeric:tabular-nums;}
[data-testid="stNumberInput"] [data-baseweb="input"], [data-testid="stTextInput"] [data-baseweb="input"], [data-baseweb="select"] > div {border-radius:10px;background:#f5f2eb;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {border-radius:10px;border-color:#d9cfc1;transition:background .15s,border-color .15s;}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {border-color:#b45e42;}
[data-testid="stDownloadButton"] button {background:#a9553b;color:#fffaf5;border-color:#a9553b;padding:10px 22px;}
[data-testid="stDownloadButton"] button:hover {background:#91472f;color:white;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
.mc-footer {border-top:1px solid #ded8ce;margin-top:28px;padding-top:18px;display:flex;justify-content:space-between;gap:20px;color:#837a6e;font-size:12px;}
@media(max-width:640px) {.block-container{padding:1rem 1rem 2rem}.mc-nav-note{display:none}.mc-hero{padding:30px 0 24px}.mc-hero h1{letter-spacing:-1px}.mc-steps{gap:6px}.mc-step{font-size:11px;padding:6px 9px}.st-key-model_panel,.st-key-results_panel,.st-key-hardware_panel,.st-key-alternatives_panel{padding:18px 15px}.mc-footer{flex-direction:column;gap:5px}}
</style>
"""


def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="mc-nav"><div class="mc-brand"><span class="mc-mark" aria-hidden="true">◇</span>ModelCompass</div><span class="mc-nav-note">A little clarity before you compute</span></div>
<div class="mc-hero"><div class="mc-eyebrow">Your local LLM, thoughtfully planned</div><h1>Big models.<br><em>Clear requirements.</em></h1><p>Find the memory your model needs, explore its context capacity,<br>and see where your hardware fits in.</p></div>
<div class="mc-steps"><span class="mc-step"><b>01</b>Choose a model</span><span class="mc-step"><b>02</b>Explore memory</span><span class="mc-step"><b>03</b>Compare hardware · optional</span></div>''', unsafe_allow_html=True)


def render_footer():
    st.markdown('<div class="mc-footer"><span>ModelCompass · Make room for your next idea.</span><span>Planning estimates. Real-world results may vary.</span></div>', unsafe_allow_html=True)
