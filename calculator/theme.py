"""Bold, graphic theme inspired by the supplied course-page reference."""
import streamlit as st

CSS = """
<style>
.stApp {background-color:#f4f1e8;color:#181916;background-image:radial-gradient(#dcd7c8 .8px,transparent .8px);background-size:18px 18px;}
.block-container {max-width:1180px;padding:1.5rem 2rem 3rem;}
header[data-testid="stHeader"] {background:#f4f1e8;}
.mc-nav {display:flex;align-items:center;justify-content:space-between;gap:16px;background:#39575e;color:#fffdf6;border:3px solid #191a17;padding:16px 22px;box-shadow:5px 5px 0 #191a17;margin-bottom:38px;}
.mc-brand {font-size:22px;font-weight:900;letter-spacing:-.8px;display:flex;align-items:center;gap:12px;}
.mc-mark {display:grid;place-items:center;background:#d7bd66;color:#191a17;border:2px solid #191a17;width:33px;height:33px;font-size:24px;}
.mc-nav-note {font-family:monospace;font-size:11px;text-transform:uppercase;letter-spacing:1.3px;}
.mc-eyebrow {display:inline-block;background:#191a17;color:#fffdf6;font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;padding:10px 14px;margin-bottom:18px;}
.mc-hero h1 {font-family:Arial,Helvetica,sans-serif;font-weight:900;font-size:clamp(35px,5.5vw,64px);line-height:1.07;letter-spacing:-2.5px;background:#d7bd66;border:3px solid #191a17;padding:25px 30px;box-shadow:7px 7px 0 #191a17;transform:rotate(-.7deg);margin:0 5px 28px 0;color:#171814;}
.mc-hero h1 span {color:#39575e;}
.mc-hero p {background:#39575e;color:#fffdf6;border:3px solid #191a17;padding:18px 22px;font-size:16px;line-height:1.6;font-weight:500;margin:0 0 22px;max-width:none;}
.mc-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 6px 32px 0;}
.mc-step {color:#191a17;background:#a8b99b;border:3px solid #191a17;box-shadow:4px 4px 0 #191a17;padding:13px 15px;font-weight:800;font-size:12px;text-transform:uppercase;text-decoration:none !important;}
.mc-step:nth-child(2) {background:#d7bd66;}.mc-step:nth-child(3) {background:#cc9586;}
.mc-step b {margin-right:9px;font-family:monospace;}.mc-step:hover{color:#191a17;transform:translate(2px,2px);box-shadow:2px 2px 0 #191a17;}
.st-key-model_panel,.st-key-results_panel,.st-key-hardware_panel,.st-key-alternatives_panel {background:#faf8f0;border:3px solid #191a17;padding:24px 26px;margin-bottom:25px;box-shadow:6px 6px 0 #191a17;scroll-margin-top:70px;}
h2,h3 {font-family:Arial,Helvetica,sans-serif !important;font-weight:850 !important;letter-spacing:-.8px;}
h3 {font-size:27px !important;}
[data-testid="stCaptionContainer"] {color:#5e6258;line-height:1.5;}
[data-testid="stMetric"] {background:#e9e5d9;border:2px solid #191a17;padding:16px 14px;min-height:108px;box-shadow:3px 3px 0 #191a17;}
[data-testid="stMetricLabel"] {color:#45483f;font-size:12px;font-weight:600;}
[data-testid="stMetricValue"] {font-family:Arial,Helvetica,sans-serif;font-size:clamp(22px,2.5vw,32px);font-weight:800;letter-spacing:-1px;color:#191a17;}
.st-key-results_panel [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"]:first-child [data-testid="stMetric"] {background:#d7bd66;}
[data-testid="stExpander"] {border:2px solid #191a17;border-radius:0;background:#f0ede3;}
[data-testid="stExpander"] summary {font-weight:650;font-size:14px;}
[data-testid="stNumberInput"] [data-baseweb="input"],[data-testid="stTextInput"] [data-baseweb="input"],[data-baseweb="select"] > div {border:1px solid #8f9384;border-radius:0;background:#fffdf7;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input {font-variant-numeric:tabular-nums;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stFormSubmitButton"] button,[data-testid="stLinkButton"] a {border:2px solid #191a17;border-radius:0;box-shadow:3px 3px 0 #191a17;background:#a8b99b;color:#191a17;font-weight:750;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover,[data-testid="stFormSubmitButton"] button:hover {border-color:#191a17;background:#d7bd66;color:#191a17;}
[data-testid="stDownloadButton"] button {background:#d7bd66;padding:12px 23px;text-transform:uppercase;font-size:12px;}
[data-testid="stAlert"] {border-radius:0;border-left:4px solid currentColor;}
[data-testid="stDataFrame"] {border:1px solid #8f9384;border-radius:0;}
.mc-footer {border-top:3px solid #191a17;margin-top:35px;padding-top:17px;display:flex;justify-content:space-between;gap:20px;color:#5e6258;font-family:monospace;font-size:11px;text-transform:uppercase;}
@media(max-width:640px){.block-container{padding:1rem 1rem 2rem}.mc-nav{padding:12px;margin-bottom:28px}.mc-nav-note{display:none}.mc-hero h1{padding:20px;letter-spacing:-1.2px}.mc-steps{grid-template-columns:1fr;gap:10px}.st-key-model_panel,.st-key-results_panel,.st-key-hardware_panel,.st-key-alternatives_panel{padding:18px 14px}.mc-footer{flex-direction:column;gap:6px}}
@media(prefers-reduced-motion:reduce){.mc-step:hover{transform:none}}
</style>
"""


def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="mc-nav"><div class="mc-brand"><span class="mc-mark" aria-hidden="true">◇</span>ModelCompass</div><span class="mc-nav-note">Local LLM planning / Made clear</span></div>
<div class="mc-hero"><div class="mc-eyebrow">An interactive guide to model memory</div><h1>Model requirements,<br><span>made visible.</span></h1><p>Choose a model. See where the memory goes. Explore how context changes the requirements, then compare with the hardware you have.</p></div>
<div class="mc-steps"><a class="mc-step" href="#start-with-your-model"><b>01</b>Choose a model ↗</a><a class="mc-step" href="#your-memory-picture"><b>02</b>Explore memory ↗</a><a class="mc-step" href="#meet-your-hardware"><b>03</b>Compare hardware ↗</a></div>''', unsafe_allow_html=True)


def render_footer():
    st.markdown('<div class="mc-footer"><span>ModelCompass / Plan before you load.</span><span>Memory estimates · Actual usage varies by runtime.</span></div>', unsafe_allow_html=True)
