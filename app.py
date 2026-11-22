import os

# Bound Hub HTTP calls, including model search. No local saved credentials are used.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

import streamlit as st
from calculator.ui import render

st.set_page_config(page_title="ModelCompass · LLM memory planner", page_icon="◈", layout="wide")
render()
