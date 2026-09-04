"""Entry point: navigation, global session-state init, page config."""

import streamlit as st

from src.config import APP_TITLE
from src.state import init_session_state

st.set_page_config(page_title=APP_TITLE, page_icon="📧", layout="wide")

init_session_state()

page = st.navigation(
    {
        "": [
            st.Page("app_pages/home.py", title="Home", icon=":material/home:"),
            st.Page("app_pages/about.py", title="About", icon=":material/info:"),
        ],
        "Data": [
            st.Page("app_pages/upload.py", title="Dataset upload", icon=":material/upload_file:"),
            st.Page("app_pages/validation_page.py", title="Data validation", icon=":material/fact_check:"),
            st.Page("app_pages/eda.py", title="Exploratory analysis", icon=":material/query_stats:"),
            st.Page("app_pages/preprocessing.py", title="Preprocessing", icon=":material/auto_fix_high:"),
        ],
        "Modeling": [
            st.Page("app_pages/training.py", title="Model training", icon=":material/model_training:"),
            st.Page("app_pages/evaluation.py", title="Model evaluation", icon=":material/insights:"),
            st.Page("app_pages/comparison.py", title="Model comparison", icon=":material/compare_arrows:"),
        ],
        "Predict": [
            st.Page("app_pages/single_prediction.py", title="Single prediction", icon=":material/chat:"),
            st.Page("app_pages/batch_prediction.py", title="Batch prediction", icon=":material/upload:"),
        ],
    },
    position="sidebar",
)

st.title(page.title, icon=page.icon)
page.run()
