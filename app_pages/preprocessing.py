"""Configurable text preprocessing: pick cleaning steps, preview raw vs. processed text.

All steps here are deterministic, content-only transforms (see
`src/preprocessing.py` docstring), safe to apply to the whole dataset before
any train/test split. The only step that must stay train-only is TF-IDF
fitting, which happens later, inside the training pipeline.
"""

import pandas as pd
import streamlit as st

from src.preprocessing import PreprocessingConfig, TextPreprocessor
from src.state import require_columns

text_col, label_col = require_columns()
df = st.session_state.dataset

st.caption("Choose which cleaning steps to apply to the text column before training. All steps are optional.")

current: PreprocessingConfig = st.session_state.get("preprocessing_config") or PreprocessingConfig()

with st.form("preprocessing_form"):
    st.markdown("**Cleaning steps**")
    c1, c2, c3 = st.columns(3)
    with c1:
        lowercase = st.checkbox("Lowercase", value=current.lowercase)
        remove_urls = st.checkbox("Remove URLs", value=current.remove_urls)
        remove_emails = st.checkbox("Remove emails", value=current.remove_emails)
    with c2:
        remove_phone_numbers = st.checkbox("Remove phone numbers", value=current.remove_phone_numbers)
        remove_punctuation = st.checkbox("Remove punctuation", value=current.remove_punctuation)
        remove_numbers = st.checkbox("Remove numbers", value=current.remove_numbers)
    with c3:
        normalize_whitespace = st.checkbox("Collapse whitespace", value=current.normalize_whitespace)
        remove_stopwords = st.checkbox("Remove stopwords", value=current.remove_stopwords)

    normalization = st.radio(
        "Word normalization",
        options=["none", "stem", "lemmatize"],
        index=["none", "stem", "lemmatize"].index(current.normalization),
        horizontal=True,
        help="Stemming uses the Porter algorithm; lemmatization here is a lightweight, "
        "rule-based simplification (not a full dictionary-based lemmatizer).",
    )

    submitted = st.form_submit_button("Apply preprocessing", icon=":material/auto_fix_high:")

if submitted:
    st.session_state.preprocessing_config = PreprocessingConfig(
        lowercase=lowercase,
        remove_urls=remove_urls,
        remove_emails=remove_emails,
        remove_phone_numbers=remove_phone_numbers,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        normalize_whitespace=normalize_whitespace,
        remove_stopwords=remove_stopwords,
        normalization=normalization,
    )


@st.cache_data(show_spinner="Cleaning text...")
def _apply_preprocessing(text_series: pd.Series, config_dict: dict) -> pd.Series:
    config = PreprocessingConfig(**config_dict)
    return TextPreprocessor(config).transform(text_series)


config = st.session_state.get("preprocessing_config")
if config is not None:
    processed = _apply_preprocessing(df[text_col], config.as_dict())
    st.session_state.processed_text = processed

    st.subheader("Raw vs. processed")
    sample = df[[text_col]].head(8).copy()
    sample["processed"] = processed.head(8)
    sample = sample.rename(columns={text_col: "raw"})
    st.dataframe(sample, width="stretch")

    avg_len_before = df[text_col].astype(str).str.len().mean()
    avg_len_after = processed.str.len().mean()
    with st.container(horizontal=True):
        st.metric("Avg. length before", f"{avg_len_before:.0f} chars", border=True)
        st.metric("Avg. length after", f"{avg_len_after:.0f} chars", border=True)
        st.metric("Reduction", f"{(1 - avg_len_after / avg_len_before) * 100:.1f}%" if avg_len_before else "N/A", border=True)

    st.divider()
    st.page_link("app_pages/training.py", label="Continue to Model Training", icon=":material/model_training:")
else:
    st.info("Configure preprocessing above and click **Apply preprocessing** to see a preview.", icon=":material/info:")
    st.page_link(
        "app_pages/training.py",
        label="Skip preprocessing and go to Model Training",
        icon=":material/model_training:",
    )
