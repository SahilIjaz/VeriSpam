"""Home / overview: KPI snapshot of where the session stands, plus a quick-start guide."""

from pathlib import Path

import streamlit as st

from src.loader import load_local_file
from src.state import set_dataset

dataset = st.session_state.get("dataset")
validation = st.session_state.get("validation_result")
trained_models = st.session_state.get("trained_models") or {}

st.caption("Upload a spam/ham dataset, validate it, explore it, train and compare models, then predict, all in one place.")

with st.container(horizontal=True):
    st.metric(
        "Dataset",
        f"{dataset.shape[0]:,} rows" if dataset is not None else "Not loaded",
        border=True,
    )
    st.metric(
        "Validation",
        "Verified" if validation and validation.is_valid else ("Failed" if validation else "Pending"),
        border=True,
    )
    st.metric("Models trained", len(trained_models), border=True)
    best_f1 = max((m["metrics"].f1 for m in trained_models.values()), default=None)
    st.metric("Best F1-score", f"{best_f1:.3f}" if best_f1 is not None else "N/A", border=True)

st.divider()

col_guide, col_start = st.columns([2, 1])

with col_guide:
    st.subheader("Quick start")
    st.markdown(
        """
1. **Dataset upload**: upload a CSV/Excel spam/ham dataset (or load the bundled sample below).
2. **Data validation**: check format, schema, missing values, duplicates, and class balance.
3. **Exploratory analysis**: inspect distributions and the most common words per class.
4. **Preprocessing**: configure text cleaning (URLs, punctuation, stopwords, stemming, ...).
5. **Model training**: pick one or more models, set the train/test split, and train.
6. **Model evaluation / comparison**: inspect metrics, confusion matrices, ROC-AUC, and pick a winner.
7. **Single / batch prediction**: classify one message or a whole new file, with explanations.
        """
    )

with col_start:
    with st.container(border=True):
        st.markdown("**New here?**")
        st.write("Try the bundled sample dataset to explore the app end-to-end.")
        if st.button("Load sample dataset", icon=":material/dataset:", width="stretch"):
            load_result = load_local_file(Path("data/sample_spam.csv"))
            if load_result.ok:
                set_dataset(load_result.dataframe, load_result)
                st.switch_page("app_pages/upload.py")
            else:
                st.error(load_result.error)
