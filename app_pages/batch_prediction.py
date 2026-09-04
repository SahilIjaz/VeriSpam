"""Batch prediction: classify a whole new file of unseen messages and download the results."""

from io import BytesIO

import pandas as pd
import streamlit as st

from src.column_detection import guess_text_label_columns
from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.loader import load_uploaded_file
from src.modeling import get_confidence_scores
from src.state import require_trained_models
from src.validation import validate_dataset

models = require_trained_models()
model_keys = list(models)
default_index = model_keys.index(st.session_state.active_model_key) if st.session_state.get("active_model_key") in models else 0
model_key = st.selectbox("Model to use for prediction", options=model_keys, index=default_index)
pipeline = models[model_key]["pipeline"]

st.caption("Upload a new file of unseen messages (no label column needed). Nothing here is written to disk.")
batch_file = st.file_uploader(
    "Upload messages to classify",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    key="batch_uploader",
)

if batch_file is None:
    st.stop()

load_result = load_uploaded_file(batch_file)
if not load_result.ok:
    st.error(load_result.error)
    st.stop()
if load_result.size_mb > MAX_FILE_SIZE_MB:
    st.error(f"File is {load_result.size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit.")
    st.stop()

validation = validate_dataset(load_result)
if not validation.is_valid:
    st.error("This file failed validation:")
    for e in validation.errors:
        st.markdown(f"- 🔴 {e}")
    st.stop()
for w in validation.warnings:
    st.markdown(f"- 🟡 {w}")

batch_df = load_result.dataframe
guessed_text, _ = guess_text_label_columns(batch_df)
columns = list(batch_df.columns)
text_col = st.selectbox(
    "Text column to classify",
    options=columns,
    index=columns.index(guessed_text) if guessed_text in columns else 0,
)

if st.button("Run predictions", icon=":material/bolt:", type="primary"):
    try:
        texts = batch_df[text_col].astype(str)
        predictions = pipeline.predict(texts)
        confidence = get_confidence_scores(pipeline, texts)
    except Exception as exc:  # noqa: BLE001 - never let a bad batch crash the page
        st.error(f"Batch prediction failed: {exc}")
        st.stop()

    result_df = batch_df.copy()
    result_df["predicted_label"] = predictions
    result_df["confidence"] = confidence if confidence is not None else pd.NA

    st.subheader("Prediction statistics")
    with st.container(horizontal=True):
        st.metric("Rows classified", f"{len(result_df):,}", border=True)
        if confidence is not None:
            st.metric("Average confidence", f"{confidence.mean():.1%}", border=True)
        for cls, count in pd.Series(predictions).value_counts().items():
            st.metric(str(cls), f"{count:,}", border=True)

    st.dataframe(result_df.head(100), width="stretch")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="predictions")

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        st.download_button(
            "Download as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv", icon=":material/download:"
        )
    with col_xlsx:
        st.download_button(
            "Download as Excel",
            data=excel_buffer.getvalue(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:",
        )
