"""Streamlit UI: dataset upload -> validation -> verification -> data-mining pipeline.

This file owns presentation only. Loading lives in src/loader.py, validation
rules in src/validation.py, and the analysis/classification pipeline in
src/mining.py, so new datasets or algorithms can be added without touching
this file.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.loader import load_uploaded_file
from src.mining import (
    class_distribution,
    dataset_overview,
    guess_text_label_columns,
    run_spam_classifier,
    text_length_stats,
)
from src.validation import validate_dataset

st.set_page_config(page_title="Data Mining Dashboard", page_icon="📧", layout="wide")


def render_workflow_status(stage: str) -> None:
    """Renders the Upload -> Validation -> Verified/Failed workflow stepper."""
    steps = ["Upload", "Validation", "Verified"]
    icons = {"pending": "⚪", "active": "🔵", "done": "🟢", "failed": "🔴"}

    stage_index = {"upload": 0, "validation": 1, "verified": 2, "failed": 2}[stage]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        if stage == "failed" and i == 2:
            icon, text = icons["failed"], "Validation Failed"
        elif i < stage_index:
            icon, text = icons["done"], label
        elif i == stage_index:
            icon, text = icons["done"] if stage in ("verified",) else icons["active"], label
        else:
            icon, text = icons["pending"], label
        col.markdown(f"### {icon} {text}")


st.title("📧 Data Mining Dashboard")
st.caption("Upload a dataset (e.g. a Kaggle spam-email dataset) to validate, verify, and analyze it.")

with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("CSV or Excel file", type=["csv", "xls", "xlsx"])
    st.divider()
    st.header("2. Pipeline Options")
    require_text_label = st.checkbox(
        "Require text + label columns (spam classification)",
        value=False,
        help="Enforce a schema suited to spam/ham datasets before allowing the dataset through.",
    )
    st.caption(
        "Get a spam dataset from Kaggle, e.g. search "
        "\"SMS Spam Collection\" or \"Spam Email Classification\"."
    )

if uploaded_file is None:
    render_workflow_status("upload")
    st.info("⬅️ Upload a CSV or Excel file from the sidebar to begin.")
    st.markdown(
        "No dataset handy? A small sample spam/ham dataset is bundled at "
        "`data/sample_spam.csv` — download a real one from Kaggle for full results."
    )
    with open("data/sample_spam.csv", "rb") as f:
        st.download_button(
            "Download sample dataset", f, file_name="sample_spam.csv", mime="text/csv"
        )
    st.stop()

# --- Load ---
load_result = load_uploaded_file(uploaded_file)

# --- Validate ---
required_cols: list[str] = []
if require_text_label:
    guessed_text, guessed_label = (
        guess_text_label_columns(load_result.dataframe) if load_result.ok else (None, None)
    )
    # Only enforce columns we can't confidently guess; guessing itself is not a hard requirement.

validation = validate_dataset(load_result)

render_workflow_status("verified" if validation.is_valid else "failed")

st.subheader("File Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("File name", load_result.filename)
c2.metric("File size", f"{load_result.size_kb:.1f} KB")
c3.metric("Rows", f"{validation.n_rows:,}" if load_result.ok else "—")
c4.metric("Columns", validation.n_cols if load_result.ok else "—")

st.subheader("Validation Result")
if validation.is_valid:
    st.success("✅ Dataset Verified — ready for the data-mining pipeline.")
else:
    st.error("❌ Validation Failed — fix the issues below and re-upload.")

if validation.errors:
    st.markdown("**Errors**")
    for e in validation.errors:
        st.markdown(f"- 🔴 {e}")
if validation.warnings:
    st.markdown("**Warnings**")
    for w in validation.warnings:
        st.markdown(f"- 🟡 {w}")
if not validation.errors and not validation.warnings:
    st.caption("No issues detected.")

if not validation.is_valid:
    st.stop()

df = load_result.dataframe
assert df is not None

# --- Verified dataset details ---
st.subheader("Dataset Details")
tab_overview, tab_columns, tab_preview = st.tabs(["Overview", "Columns", "Preview"])

with tab_overview:
    o1, o2, o3 = st.columns(3)
    o1.metric("Duplicate rows", f"{validation.n_duplicates:,}")
    o2.metric("Missing cells", f"{validation.n_missing_total:,}")
    total_cells = validation.n_rows * validation.n_cols
    o3.metric(
        "Missing %",
        f"{(validation.n_missing_total / total_cells * 100) if total_cells else 0:.2f}%",
    )
    overview = dataset_overview(df)
    st.markdown("**Data type breakdown**")
    st.write(pd.Series(overview["dtype_counts"], name="count"))
    if overview["numeric_summary"] is not None:
        st.markdown("**Numeric column summary**")
        st.dataframe(overview["numeric_summary"], use_container_width=True)

with tab_columns:
    profile_df = pd.DataFrame(
        [
            {
                "Column": p.name,
                "Type": p.dtype,
                "Missing": p.missing_count,
                "Missing %": round(p.missing_pct, 2),
                "Unique values": p.n_unique,
            }
            for p in validation.column_profiles
        ]
    )
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

with tab_preview:
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

# --- Data-mining pipeline ---
st.subheader("3. Data-Mining Pipeline: Spam Classification")

guessed_text, guessed_label = guess_text_label_columns(df)
col_a, col_b, col_c = st.columns([2, 2, 1])
text_col = col_a.selectbox(
    "Text column", options=list(df.columns), index=list(df.columns).index(guessed_text) if guessed_text in df.columns else 0
)
label_col = col_b.selectbox(
    "Label column", options=list(df.columns), index=list(df.columns).index(guessed_label) if guessed_label in df.columns else 0
)
run = col_c.button("Run analysis", type="primary")

if run:
    if text_col == label_col:
        st.error("Text column and label column must be different.")
        st.stop()

    st.markdown("**Class distribution**")
    dist = class_distribution(df, label_col)
    st.bar_chart(dist)

    st.markdown("**Text length distribution**")
    st.write(text_length_stats(df, text_col))

    try:
        with st.spinner("Training TF-IDF + Naive Bayes classifier..."):
            report = run_spam_classifier(df, text_col, label_col)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("**Model performance** (TF-IDF + Multinomial Naive Bayes)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report.accuracy * 100:.1f}%")
    m2.metric("Precision (macro)", f"{report.precision * 100:.1f}%")
    m3.metric("Recall (macro)", f"{report.recall * 100:.1f}%")
    m4.metric("F1 (macro)", f"{report.f1 * 100:.1f}%")
    st.caption(f"Trained on {report.n_train} rows, evaluated on {report.n_test} held-out rows.")

    st.markdown("**Confusion matrix**")
    cm_df = pd.DataFrame(report.confusion, index=report.classes, columns=report.classes)
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Most predictive tokens per class**")
    token_cols = st.columns(len(report.top_tokens_per_class))
    for col, (cls, tokens) in zip(token_cols, report.top_tokens_per_class.items()):
        with col:
            st.markdown(f"*{cls}*")
            st.dataframe(
                pd.DataFrame(tokens, columns=["token", "log-prob"]),
                use_container_width=True,
                hide_index=True,
            )
