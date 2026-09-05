"""Data validation: format/schema/quality checks, verified/failed banner, class imbalance."""

import streamlit as st

from src.state import require_dataset
from src.validation import detect_class_imbalance, validate_dataset

require_dataset()
load_result = st.session_state.dataset_meta

validation = validate_dataset(load_result)
st.session_state.validation_result = validation

steps = ["Upload", "Validation", "Verified" if validation.is_valid else "Failed"]
icons = ["🟢", "🟢", "🟢" if validation.is_valid else "🔴"]
with st.container(horizontal=True):
    for icon, label in zip(icons, steps):
        st.markdown(f"##### {icon} {label}")

st.divider()

if validation.is_valid:
    st.success("Dataset verified: ready for exploratory analysis, preprocessing, and training.", icon=":material/check_circle:")
else:
    st.error("Validation failed: fix the issues below and re-upload.", icon=":material/error:")

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

with st.container(horizontal=True):
    st.metric("Rows", f"{validation.n_rows:,}", border=True)
    st.metric("Columns", validation.n_cols, border=True)
    st.metric("Duplicate rows", f"{validation.n_duplicates:,}", border=True)
    st.metric("Missing cells", f"{validation.n_missing_total:,}", border=True)

label_col = st.session_state.get("label_col")
if validation.is_valid and label_col:
    st.subheader("Class balance")
    imbalance = detect_class_imbalance(st.session_state.dataset, label_col)
    if imbalance is None:
        st.info(f"Column '{label_col}' doesn't have at least 2 classes to compare.", icon=":material/info:")
    else:
        with st.container(horizontal=True):
            st.metric("Majority class", f"{imbalance.majority_class} ({imbalance.class_counts.iloc[0]:,})", border=True)
            st.metric("Minority class", f"{imbalance.minority_class} ({imbalance.class_counts.iloc[-1]:,})", border=True)
            st.metric("Imbalance ratio", f"{imbalance.ratio:.1f} : 1", border=True)
        st.bar_chart(imbalance.class_counts)
        if imbalance.is_severe:
            st.warning(
                "Severe class imbalance detected: accuracy alone will be misleading; "
                "watch precision/recall/F1 per class during evaluation.",
                icon=":material/warning:",
            )
        elif imbalance.is_imbalanced:
            st.info("Moderate class imbalance detected.", icon=":material/info:")

if validation.is_valid:
    st.divider()
    st.page_link("app_pages/eda.py", label="Continue to Exploratory Analysis", icon=":material/query_stats:")
