"""Central session-state schema and page guard-clauses.

Every page that depends on a prior step (a verified dataset, chosen columns,
a trained model) calls the matching `require_*()` helper at the top instead
of re-implementing the check; this is what keeps navigating out of order
from ever crashing the app.
"""

from __future__ import annotations

import streamlit as st

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE


def init_session_state() -> None:
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("dataset_meta", None)
    st.session_state.setdefault("validation_result", None)
    st.session_state.setdefault("text_col", None)
    st.session_state.setdefault("label_col", None)
    st.session_state.setdefault("preprocessing_config", None)
    st.session_state.setdefault("split_config", {"test_size": DEFAULT_TEST_SIZE, "random_state": DEFAULT_RANDOM_STATE})
    st.session_state.setdefault("trained_models", {})
    st.session_state.setdefault("active_model_key", None)
    st.session_state.setdefault("prediction_history", [])


def set_dataset(dataset_df, load_result) -> None:
    """Store a newly loaded dataset and reset everything downstream of it
    (column choices, validation, trained models) so stale state from a
    previous dataset can never leak into the new one."""
    st.session_state.dataset = dataset_df
    st.session_state.dataset_meta = load_result
    st.session_state.validation_result = None
    st.session_state.text_col = None
    st.session_state.label_col = None
    st.session_state.trained_models = {}
    st.session_state.active_model_key = None
    st.session_state.prediction_history = []


def require_dataset():
    if st.session_state.get("dataset") is None:
        st.info("Upload a dataset first to use this page.", icon=":material/upload_file:")
        st.page_link("app_pages/upload.py", label="Go to Dataset Upload", icon=":material/upload_file:")
        st.stop()
    return st.session_state.dataset


def require_verified_dataset():
    dataset = require_dataset()
    validation = st.session_state.get("validation_result")
    if validation is None or not validation.is_valid:
        st.warning("This dataset hasn't passed validation yet.", icon=":material/error:")
        st.page_link("app_pages/validation_page.py", label="Go to Data Validation", icon=":material/fact_check:")
        st.stop()
    return dataset


def require_columns():
    require_verified_dataset()
    text_col = st.session_state.get("text_col")
    label_col = st.session_state.get("label_col")
    if not text_col or not label_col:
        st.info("Choose the text and label columns on the Upload page first.", icon=":material/view_column:")
        st.page_link("app_pages/upload.py", label="Go to Dataset Upload", icon=":material/upload_file:")
        st.stop()
    return text_col, label_col


def require_trained_models() -> dict:
    models = st.session_state.get("trained_models") or {}
    if not models:
        st.info("Train a model first to use this page.", icon=":material/model_training:")
        st.page_link("app_pages/training.py", label="Go to Model Training", icon=":material/model_training:")
        st.stop()
    return models


def get_active_model_entry() -> tuple[str, dict]:
    """The model used for single/batch prediction: defaults to the best F1 score."""
    models = require_trained_models()
    active_key = st.session_state.get("active_model_key")
    if active_key not in models:
        active_key = max(models, key=lambda k: models[k]["metrics"].f1)
        st.session_state.active_model_key = active_key
    return active_key, models[active_key]
