"""Model evaluation: metrics, confusion matrix, classification report, ROC-AUC, persistence."""

import altair as alt
import pandas as pd
import streamlit as st

from src import persistence
from src.report import build_html_report
from src.state import require_trained_models

models = require_trained_models()
model_keys = list(models)
default_index = model_keys.index(st.session_state.active_model_key) if st.session_state.get("active_model_key") in models else 0
model_key = st.selectbox("Model", options=model_keys, index=default_index)
st.session_state.active_model_key = model_key
entry = models[model_key]
metrics = entry["metrics"]

with st.container(horizontal=True):
    st.metric("Accuracy", f"{metrics.accuracy:.1%}", border=True)
    st.metric("Precision (macro)", f"{metrics.precision:.1%}", border=True)
    st.metric("Recall (macro)", f"{metrics.recall:.1%}", border=True)
    st.metric("F1-score (macro)", f"{metrics.f1:.1%}", border=True)
    st.metric("ROC-AUC", f"{metrics.roc_auc:.3f}" if metrics.roc_auc is not None else "N/A", border=True, help=metrics.roc_auc_note)

with st.container(horizontal=True):
    st.metric("Training time", f"{entry['train_time_sec']:.3f}s", border=True)
    st.metric("Prediction time", f"{metrics.predict_time_sec:.3f}s", border=True)
    st.metric("Train / test rows", f"{entry['n_train']:,} / {entry['n_test']:,}", border=True)

if metrics.confusion is None:
    st.info(
        "This model was loaded from disk rather than freshly trained this session, so a confusion "
        "matrix and ROC curve aren't available; only the saved summary metrics above.",
        icon=":material/info:",
    )
else:
    tab_cm, tab_report, tab_roc = st.tabs(["Confusion matrix", "Classification report", "ROC curve"])

    with tab_cm:
        cm_df = pd.DataFrame(metrics.confusion, index=metrics.classes, columns=metrics.classes)
        st.dataframe(cm_df, width="stretch")
        cm_long = cm_df.reset_index(names="actual").melt(id_vars="actual", var_name="predicted", value_name="count")
        heatmap = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(x=alt.X("predicted:N"), y=alt.Y("actual:N"), color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")))
        )
        labels = alt.Chart(cm_long).mark_text().encode(x="predicted:N", y="actual:N", text="count:Q")
        st.altair_chart(heatmap + labels)

    with tab_report:
        st.code(metrics.classification_report, language=None)

    with tab_roc:
        if metrics.fpr is not None and metrics.tpr is not None:
            roc_df = pd.DataFrame({"False positive rate": metrics.fpr, "True positive rate": metrics.tpr})
            chart = (
                alt.Chart(roc_df)
                .mark_line()
                .encode(x="False positive rate", y="True positive rate")
            )
            diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            st.altair_chart(chart + diagonal)
        else:
            st.info(metrics.roc_auc_note or "ROC curve not available for this model/dataset.", icon=":material/info:")

st.divider()
col_save, col_report = st.columns(2)
with col_save:
    if st.button("Save this model", icon=":material/save:"):
        metadata = {
            "trained_at": entry["trained_at"],
            "n_train": entry["n_train"],
            "n_test": entry["n_test"],
            "classes": entry["classes"],
            "text_col": st.session_state.get("text_col"),
            "label_col": st.session_state.get("label_col"),
            "preprocessing_config": (
                st.session_state.preprocessing_config.as_dict() if st.session_state.get("preprocessing_config") else {}
            ),
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "predict_time_sec": metrics.predict_time_sec,
                "train_time_sec": entry["train_time_sec"],
            },
        }
        slug = persistence.save_pipeline(entry["pipeline"], model_key, metadata)
        st.success(f"Saved as '{slug}'. Reload it anytime from Model Training.", icon=":material/check_circle:")

with col_report:
    report_rows = [
        {
            "model_key": key,
            "accuracy": e["metrics"].accuracy,
            "precision": e["metrics"].precision,
            "recall": e["metrics"].recall,
            "f1": e["metrics"].f1,
            "roc_auc": e["metrics"].roc_auc,
            "train_time_sec": e["train_time_sec"],
            "predict_time_sec": e["metrics"].predict_time_sec,
        }
        for key, e in models.items()
    ]
    html_report = build_html_report(
        dataset_name=st.session_state.dataset_meta.filename if st.session_state.get("dataset_meta") else "dataset",
        n_rows=len(st.session_state.dataset) if st.session_state.get("dataset") is not None else 0,
        n_cols=st.session_state.dataset.shape[1] if st.session_state.get("dataset") is not None else 0,
        text_col=st.session_state.get("text_col") or "N/A",
        label_col=st.session_state.get("label_col") or "N/A",
        preprocessing_config=(
            st.session_state.preprocessing_config.as_dict() if st.session_state.get("preprocessing_config") else {}
        ),
        model_results=report_rows,
    )
    st.download_button(
        "Download HTML report",
        data=html_report,
        file_name="model_report.html",
        mime="text/html",
        icon=":material/download:",
    )

st.page_link("app_pages/comparison.py", label="Compare all trained models", icon=":material/compare_arrows:")
