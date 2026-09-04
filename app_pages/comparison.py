"""Model comparison: train every registry model on the same split, compare, pick a winner."""

import altair as alt
import pandas as pd
import streamlit as st

from src import modeling
from src.evaluation import compute_metrics
from src.state import require_columns, require_trained_models

text_col, label_col = require_columns()
df = st.session_state.dataset

st.caption("Every model below was trained on the same train/test split, so the comparison is apples-to-apples.")

untrained = [m for m in modeling.MODEL_REGISTRY if m not in (st.session_state.trained_models or {})]
if untrained:
    if st.button(f"Train remaining {len(untrained)} model(s) for comparison", icon=":material/model_training:"):
        processed_text = st.session_state.get("processed_text")
        text_series = processed_text if processed_text is not None else df[text_col].astype(str)
        work_df = pd.DataFrame({"__text__": text_series, "__label__": df[label_col]})
        X, y = modeling.prepare_data(work_df, "__text__", "__label__")
        split_config = st.session_state.split_config

        try:
            split = modeling.make_split(X, y, **split_config)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        progress = st.progress(0.0, text="Training...")
        for i, model_key in enumerate(untrained):
            try:
                trained = modeling.train_model(split, model_key)
                metrics = compute_metrics(trained.pipeline, split.X_test, split.y_test, trained.classes)
            except Exception as exc:  # noqa: BLE001 - keep the app running even if one model fails
                st.error(f"Training '{model_key}' failed: {exc}")
                continue
            st.session_state.trained_models[model_key] = {
                "pipeline": trained.pipeline,
                "metrics": metrics,
                "train_time_sec": trained.train_time_sec,
                "n_train": trained.n_train,
                "n_test": trained.n_test,
                "trained_at": trained.trained_at,
                "classes": trained.classes,
            }
            progress.progress((i + 1) / len(untrained), text=f"Trained {model_key}")
        progress.empty()
        st.rerun()

models = require_trained_models()

rows = [
    {
        "Model": key,
        "Accuracy": e["metrics"].accuracy,
        "Precision": e["metrics"].precision,
        "Recall": e["metrics"].recall,
        "F1-score": e["metrics"].f1,
        "ROC-AUC": e["metrics"].roc_auc,
    }
    for key, e in models.items()
]
comparison_df = pd.DataFrame(rows)

best_model = comparison_df.loc[comparison_df["F1-score"].idxmax(), "Model"] if not comparison_df.empty else None
if best_model:
    st.success(f"Best model by F1-score: **{best_model}**", icon=":material/emoji_events:")

st.dataframe(
    comparison_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], color="#d3f2dc"),
    width="stretch",
    hide_index=True,
)

long_df = comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Value").dropna(subset=["Value"])
chart = (
    alt.Chart(long_df)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", title=None),
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1])),
        color="Model:N",
        column=alt.Column("Metric:N"),
    )
)
st.altair_chart(chart)

st.divider()
st.page_link("app_pages/single_prediction.py", label="Try a single prediction", icon=":material/chat:")
