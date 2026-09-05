"""Model training: pick model(s), configure the split, train, or load a saved model."""

import pandas as pd
import streamlit as st

from src import modeling, persistence
from src.config import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE
from src.evaluation import compute_metrics, metrics_from_saved
from src.state import require_columns

text_col, label_col = require_columns()
df = st.session_state.dataset

processed_text = st.session_state.get("processed_text")
using_processed = processed_text is not None
st.caption(
    f"Training on the **{'preprocessed' if using_processed else 'raw'}** text column."
    + ("" if using_processed else " Visit Preprocessing first if you want cleaned text.")
)

with st.form("training_form"):
    selected_models = st.multiselect(
        "Model(s) to train",
        options=list(modeling.MODEL_REGISTRY),
        default=["Naive Bayes"],
        help="Train one model, or several to compare on the Model Comparison page.",
    )
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=DEFAULT_TEST_SIZE, step=0.05)
    with c2:
        random_state = st.number_input("Random seed", min_value=0, value=DEFAULT_RANDOM_STATE, step=1)
    train_clicked = st.form_submit_button("Train", icon=":material/model_training:")

if train_clicked:
    if not selected_models:
        st.error("Select at least one model to train.")
        st.stop()

    text_series = processed_text if using_processed else df[text_col].astype(str)
    work_df = pd.DataFrame({"__text__": text_series, "__label__": df[label_col]})
    X, y = modeling.prepare_data(work_df, "__text__", "__label__")

    try:
        split = modeling.make_split(X, y, test_size=test_size, random_state=int(random_state))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not split.stratified:
        st.warning(
            "Could not stratify the split (a class is too small); using a plain random split instead.",
            icon=":material/warning:",
        )
    st.session_state.split_config = {"test_size": test_size, "random_state": int(random_state)}

    progress = st.progress(0.0, text="Training models...")
    succeeded = []
    for i, model_key in enumerate(selected_models):
        try:
            trained = modeling.train_model(split, model_key)
            metrics = compute_metrics(trained.pipeline, split.X_test, split.y_test, trained.classes)
        except Exception as exc:  # noqa: BLE001 - surface any training/eval failure without crashing the app
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
        succeeded.append(model_key)
        progress.progress((i + 1) / len(selected_models), text=f"Trained {model_key}")
    progress.empty()
    if succeeded:
        st.success(f"Trained {len(succeeded)} of {len(selected_models)} model(s).", icon=":material/check_circle:")

if st.session_state.trained_models:
    st.subheader("Trained models this session")
    summary = pd.DataFrame(
        [
            {
                "Model": key,
                "Accuracy": entry["metrics"].accuracy,
                "F1-score": entry["metrics"].f1,
                "Train time (s)": entry["train_time_sec"],
                "Trained at": entry["trained_at"],
            }
            for key, entry in st.session_state.trained_models.items()
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)
    st.page_link("app_pages/evaluation.py", label="Continue to Model Evaluation", icon=":material/insights:")

with st.expander("Load a previously saved model", expanded=False):
    saved = persistence.list_saved_models()
    if not saved:
        st.caption("No saved models yet; train and save one from the Model Evaluation page.")
    else:
        options = {f"{m['model_key']} ({m.get('trained_at', m['slug'])})": m for m in saved}
        choice = st.selectbox("Saved models", options=list(options))
        if st.button("Load into this session", icon=":material/folder_open:"):
            meta = options[choice]
            try:
                pipeline, meta = persistence.load_pipeline(meta["slug"])
            except Exception as exc:  # noqa: BLE001 - surface any load failure without crashing the app
                st.error(f"Could not load model: {exc}")
            else:
                st.session_state.trained_models[f"{meta['model_key']} (loaded)"] = {
                    "pipeline": pipeline,
                    "metrics": metrics_from_saved(meta.get("metrics", {})),
                    "train_time_sec": meta.get("metrics", {}).get("train_time_sec", 0.0),
                    "n_train": meta.get("n_train", 0),
                    "n_test": meta.get("n_test", 0),
                    "trained_at": meta.get("trained_at", ""),
                    "classes": meta.get("classes", []),
                }
                st.success(f"Loaded '{meta['model_key']}': ready for prediction.", icon=":material/check_circle:")
