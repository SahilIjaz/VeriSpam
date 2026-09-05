"""Single message prediction: verdict, confidence, and word-level explanation."""

import pandas as pd
import streamlit as st

from src.explain import explain_prediction
from src.modeling import get_confidence_scores
from src.state import get_active_model_entry, require_trained_models

models = require_trained_models()
model_keys = list(models)
active_key, _ = get_active_model_entry()
model_key = st.selectbox("Model", options=model_keys, index=model_keys.index(active_key))
st.session_state.active_model_key = model_key
entry = models[model_key]
pipeline = entry["pipeline"]

message = st.text_area("Message text", height=120, placeholder="Paste an SMS or email message to classify...")
predict_clicked = st.button("Predict", icon=":material/send:", type="primary", disabled=not message.strip())

if predict_clicked and message.strip():
    try:
        prediction = pipeline.predict([message])[0]
        confidence = get_confidence_scores(pipeline, pd.Series([message]))
        confidence_value = float(confidence[0]) if confidence is not None else None
        explanation = explain_prediction(pipeline, message)
    except Exception as exc:  # noqa: BLE001 - never let a bad input crash the page
        st.error(f"Prediction failed: {exc}")
    else:
        is_spam_like = str(prediction).lower() in ("spam", "1", "true", "yes")
        badge = st.error if is_spam_like else st.success
        badge(f"Prediction: **{prediction}**" + (f" ({confidence_value:.1%} confidence)" if confidence_value else ""), icon=":material/label:")

        if explanation.top_words:
            st.markdown(f"**Why:** {explanation.method}")
            words_df = pd.DataFrame(
                [{"word": c.word, "influence": c.weight} for c in explanation.top_words]
            ).set_index("word")
            st.bar_chart(words_df)
            top_3 = ", ".join(c.word for c in explanation.top_words[:3])
            st.caption(f"Most influential words: {top_3}.")
        else:
            st.caption(explanation.method)

        st.session_state.prediction_history.insert(
            0,
            {
                "message": message[:80] + ("..." if len(message) > 80 else ""),
                "prediction": str(prediction),
                "confidence": f"{confidence_value:.1%}" if confidence_value else "N/A",
                "model": model_key,
            },
        )

if st.session_state.prediction_history:
    st.divider()
    st.subheader("Prediction history (this session)")
    st.caption("Kept in memory only for this session; never written to disk.")
    st.dataframe(pd.DataFrame(st.session_state.prediction_history), width="stretch", hide_index=True)
    if st.button("Clear history", icon=":material/delete:"):
        st.session_state.prediction_history = []
        st.rerun()
