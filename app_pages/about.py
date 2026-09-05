"""About: what this app does, how the pipeline works, and its privacy posture."""

import streamlit as st

st.markdown(
    """
This is an end-to-end **spam/ham text classification and analytics platform**, built to work
with any spam/ham-style dataset, not just one hardcoded schema.

### Pipeline

1. **Dataset upload**: CSV/Excel, auto-detected text/label columns.
2. **Data validation**: format, schema, missing values, duplicates, class imbalance.
3. **Exploratory analysis**: distributions, common words, data-quality breakdowns.
4. **Preprocessing**: configurable, deterministic text cleaning.
5. **Model training**: Naive Bayes, Logistic Regression, Linear SVM, or Random Forest,
   each wrapped in a TF-IDF pipeline fit only on the training split (no data leakage).
6. **Evaluation & comparison**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
   timing, and a best-model recommendation by F1-score.
7. **Prediction**: a single message, or a whole new file, with word-level explanations.

### Privacy

- Uploaded datasets are kept **in memory for this session only**; never written to disk.
- Only a trained model (and the metrics/columns you choose to save) can be persisted, via
  the **Save this model** button on the Model Evaluation page, and only when you click it.
- Single-prediction history lives in this session's memory and is cleared when the tab closes;
  message content is never logged.

### Tech stack

Streamlit for the interface; pandas for data handling; scikit-learn for TF-IDF, the four
classifiers, and evaluation metrics; joblib for model persistence.
    """
)
