"""Data-mining / analysis pipeline, run only after a dataset passes validation.

Kept separate from validation and UI so new datasets or algorithms can be
plugged in without touching either. Currently ships one pipeline
(TF-IDF + Multinomial Naive Bayes text classification, suited to the Kaggle
spam-email/SMS datasets this app targets) plus generic exploratory helpers
that work on any tabular dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Common column names seen across Kaggle spam-classification datasets
# (e.g. "SMS Spam Collection": v1/v2, or "Spam Email": label/text).
TEXT_COLUMN_CANDIDATES = ["text", "message", "email", "body", "v2", "content", "sms"]
LABEL_COLUMN_CANDIDATES = ["label", "target", "spam", "category", "v1", "class"]


def guess_text_label_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Best-effort guess of which columns hold the message text and the spam/ham label."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    text_col = next((cols_lower[c] for c in TEXT_COLUMN_CANDIDATES if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in LABEL_COLUMN_CANDIDATES if c in cols_lower), None)

    if text_col is None:
        # Fall back to the object column with the longest average string length.
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols) > 0:
            avg_lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
            text_col = max(avg_lens, key=avg_lens.get)

    if label_col is None:
        # Fall back to a low-cardinality column (likely categorical) that isn't the text column.
        candidates = [c for c in df.columns if c != text_col and df[c].nunique(dropna=True) <= 10]
        if candidates:
            label_col = min(candidates, key=lambda c: df[c].nunique(dropna=True))

    return text_col, label_col


def dataset_overview(df: pd.DataFrame) -> dict:
    """Generic exploratory summary that applies to any tabular dataset."""
    numeric_df = df.select_dtypes(include="number")
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "numeric_summary": numeric_df.describe().transpose() if not numeric_df.empty else None,
    }


def class_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(dropna=False)


def text_length_stats(df: pd.DataFrame, text_col: str) -> pd.Series:
    return df[text_col].astype(str).str.len().describe()


@dataclass
class ClassificationReport:
    text_col: str
    label_col: str
    classes: list[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    top_tokens_per_class: dict[str, list[tuple[str, float]]]
    n_train: int
    n_test: int


def run_spam_classifier(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
    top_n_tokens: int = 15,
) -> ClassificationReport:
    """Train a TF-IDF + Multinomial Naive Bayes classifier and report metrics.

    A lightweight, dependency-light baseline appropriate for spam/ham text
    classification; swap in a different estimator here without touching the
    UI or validation layers.
    """
    from sklearn.metrics import precision_recall_fscore_support

    work = df[[text_col, label_col]].dropna()
    X = work[text_col].astype(str)
    y = work[label_col].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            f"Label column '{label_col}' has only {y.nunique()} distinct value(s); "
            "at least 2 classes are required to train a classifier."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    classes = sorted(y.unique().tolist())
    accuracy = float((y_pred == y_test.values).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_tokens_per_class: dict[str, list[tuple[str, float]]] = {}
    for idx, cls in enumerate(model.classes_):
        log_probs = model.feature_log_prob_[idx]
        top_idx = np.argsort(log_probs)[-top_n_tokens:][::-1]
        top_tokens_per_class[str(cls)] = [
            (feature_names[i], float(log_probs[i])) for i in top_idx
        ]

    return ClassificationReport(
        text_col=text_col,
        label_col=label_col,
        classes=classes,
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=cm,
        top_tokens_per_class=top_tokens_per_class,
        n_train=len(X_train),
        n_test=len(X_test),
    )
