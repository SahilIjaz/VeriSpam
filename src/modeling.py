"""Model registry, train/test splitting, and training: the leakage-safe core.

Leakage guard: TF-IDF vectorization is wrapped inside an sklearn `Pipeline`
together with the classifier, and the pipeline is only ever `.fit()` on the
training split. The vectorizer's vocabulary/IDF weights are therefore learned
exclusively from `X_train`; `X_test` (and any later batch-prediction data)
only ever passes through `.transform()`. The same `SplitData` (same
train/test rows) is reused across every model trained in a session, so model
comparisons are apples-to-apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    MIN_ROWS_TO_TRAIN,
    TFIDF_MAX_FEATURES,
)

MODEL_REGISTRY: dict[str, "type"] = {
    "Naive Bayes": lambda: MultinomialNB(),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Linear SVM": lambda: LinearSVC(),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1
    ),
}


@dataclass
class SplitData:
    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    stratified: bool
    classes: list[str]


@dataclass
class TrainedModel:
    model_key: str
    pipeline: Pipeline
    classes: list[str]
    train_time_sec: float
    n_train: int
    n_test: int
    trained_at: str


def prepare_data(df: pd.DataFrame, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    """Drop rows missing text or label, and coerce both columns to string."""
    work = df[[text_col, label_col]].dropna()
    X = work[text_col].astype(str)
    y = work[label_col].astype(str)
    return X, y


def validate_trainable(y: pd.Series) -> None:
    """Raise a clear, catchable ValueError if the label column can't support training."""
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError(
            f"The label column has only {n_classes} distinct value(s); "
            "at least 2 classes are required to train a classifier."
        )
    if len(y) < MIN_ROWS_TO_TRAIN:
        raise ValueError(
            f"Only {len(y)} labeled row(s) available; at least {MIN_ROWS_TO_TRAIN} are "
            "required to train a classifier."
        )


def make_split(
    X: pd.Series,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> SplitData:
    """Single shared train/test split, reused for every model in a session.

    Falls back to a non-stratified split (with `stratified=False` on the
    result) if a class is too small to stratify, instead of raising.

    `classes` is computed from the *full* label column, not just `y_train` —
    this matters because a non-stratified fallback split on a small/imbalanced
    dataset can (by chance) put every example of a rare class into the test
    set and none into training. If `classes` were taken from `y_train` alone,
    a later `confusion_matrix(y_test, y_pred, labels=classes)` call could end
    up with zero overlap between `labels` and `y_test` and raise a cryptic
    sklearn error instead of a clear one.
    """
    validate_trainable(y)
    classes = sorted(y.unique().tolist())
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        stratified = True
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        stratified = False

    if y_train.nunique() < 2:
        raise ValueError(
            "This dataset is too small or too imbalanced to split reliably: the training "
            "set would end up with examples of only one class. Try a smaller test set size, "
            "a different random seed, or add more examples of the minority class."
        )

    return SplitData(X_train, X_test, y_train, y_test, stratified=stratified, classes=classes)


def build_pipeline(model_key: str) -> Pipeline:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Available: {', '.join(MODEL_REGISTRY)}")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)),
            ("clf", MODEL_REGISTRY[model_key]()),
        ]
    )


def train_model(split: SplitData, model_key: str) -> TrainedModel:
    """Fit a model on the training split only (see module docstring re: leakage)."""
    pipeline = build_pipeline(model_key)
    start = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    train_time = time.perf_counter() - start
    return TrainedModel(
        model_key=model_key,
        pipeline=pipeline,
        classes=split.classes,
        train_time_sec=train_time,
        n_train=len(split.X_train),
        n_test=len(split.X_test),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def supports_predict_proba(pipeline: Pipeline) -> bool:
    return hasattr(pipeline.named_steps["clf"], "predict_proba")


def supports_decision_function(pipeline: Pipeline) -> bool:
    return hasattr(pipeline.named_steps["clf"], "decision_function")


def get_confidence_scores(pipeline: Pipeline, X: pd.Series) -> np.ndarray | None:
    """Per-sample confidence in [0, 1] for the predicted class, or None if unsupported."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = pipeline.predict_proba(X)
        return proba.max(axis=1)
    if hasattr(clf, "decision_function"):
        scores = pipeline.decision_function(X)
        scores = np.atleast_1d(scores)
        if scores.ndim == 1:
            # Binary: squash the margin into a (0, 1)-ish confidence via a logistic curve.
            return 1.0 / (1.0 + np.exp(-np.abs(scores)))
        return 1.0 / (1.0 + np.exp(-np.abs(scores).max(axis=1)))
    return None
