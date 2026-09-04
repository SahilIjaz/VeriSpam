"""Model evaluation: metrics, confusion matrix, classification report, ROC-AUC, timing."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.modeling import supports_decision_function, supports_predict_proba


@dataclass
class Metrics:
    classes: list[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    classification_report: str
    roc_auc: float | None
    roc_auc_note: str | None
    predict_time_sec: float
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None


def _positive_class(classes: list[str]) -> str:
    """Pick the 'positive' class for binary ROC-AUC — prefer an explicit spam-like label."""
    for candidate in ("spam", "1", "true", "yes"):
        for c in classes:
            if str(c).lower() == candidate:
                return c
    return classes[-1]


def _roc_auc(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, classes: list[str]):
    """Returns (roc_auc, note, fpr, tpr). roc_auc is None when it can't be computed."""
    from sklearn.metrics import roc_curve

    n_classes = len(classes)
    has_proba = supports_predict_proba(pipeline)
    has_decision = supports_decision_function(pipeline)

    if not has_proba and not has_decision:
        return None, "Not available: this model does not expose probability or decision scores.", None, None

    try:
        if n_classes == 2:
            pos_class = _positive_class(classes)
            if has_proba:
                class_index = list(pipeline.classes_).index(pos_class)
                scores = pipeline.predict_proba(X_test)[:, class_index]
            else:
                raw = np.atleast_1d(pipeline.decision_function(X_test))
                # decision_function is signed toward classes_[1]; flip if positive class is classes_[0]
                scores = raw if list(pipeline.classes_).index(pos_class) == 1 else -raw
            y_true_binary = (y_test == pos_class).astype(int)
            auc = float(roc_auc_score(y_true_binary, scores))
            fpr, tpr, _ = roc_curve(y_true_binary, scores)
            return auc, None, fpr, tpr

        # Multi-class: macro one-vs-rest, only meaningful with probability estimates.
        if has_proba:
            proba = pipeline.predict_proba(X_test)
            auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=pipeline.classes_))
            return auc, "Macro one-vs-rest average (multi-class).", None, None
        return None, "Not available: multi-class ROC-AUC requires probability estimates.", None, None
    except ValueError as exc:
        return None, f"Not available: {exc}", None, None


def metrics_from_saved(saved: dict) -> Metrics:
    """Reconstruct a lightweight Metrics view from a persisted model's metadata.

    A loaded (not freshly trained) model wasn't just re-evaluated against a
    live test split, so the confusion matrix / classification report / ROC
    curve points aren't available — only the summary numbers saved alongside
    it. The Evaluation page shows a note explaining the difference.
    """
    return Metrics(
        classes=saved.get("classes", []),
        accuracy=saved.get("accuracy", 0.0),
        precision=saved.get("precision", 0.0),
        recall=saved.get("recall", 0.0),
        f1=saved.get("f1", 0.0),
        confusion=None,
        classification_report="(not available — this model was loaded from disk, not freshly evaluated.)",
        roc_auc=saved.get("roc_auc"),
        roc_auc_note=None,
        predict_time_sec=saved.get("predict_time_sec", 0.0),
        fpr=None,
        tpr=None,
    )


def compute_metrics(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, classes: list[str]) -> Metrics:
    from sklearn.metrics import classification_report as sk_classification_report

    start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - start

    accuracy = float((y_pred == y_test.values).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0, labels=classes
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    report_text = sk_classification_report(y_test, y_pred, labels=classes, zero_division=0)

    roc_auc, roc_note, fpr, tpr = _roc_auc(pipeline, X_test, y_test, classes)

    return Metrics(
        classes=classes,
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=cm,
        classification_report=report_text,
        roc_auc=roc_auc,
        roc_auc_note=roc_note,
        predict_time_sec=predict_time,
        fpr=fpr,
        tpr=tpr,
    )
