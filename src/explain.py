"""Per-prediction explanations: which words drove a spam/ham verdict.

No SHAP dependency — each model family already exposes weights that are
cheap and fast to read directly:
  - Naive Bayes: difference in per-class log-probability for each word.
  - Logistic Regression / Linear SVM: linear coefficient for each word.
  - Random Forest: feature importances (global, not instance-specific —
    called out explicitly, since true per-instance tree explanations need
    something like SHAP's TreeExplainer, which we're deliberately not adding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import TOP_N_EXPLAIN_WORDS


@dataclass
class WordContribution:
    word: str
    weight: float


@dataclass
class Explanation:
    predicted_class: str
    top_words: list[WordContribution]
    method: str
    is_instance_specific: bool


def explain_prediction(pipeline: Pipeline, text: str, top_n: int = TOP_N_EXPLAIN_WORDS) -> Explanation:
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    x_vec = vectorizer.transform([text])
    present_indices = x_vec.nonzero()[1]
    present_words = feature_names[present_indices]

    predicted_class = pipeline.predict([text])[0]
    classes = list(clf.classes_)
    class_idx = classes.index(predicted_class)

    if len(present_indices) == 0:
        return Explanation(
            predicted_class=str(predicted_class),
            top_words=[],
            method="No recognized vocabulary words found in this text.",
            is_instance_specific=True,
        )

    if isinstance(clf, MultinomialNB):
        log_probs = clf.feature_log_prob_  # (n_classes, n_features)
        if len(classes) == 2:
            diff = log_probs[class_idx] - log_probs[1 - class_idx]
        else:
            other_mean = np.delete(log_probs, class_idx, axis=0).mean(axis=0)
            diff = log_probs[class_idx] - other_mean
        values = diff[present_indices]
        method = "Naive Bayes: log-probability difference (how much each word favors this class)"
        instance_specific = True
    elif isinstance(clf, (LogisticRegression, LinearSVC)):
        coef = clf.coef_
        row = coef[0] if coef.shape[0] == 1 and class_idx == 1 else (
            -coef[0] if coef.shape[0] == 1 else coef[class_idx]
        )
        values = row[present_indices]
        method = "Linear model coefficient (how much each word pushes the score toward this class)"
        instance_specific = True
    elif isinstance(clf, RandomForestClassifier):
        values = clf.feature_importances_[present_indices]
        method = "Random Forest feature importance (global signal strength, not specific to this message)"
        instance_specific = False
    else:
        return Explanation(
            predicted_class=str(predicted_class),
            top_words=[],
            method="No explanation available for this model type.",
            is_instance_specific=False,
        )

    contributions = [
        WordContribution(word=str(w), weight=float(v)) for w, v in zip(present_words, values)
    ]
    contributions.sort(key=lambda c: abs(c.weight), reverse=True)

    return Explanation(
        predicted_class=str(predicted_class),
        top_words=contributions[:top_n],
        method=method,
        is_instance_specific=instance_specific,
    )
