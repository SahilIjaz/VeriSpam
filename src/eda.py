"""Exploratory data analysis helpers: distributions, common words, data-quality breakdowns.

Purely descriptive (no fitting), so it's safe to run on the whole dataset —
these are analysis outputs for the user to look at, not features derived for
a model.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def dataset_overview(df: pd.DataFrame) -> dict:
    """Generic exploratory summary that applies to any tabular dataset."""
    numeric_df = df.select_dtypes(include="number")
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "numeric_summary": numeric_df.describe().transpose() if not numeric_df.empty else None,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }


def class_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(dropna=False)


def text_stats(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Per-row character count, word count, and average word length."""
    text = df[text_col].astype(str)
    char_count = text.str.len()
    word_count = text.str.split().str.len()
    return pd.DataFrame(
        {
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": (char_count / word_count.replace(0, pd.NA)).fillna(0),
        }
    )


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower())


def most_common_words(texts: pd.Series, top_n: int = 20, exclude_stopwords: bool = True) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for text in texts.astype(str):
        tokens = tokenize(text)
        if exclude_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
        counter.update(tokens)
    common = counter.most_common(top_n)
    return pd.DataFrame(common, columns=["word", "count"])


def most_common_words_by_class(
    df: pd.DataFrame, text_col: str, label_col: str, top_n: int = 20
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for cls, group in df.groupby(label_col, dropna=True):
        result[str(cls)] = most_common_words(group[text_col], top_n=top_n)
    return result


def missing_value_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100) if len(df) else missing * 0
    return (
        pd.DataFrame({"column": missing.index, "missing_count": missing.values, "missing_pct": pct.values})
        .sort_values("missing_count", ascending=False)
        .reset_index(drop=True)
    )


def duplicate_summary(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Returns (duplicate row count, a preview of the duplicated rows)."""
    dup_mask = df.duplicated(keep=False)
    return int(df.duplicated().sum()), df[dup_mask].head(50)
