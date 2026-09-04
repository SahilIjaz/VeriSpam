"""Best-effort detection of which columns hold message text and the spam/ham label.

Kept dataset-agnostic (no hardcoded dataset-specific column names beyond a
list of common candidates) so the app works with different spam/ham datasets,
not just one.
"""

from __future__ import annotations

import pandas as pd

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
