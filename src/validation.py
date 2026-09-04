"""Automated dataset validation layer.

Runs before a dataset is allowed into the data-mining pipeline. Produces a
structured ValidationResult (errors block progress, warnings do not) so the
UI layer can render the Upload -> Validation -> Verified/Failed workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.loader import LoadResult

MAX_MISSING_RATIO_ERROR = 0.5   # above this, warn strongly about overall sparsity (not a hard block)
MAX_MISSING_RATIO_WARNING = 0.1
MIN_ROWS = 2


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    n_unique: int


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    n_duplicates: int = 0
    n_missing_total: int = 0


def _profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    n_rows = len(df)
    profiles = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(df[col].dtype),
                missing_count=missing,
                missing_pct=(missing / n_rows * 100) if n_rows else 0.0,
                n_unique=int(df[col].nunique(dropna=True)),
            )
        )
    return profiles


def validate_dataset(load_result: LoadResult, required_columns: list[str] | None = None) -> ValidationResult:
    """Validate a loaded dataset and return a structured pass/fail result.

    `required_columns` lets a downstream pipeline (e.g. spam classification
    expecting a text + label column) enforce a schema; left empty, validation
    stays generic (format, readability, emptiness, quality thresholds).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # File-level checks (format/readability already happened in the loader).
    if not load_result.ok:
        return ValidationResult(is_valid=False, errors=[load_result.error or "Unknown load error."])

    df = load_result.dataframe
    assert df is not None

    if load_result.size_mb > 200:
        warnings.append(
            f"File is large ({load_result.size_mb:.1f} MB); processing may be slow."
        )

    # Schema / emptiness checks.
    n_rows, n_cols = df.shape
    if n_cols == 0:
        errors.append("The dataset has no columns.")
    if n_rows == 0:
        errors.append("The dataset has no rows (file is empty after parsing).")
    elif n_rows < MIN_ROWS:
        errors.append(f"The dataset only has {n_rows} row(s); at least {MIN_ROWS} are required.")

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        errors.append(f"Duplicate column names found: {', '.join(sorted(set(duplicate_cols)))}")

    if n_cols == 0 or n_rows == 0:
        # Can't safely profile further; stop here.
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, n_rows=n_rows, n_cols=n_cols)

    # Required-columns check (schema enforcement for a specific pipeline).
    if required_columns:
        missing_required = [c for c in required_columns if c not in df.columns]
        if missing_required:
            errors.append(
                f"Missing required column(s): {', '.join(missing_required)}. "
                f"Found columns: {', '.join(map(str, df.columns))}"
            )

    # Fully-empty columns/rows.
    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(f"Column(s) entirely empty: {', '.join(empty_cols)}")

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        warnings.append(f"{empty_rows} row(s) are completely empty.")

    # Missing-value quality thresholds.
    n_missing_total = int(df.isna().sum().sum())
    overall_missing_ratio = n_missing_total / (n_rows * n_cols) if n_rows * n_cols else 0
    if overall_missing_ratio > MAX_MISSING_RATIO_ERROR:
        warnings.append(
            f"Dataset is {overall_missing_ratio * 100:.1f}% missing values overall "
            f"(a small number of very sparse columns can drive this up without the "
            f"dataset itself being unusable — check the per-column breakdown)."
        )
    elif overall_missing_ratio > MAX_MISSING_RATIO_WARNING:
        warnings.append(
            f"Dataset has a notable amount of missing data "
            f"({overall_missing_ratio * 100:.1f}% of all cells)."
        )

    column_profiles = _profile_columns(df)
    for profile in column_profiles:
        if profile.missing_pct > MAX_MISSING_RATIO_WARNING * 100 and str(profile.name) not in empty_cols:
            warnings.append(
                f"Column '{profile.name}' has {profile.missing_pct:.1f}% missing values."
            )

    # Duplicate rows.
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        dup_ratio = n_duplicates / n_rows
        msg = f"{n_duplicates} duplicate row(s) found ({dup_ratio * 100:.1f}% of the dataset)."
        if dup_ratio > 0.5:
            errors.append(msg)
        else:
            warnings.append(msg)

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        column_profiles=column_profiles,
        n_rows=n_rows,
        n_cols=n_cols,
        n_duplicates=n_duplicates,
        n_missing_total=n_missing_total,
    )
