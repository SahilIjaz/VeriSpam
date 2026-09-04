"""Centralized configuration: limits, defaults, and shared constants.

Keeping these in one place means every page and module tunes the same knobs
instead of scattering magic numbers across the codebase.
"""

from __future__ import annotations

from pathlib import Path

APP_TITLE = "Spam/Ham Classification Platform"

# --- File upload limits ---
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = (".csv", ".xls", ".xlsx")

# --- Train/test split defaults ---
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_STATE = 42
MIN_ROWS_TO_TRAIN = 10
MIN_ROWS_PER_CLASS_TO_STRATIFY = 2

# --- Class imbalance thresholds ---
IMBALANCE_WARNING_RATIO = 3.0   # majority:minority ratio above this -> warning
IMBALANCE_SEVERE_RATIO = 9.0    # above this -> flagged as severe imbalance

# --- Preprocessing ---
URL_PATTERN = r"https?://\S+|www\.\S+"
EMAIL_PATTERN = r"\S+@\S+\.\S+"
PHONE_PATTERN = r"\+?\d[\d\-\.\s]{7,}\d"
NUMBER_PATTERN = r"\d+"
WHITESPACE_PATTERN = r"\s+"
PUNCTUATION_PATTERN = r"[^\w\s]"
SPECIAL_CHARS_PATTERN = r"[^a-zA-Z0-9\s]"

# --- Modeling ---
TFIDF_MAX_FEATURES = 5000
TOP_N_EXPLAIN_WORDS = 10
TOP_N_EDA_WORDS = 20

# --- Persistence ---
SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"

# --- Display ---
CLASS_COLORS = {"spam": "#e05252", "ham": "#3aa76d"}
