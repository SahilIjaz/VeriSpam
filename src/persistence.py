"""Save and load trained pipelines to disk, with metadata sidecars.

Only fitted pipelines + their metrics/metadata are ever written here; raw
uploaded datasets are never persisted (see privacy notes in the About page).
"""

from __future__ import annotations

import json
import re
import time

import joblib
from sklearn.pipeline import Pipeline

from src.config import SAVED_MODELS_DIR


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "model"


def save_pipeline(pipeline: Pipeline, model_key: str, metadata: dict) -> str:
    """Save a fitted pipeline + metadata; returns the slug it was saved under."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{_slugify(model_key)}-{int(time.time())}"

    joblib.dump(pipeline, SAVED_MODELS_DIR / f"{slug}.joblib")
    full_metadata = {"slug": slug, "model_key": model_key, **metadata}
    (SAVED_MODELS_DIR / f"{slug}.json").write_text(json.dumps(full_metadata, indent=2, default=str))
    return slug


def list_saved_models() -> list[dict]:
    """Metadata for every saved model, newest first."""
    if not SAVED_MODELS_DIR.exists():
        return []
    entries = []
    for meta_path in SAVED_MODELS_DIR.glob("*.json"):
        try:
            entries.append(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    entries.sort(key=lambda e: e.get("slug", ""), reverse=True)
    return entries


def load_pipeline(slug: str) -> tuple[Pipeline, dict]:
    joblib_path = SAVED_MODELS_DIR / f"{slug}.joblib"
    meta_path = SAVED_MODELS_DIR / f"{slug}.json"
    if not joblib_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No saved model found for '{slug}'.")
    pipeline = joblib.load(joblib_path)
    metadata = json.loads(meta_path.read_text())
    return pipeline, metadata


def delete_pipeline(slug: str) -> None:
    (SAVED_MODELS_DIR / f"{slug}.joblib").unlink(missing_ok=True)
    (SAVED_MODELS_DIR / f"{slug}.json").unlink(missing_ok=True)
