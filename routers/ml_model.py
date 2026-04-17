"""
ML model loading and prediction.

Loads the serialized sklearn pipeline and training stats at startup.
Provides a predict function that takes extracted features and returns a price.
"""

import json
import logging

import joblib
import pandas as pd

from .config import MODELS_DIR
from .schemas import ExtractedFeatures

logger = logging.getLogger(__name__)

# Module-level singletons — loaded once at import / startup
_pipeline = None
_training_stats = None


def load_model() -> None:
    """Load the pipeline and training stats from disk. Call once at startup."""
    global _pipeline, _training_stats

    pipeline_path = MODELS_DIR / "pipeline.joblib"
    stats_path = MODELS_DIR / "training_stats.json"

    _pipeline = joblib.load(pipeline_path)
    logger.info("Loaded pipeline from %s", pipeline_path)

    with open(stats_path) as f:
        _training_stats = json.load(f)
    logger.info("Loaded training stats from %s", stats_path)


def get_training_stats() -> dict:
    if _training_stats is None:
        load_model()
    return _training_stats


def get_pipeline():
    if _pipeline is None:
        load_model()
    return _pipeline


def predict_price(features: ExtractedFeatures) -> float:
    """
    Convert ExtractedFeatures into a DataFrame row and run through the pipeline.
    Missing features are left as NaN — the pipeline's imputer handles them.
    """
    pipeline = get_pipeline()
    stats = get_training_stats()

    feature_names = stats["features"]

    # Build a single-row dict with None → NaN for the pipeline
    row = {}
    for name in feature_names:
        val = getattr(features, name, None)
        row[name] = val

    df = pd.DataFrame([row])

    prediction = pipeline.predict(df)[0]
    return float(prediction)
