"""Classifier component: adds a sentiment label and a confidence to each headline.

The saved final pipeline (see :func:`newsentiment.model.load_model`) is applied
to the ``title`` column of a preprocessed news frame. The confidence is the
highest predicted class probability; predictions below
``config.LOW_CONFIDENCE_THRESHOLD`` are flagged so the report can mark them.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from newsentiment import config
from newsentiment.model import load_model

# Columns added by classify()
RESULT_COLUMNS = ("sentiment", "confidence", "low_confidence")


def classify(df: pd.DataFrame, model: Pipeline | None = None) -> pd.DataFrame:
    """Predict the sentiment of every headline in ``df``.

    Args:
        df: Preprocessed news frame with a ``title`` column (output of
            :func:`newsentiment.preprocessing.preprocess`).
        model: Fitted pipeline with ``predict_proba``. If None, the saved final
            model is loaded with :func:`load_model`.

    Returns:
        A copy of ``df`` with three extra columns: ``sentiment`` (predicted
        label), ``confidence`` (highest class probability, rounded to 3
        decimals) and ``low_confidence`` (True if confidence is below the
        threshold). The input frame is not changed.
    """
    result = df.copy()

    # Empty input: return the (empty) frame with correctly typed new columns
    if result.empty:
        result["sentiment"] = pd.Series(dtype=object)
        result["confidence"] = pd.Series(dtype=float)
        result["low_confidence"] = pd.Series(dtype=bool)
        return result

    if model is None:
        model = load_model()

    # Probabilities per class; columns follow model.classes_, so the label is
    # taken from there rather than from config.LABELS
    proba = model.predict_proba(result["title"].astype(str).tolist())
    best = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    result["sentiment"] = np.asarray(model.classes_)[best]
    result["confidence"] = confidence.round(3)
    # Flag on the rounded value, so the flag always agrees with the confidence shown
    result["low_confidence"] = result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    return result
