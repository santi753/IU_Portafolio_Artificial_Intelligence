"""Tests for the model pipelines (newsentiment.model) and out-of-fold predictions."""

import json

import numpy as np
import pandas as pd
import pytest

from newsentiment import config
from newsentiment.evaluation import cross_val_predictions
from newsentiment.model import build_pipeline, key_params

# Each class gets its own signal words, so even a tiny model can learn them
WORDS = {
    "negative": ["loss fell down", "profit decreased sharply", "sales dropped"],
    "neutral": ["company is based in", "the board met", "report was published"],
    "positive": ["profit rose up", "sales increased strongly", "earnings grew"],
}


@pytest.fixture
def tiny_data() -> tuple[pd.Series, pd.Series]:
    """60 synthetic sentences, 20 per class."""
    rows = [
        (label, f"{WORDS[label][i % 3]} in quarter {i}")
        for label in config.LABELS
        for i in range(20)
    ]
    df = pd.DataFrame(rows, columns=["label", "text"]).sample(frac=1, random_state=0)
    return df["text"].reset_index(drop=True), df["label"].reset_index(drop=True)


@pytest.mark.parametrize("classifier", ["nb", "logreg"])
def test_pipeline_fits_and_predicts(classifier, tiny_data):
    """Both classifiers build, fit, predict known labels and give valid probabilities."""
    X, y = tiny_data
    pipeline = build_pipeline(classifier).fit(X, y)

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= set(config.LABELS)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), len(config.LABELS))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("classifier", ["nb", "logreg"])
def test_stop_words_are_kept(classifier):
    """No stop-word list: words like 'not', 'up' and 'down' must stay in the features."""
    pipeline = build_pipeline(classifier)
    tfidf = pipeline.named_steps["tfidf"]
    assert tfidf.stop_words is None
    assert tfidf.lowercase is True


def test_params_reach_the_steps():
    """Keyword parameters are forwarded to the pipeline steps via set_params."""
    pipeline = build_pipeline("logreg", clf__class_weight="balanced", tfidf__ngram_range=(1, 2))
    assert pipeline.named_steps["clf"].class_weight == "balanced"
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)

    nb = build_pipeline("nb", clf__alpha=0.5)
    assert nb.named_steps["clf"].alpha == 0.5


def test_unknown_classifier_raises():
    """An unknown classifier name is rejected."""
    with pytest.raises(ValueError, match="Unknown classifier"):
        build_pipeline("svm")


@pytest.mark.parametrize(
    "pipeline",
    [
        build_pipeline("nb"),
        build_pipeline("logreg", clf__class_weight="balanced", tfidf__ngram_range=(1, 2)),
    ],
)
def test_key_params_is_json_serialisable(pipeline):
    """key_params returns plain JSON values, including the tuple ngram_range."""
    params = key_params(pipeline)
    assert json.loads(json.dumps(params)) == params
    assert "tfidf__ngram_range" in params
    if params["classifier"] == "MultinomialNB":
        assert "clf__alpha" in params
    else:
        assert {"clf__C", "clf__class_weight"} <= params.keys()


def test_cross_val_predictions_one_per_input(tiny_data):
    """Out-of-fold predictions: exactly one valid label per input row."""
    X, y = tiny_data
    predictions = cross_val_predictions(build_pipeline("logreg"), X, y)
    assert len(predictions) == len(X)
    assert set(predictions) <= set(config.LABELS)
