"""Tests for the classifier component (newsentiment.classifier), using the real saved model."""

import pandas as pd
import pytest

from newsentiment import config
from newsentiment.classifier import classify
from newsentiment.model import load_model


@pytest.fixture(scope="module")
def model():
    """The saved final pipeline, loaded once for all tests."""
    return load_model()


@pytest.fixture
def news() -> pd.DataFrame:
    """A small preprocessed-looking news frame."""
    titles = [
        "Operating profit rose sharply compared with the previous year",
        "The company will cut 2,000 jobs after quarterly losses widened",
        "The annual general meeting will be held in Helsinki",
        "Siemens",
    ]
    return pd.DataFrame(
        {
            "company": "Test",
            "title": titles,
            "source": "Example",
            "published": pd.to_datetime(["2026-11-01"] * len(titles), utc=True),
            "link": "https://example.com",
        }
    )


def test_labels_and_confidence_valid(model, news):
    result = classify(news, model)
    assert len(result) == len(news)
    assert result["sentiment"].isin(config.LABELS).all()
    assert result["confidence"].between(0, 1).all()
    # With three classes the highest probability is at least 1/3
    assert (result["confidence"] >= 1 / 3 - 1e-3).all()


def test_low_confidence_flag_matches_threshold(model, news):
    result = classify(news, model)
    expected = result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    assert (result["low_confidence"] == expected).all()


def test_input_unchanged(model, news):
    before = news.copy()
    classify(news, model)
    pd.testing.assert_frame_equal(news, before)


def test_loads_model_when_none_given(news):
    result = classify(news.head(1))
    assert result["sentiment"].iloc[0] in config.LABELS


def test_empty_input(model, news):
    result = classify(news.iloc[0:0], model)
    assert result.empty
    for column in ("sentiment", "confidence", "low_confidence"):
        assert column in result.columns
