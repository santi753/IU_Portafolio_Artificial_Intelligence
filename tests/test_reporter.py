"""Tests for the reporter component (newsentiment.reporter) on a synthetic labelled frame."""

import pandas as pd
import pytest

from newsentiment import config, reporter
from newsentiment.reporter import format_report, save_results, summarise


@pytest.fixture
def labelled() -> pd.DataFrame:
    """10 classified headlines: 5 positive, 3 negative, 2 neutral; 2 low confidence."""
    sentiments = ["positive"] * 5 + ["negative"] * 3 + ["neutral"] * 2
    confidence = [0.9, 0.8, 0.7, 0.6, 0.45, 0.9, 0.8, 0.4, 0.7, 0.6]
    return pd.DataFrame(
        {
            "company": "Acme",
            "title": [f"Headline number {i} " + "x" * (100 if i == 0 else 0) for i in range(10)],
            "source": "Example",
            "published": pd.date_range("2026-11-01", periods=10, freq="D", tz="UTC"),
            "link": "https://example.com",
            "sentiment": sentiments,
            "confidence": confidence,
            "low_confidence": [c < config.LOW_CONFIDENCE_THRESHOLD for c in confidence],
        }
    )


def test_counts_and_shares(labelled):
    s = summarise(labelled)
    assert s["n"] == 10
    assert (s["count_negative"], s["count_neutral"], s["count_positive"]) == (3, 2, 5)
    assert sum(s[f"count_{label}"] for label in config.LABELS) == s["n"]
    assert sum(s[f"share_{label}"] for label in config.LABELS) == pytest.approx(1.0)
    assert s["n_low_confidence"] == 2


def test_net_sentiment(labelled):
    # (5 - 3) / 10 * 100 = 20.0
    assert summarise(labelled)["net_sentiment"] == 20.0


def test_empty_input(labelled):
    s = summarise(labelled.iloc[0:0])
    assert s["n"] == 0
    assert s["net_sentiment"] is None
    assert s["mean_confidence"] is None
    assert all(s[f"share_{label}"] is None for label in config.LABELS)
    assert "No headlines" in format_report(labelled.iloc[0:0], s, "Acme")


def test_format_report(labelled):
    text = format_report(labelled, summarise(labelled), "Acme", top_n=3)
    assert "Acme" in text
    assert "50.0%" in text and "30.0%" in text and "20.0%" in text
    assert "+20.0" in text
    # Newest headline (2026-11-10) is listed, the oldest is not
    assert "2026-11-10" in text and "Headline number 0" not in text
    # Low-confidence marker on the 0.400 headline (2026-11-08 is within the top 3)
    assert "0.400*" in text


def test_title_is_cut():
    assert len(reporter._cut("a" * 200)) == reporter.TITLE_WIDTH


def test_save_results(labelled, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    s = summarise(labelled)
    headlines_path, summary_path = save_results(labelled, s, "Acme Corp")
    assert headlines_path.exists() and summary_path.exists()
    assert headlines_path.parent == tmp_path
    assert headlines_path.name.startswith("acme-corp_") and headlines_path.name.endswith("_headlines.csv")
    assert summary_path.name.endswith("_summary.csv")
    saved = pd.read_csv(headlines_path)
    assert list(saved.columns) == list(reporter.HEADLINE_COLUMNS)
    assert len(saved) == 10
    row = pd.read_csv(summary_path)
    assert len(row) == 1 and row.loc[0, "n"] == 10 and row.loc[0, "company"] == "Acme Corp"
