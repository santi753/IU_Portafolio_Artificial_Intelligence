"""Tests for the news preprocessing (newsentiment.preprocessing)."""

import pandas as pd
import pytest

from newsentiment.preprocessing import OUTPUT_COLUMNS, REPORT_KEYS, preprocess


@pytest.fixture
def raw():
    """One synthetic feed for "Tesla" with one row per rule (comment = expected outcome)."""
    rows = [
        ("Tesla shares rise after deliveries - Reuters", "Reuters"),      # kept, suffix removed
        ("", "CNBC"),                                                     # a) empty title
        ("Tesla announces Q3 results - Tesla", " tesla "),                # b) own press release
        ("TSLA Stock Price and Chart — NASDAQ:TSLA - TradingView", "TradingView"),  # c) quote page
        ("Apple unveils new iPhone - CNBC", "CNBC"),                      # d) off-topic
        ("Tesla's Cybertruck is recalled - Teslarati", "Teslarati"),      # kept: Tesla's + Teslarati
        ("TESLA shares rise after deliveries! - Bloomberg", "Bloomberg"),  # e) duplicate of row 0
        ("Tesla faces probe - Electrek", "Reuters"),                      # kept, suffix not the source
    ]
    return pd.DataFrame({
        "company": "Tesla",
        "title": [t for t, _ in rows],
        "source": [s for _, s in rows],
        "published": pd.to_datetime(["2026-11-09"] * len(rows), utc=True),
        "link": [f"https://example.com/{i}" for i in range(len(rows))],
        "fetched_at": pd.Timestamp("2026-11-10", tz="UTC"),
    })


def test_kept_titles(raw):
    clean, _ = preprocess(raw, "Tesla")
    assert clean["title"].tolist() == [
        "Tesla shares rise after deliveries",
        "Tesla's Cybertruck is recalled",
        "Tesla faces probe - Electrek",
    ]
    assert clean["source"].tolist() == ["Reuters", "Teslarati", "Reuters"]


def test_report_counts(raw):
    _, report = preprocess(raw, "Tesla")
    assert list(report) == list(REPORT_KEYS)
    assert report == {
        "rows_in": 8,
        "empty_removed": 1,
        "press_releases_removed": 1,
        "quote_pages_removed": 1,
        "off_topic_removed": 1,
        "duplicates_removed": 1,
        "rows_out": 3,
    }


def test_output_columns_and_index(raw):
    clean, _ = preprocess(raw, "Tesla")
    assert list(clean.columns) == OUTPUT_COLUMNS
    assert clean.index.tolist() == [0, 1, 2]


def test_input_not_modified(raw):
    before = raw.copy()
    preprocess(raw, "Tesla")
    pd.testing.assert_frame_equal(raw, before)


def test_empty_input():
    empty = pd.DataFrame(columns=["company", "title", "source", "published", "link", "fetched_at"])
    clean, report = preprocess(empty, "Tesla")
    assert clean.empty
    assert list(clean.columns) == OUTPUT_COLUMNS
    assert report["rows_in"] == 0 and report["rows_out"] == 0
