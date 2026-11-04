"""Tests for loading and cleaning the training data (newsentiment.data)."""

import pandas as pd
import pytest

from newsentiment import config
from newsentiment.data import clean_training_data, load_raw_training_data, load_training_data


def make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a small raw-style DataFrame from (label, text) pairs."""
    return pd.DataFrame(rows, columns=["label", "text"])


def test_exact_duplicates_removed():
    """Identical (label, text) rows are reduced to one, keeping the first."""
    df = make_df([
        ("positive", "Sales rose."),
        ("neutral", "The meeting is on Monday."),
        ("positive", "Sales rose."),
    ])
    cleaned, report = clean_training_data(df)
    assert cleaned["text"].tolist() == ["Sales rose.", "The meeting is on Monday."]
    assert report["duplicate_rows_removed"] == 1


def test_conflicting_text_removed_completely():
    """Both rows of a text that carries two different labels are removed."""
    df = make_df([
        ("positive", "Profit doubled."),
        ("negative", "Profit doubled."),
        ("neutral", "The office moved."),
    ])
    cleaned, report = clean_training_data(df)
    assert "Profit doubled." not in cleaned["text"].tolist()
    assert len(cleaned) == 1
    assert report["conflicting_texts"] == 1
    assert report["conflicting_rows_removed"] == 2


def test_whitespace_stripped_and_empty_removed():
    """Surrounding whitespace is stripped and empty / blank texts are dropped."""
    df = make_df([
        ("neutral", "  Shares unchanged.  "),
        ("negative", "   "),
        ("positive", ""),
    ])
    cleaned, report = clean_training_data(df)
    assert cleaned["text"].tolist() == ["Shares unchanged."]
    assert report["empty_removed"] == 2


def test_stripping_reveals_duplicates():
    """Texts differing only in surrounding whitespace count as duplicates."""
    df = make_df([("neutral", "Same text."), ("neutral", " Same text. ")])
    cleaned, report = clean_training_data(df)
    assert len(cleaned) == 1
    assert report["duplicate_rows_removed"] == 1


def test_report_counts():
    """All report fields are correct on a mixed example."""
    df = make_df([
        ("positive", "Up."),
        ("positive", "Up."),            # duplicate
        ("negative", "Mixed."),
        ("neutral", "Mixed."),          # conflict with the row above
        ("neutral", " "),               # empty
        ("negative", "Down."),
        ("neutral", "Flat."),
    ])
    cleaned, report = clean_training_data(df)
    assert report == {
        "rows_in": 7,
        "empty_removed": 1,
        "conflicting_texts": 1,
        "conflicting_rows_removed": 2,
        "duplicate_rows_removed": 1,
        "rows_out": 3,
        "class_counts": {"negative": 1, "neutral": 1, "positive": 1},
    }
    assert list(report["class_counts"]) == list(config.LABELS)
    assert cleaned.index.tolist() == [0, 1, 2]


def test_input_not_modified():
    """clean_training_data leaves its input DataFrame untouched."""
    df = make_df([
        ("positive", " Up. "),
        ("positive", " Up. "),
        ("negative", "Mixed."),
        ("neutral", "Mixed."),
    ])
    original = df.copy()
    clean_training_data(df)
    pd.testing.assert_frame_equal(df, original)


def test_invalid_label_raises(tmp_path):
    """A CSV containing a label outside config.LABELS raises ValueError."""
    path = tmp_path / "bad.csv"
    path.write_text("positive,Good news.\nbullish,Very good news.\n", encoding="latin-1")
    with pytest.raises(ValueError, match="bullish"):
        load_raw_training_data(path)


def test_missing_file_raises(tmp_path):
    """A missing file raises FileNotFoundError that points to the Kaggle page."""
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        load_raw_training_data(tmp_path / "does-not-exist.csv")


def test_real_training_file():
    """Integration test: cleaning the real Financial PhraseBank file gives the known counts."""
    if not config.TRAINING_FILE.is_file():
        pytest.skip(f"Training file not available at {config.TRAINING_FILE}")
    cleaned, report = load_training_data()
    assert report["rows_in"] == 4846
    assert report["duplicate_rows_removed"] == 6
    assert report["conflicting_texts"] == 2
    assert report["conflicting_rows_removed"] == 4
    assert report["rows_out"] == 4836
    assert report["class_counts"] == {"negative": 604, "neutral": 2871, "positive": 1361}
    assert len(cleaned) == 4836
