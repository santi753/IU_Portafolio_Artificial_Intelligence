"""Tests for loading and cleaning the training data (newsentiment.data)."""

import pandas as pd
import pytest

from newsentiment import config
from newsentiment.data import (
    clean_training_data,
    load_agreement_levels,
    load_raw_training_data,
    load_training_data,
)


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


def _write_agreement_files(folder, contents: dict[int, list[str]]) -> dict:
    """Write tiny cumulative agreement files (latin-1, Windows line endings)."""
    files = {}
    for level, lines in contents.items():
        path = folder / f"Sentences_{level}Agree.txt"
        path.write_bytes("\r\n".join(lines).encode("latin-1") + b"\r\n")
        files[level] = path
    return files


def test_agreement_levels_values(tmp_path):
    """Highest matching level wins (text and label), including non-ASCII normalisation and an '@' inside the text."""
    all_lines = ["Profit rose .@positive"]
    l75 = all_lines + ["Sales fell .@negative"]
    l66 = l75 + ["The board met .@neutral"]
    l50 = l66 + ["Caf\xe9 Oy opened .@neutral", "Email a@b .@positive"]
    files = _write_agreement_files(tmp_path, {100: all_lines, 75: l75, 66: l66, 50: l50})

    df = make_df([
        ("neutral", "The board met ."),
        ("positive", "Profit rose ."),
        ("neutral", "Caf\u00ff Oy opened ."),  # other non-ASCII character, same position
        ("negative", "Sales fell ."),
        ("positive", "Email a@b ."),
    ])
    levels = load_agreement_levels(df, files=files)
    assert levels.tolist() == [66, 100, 50, 75, 50]
    assert levels.dtype.kind == "i"
    assert levels.index.equals(df.index)


def test_agreement_levels_unmatched_raises(tmp_path):
    """A row missing from all files, or with a different label, raises ValueError with the count."""
    files = _write_agreement_files(tmp_path, {100: ["Up .@positive"], 50: ["Up .@positive"]})
    df = make_df([("positive", "Up ."), ("negative", "Up ."), ("neutral", "Unknown .")])
    with pytest.raises(ValueError, match="2 of 3 rows"):
        load_agreement_levels(df, files=files)


def test_real_agreement_files():
    """Integration test: every cleaned sentence is matched, with the known band counts."""
    files = [config.TRAINING_FILE, *config.AGREEMENT_FILES.values()]
    missing = [p for p in files if not p.is_file()]
    if missing:
        pytest.skip(f"Data files not available: {missing}")
    cleaned, _ = load_training_data()
    levels = load_agreement_levels(cleaned)
    assert len(levels) == len(cleaned) == 4836
    assert levels.value_counts().to_dict() == {100: 2259, 75: 1189, 66: 761, 50: 627}
