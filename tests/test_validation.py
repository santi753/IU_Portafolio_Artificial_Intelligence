"""Tests for the real-news validation set (newsentiment.validation).

All tests use synthetic news caches and label files in tmp_path; the config
paths are redirected there, so the real data/validation folder is never touched.
"""

import hashlib

import pandas as pd
import pytest

from newsentiment import config, validation
from newsentiment.news_fetcher import NEWS_COLUMNS, slugify
from newsentiment.validation import (
    COMPANIES,
    SHEET_COLUMNS,
    check_frozen,
    create_labelling_sheet,
    freeze_labels,
    latest_cache,
    load_labels,
)

OLD_STAMP, NEW_STAMP = "20261110-1651", "20261114-1013"
N_ROWS = len(COMPANIES) * config.VALIDATION_PER_COMPANY  # 30


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Redirect the news cache and validation files to tmp_path."""
    news_dir, val_dir = tmp_path / "news", tmp_path / "validation"
    news_dir.mkdir()
    val_dir.mkdir()
    monkeypatch.setattr(config, "NEWS_DIR", news_dir)
    monkeypatch.setattr(config, "VALIDATION_SHEET", val_dir / "real_news_to_label.csv")
    monkeypatch.setattr(config, "VALIDATION_LABELLED", val_dir / "real_news_labelled.csv")
    monkeypatch.setattr(config, "VALIDATION_HASH", val_dir / "real_news_labelled.sha256")
    return tmp_path


def write_cache(company: str, stamp: str, tag: str, n: int = 15):
    """Write a cache file like save_news does; every title contains ``tag``."""
    titles = [f"{company} headline number {i} {tag}" for i in range(n)]
    df = pd.DataFrame(
        {
            "company": company,
            # The feed appends the publisher; preprocess() removes it
            "title": [f"{t} - Example News" for t in titles],
            "source": "Example News",
            "published": pd.date_range("2026-11-01 08:30", periods=n, freq="h", tz="UTC"),
            "link": [f"https://example.com/{tag}/{i}" for i in range(n)],
        },
        columns=NEWS_COLUMNS,
    )
    df["fetched_at"] = pd.Timestamp("2026-11-14 10:13", tz="UTC")
    path = config.NEWS_DIR / f"{slugify(company)}_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def caches(paths):
    """An older and a newer cache for each company."""
    for company in COMPANIES:
        write_cache(company, OLD_STAMP, "old")
        write_cache(company, NEW_STAMP, "new")


def labelled_frame() -> pd.DataFrame:
    """A valid labelled sheet with 30 rows; labels vary in case and spacing."""
    raw_labels = [" Positive", "NEGATIVE ", "neutral"]
    return pd.DataFrame(
        {
            "id": range(1, N_ROWS + 1),
            "company": [COMPANIES[i % 3] for i in range(N_ROWS)],
            "published": "2026-11-12",
            "source": "Example News",
            # Comma, quotes and an accent check quoting and encoding
            "title": [f'Siemens Énergie order {i}, "record" year' for i in range(N_ROWS)],
            "label": [raw_labels[i % 3] for i in range(N_ROWS)],
        }
    )


def write_labels(df: pd.DataFrame, sep: str = ",", encoding: str = "utf-8-sig"):
    """Write a labelled file to config.VALIDATION_LABELLED."""
    df.to_csv(config.VALIDATION_LABELLED, index=False, sep=sep, encoding=encoding)
    return config.VALIDATION_LABELLED


# --- latest_cache ----------------------------------------------------------


def test_latest_cache_picks_newest(caches):
    """The file with the latest time stamp is returned."""
    assert latest_cache("Tesla").name == f"tesla_{NEW_STAMP}.csv"


def test_latest_cache_missing_raises(paths):
    """A company without a cache file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Tesla"):
        latest_cache("Tesla")


# --- create_labelling_sheet ------------------------------------------------


def test_sheet_has_n_rows_per_company(caches):
    """n rows per company, ids 1..3n, all from the newest caches."""
    sheet = create_labelling_sheet(n=4)
    assert len(sheet) == 12
    assert sheet["company"].value_counts().to_dict() == {c: 4 for c in COMPANIES}
    assert sheet["id"].tolist() == list(range(1, 13))
    assert sheet["title"].str.endswith(" new").all()


def test_sheet_has_no_prediction_columns(caches):
    """Only the sheet columns, with an empty label column, in memory and on disk."""
    sheet = create_labelling_sheet(n=4)
    on_disk = pd.read_csv(config.VALIDATION_SHEET, encoding="utf-8-sig",
                          dtype=str, keep_default_na=False)
    for df in (sheet, on_disk):
        assert list(df.columns) == SHEET_COLUMNS
        assert not any("predict" in c or "confidence" in c for c in df.columns)
        assert (df["label"] == "").all()


def test_sheet_same_seed_same_result(caches):
    """Running twice on the same caches gives the same sheet."""
    first = create_labelling_sheet(n=4)
    second = create_labelling_sheet(n=4)
    pd.testing.assert_frame_equal(first, second)


def test_sheet_is_preprocessed_with_dates_only(caches):
    """Publisher suffix removed, published reduced to YYYY-MM-DD, file has a BOM."""
    sheet = create_labelling_sheet(n=4)
    assert not sheet["title"].str.endswith(" - Example News").any()
    assert sheet["published"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert config.VALIDATION_SHEET.read_bytes().startswith(b"\xef\xbb\xbf")


def test_sheet_refuses_when_labelled_exists(caches):
    """An existing labelled file blocks a new sheet."""
    config.VALIDATION_LABELLED.write_text("already labelled", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_labelling_sheet(n=4)
    assert not config.VALIDATION_SHEET.exists()


def test_sheet_too_few_headlines_raises(caches):
    """A company with fewer than n cleaned headlines raises ValueError."""
    with pytest.raises(ValueError, match="Tesla"):
        create_labelling_sheet(n=20)


# --- load_labels -----------------------------------------------------------


@pytest.mark.parametrize(
    "sep, encoding",
    [(",", "utf-8"), (",", "utf-8-sig"), (";", "utf-8-sig"), (";", "cp1252")],
)
def test_load_labels_delimiters_and_encodings(paths, sep, encoding):
    """Comma and semicolon files, with or without BOM, and Excel's cp1252 are read."""
    write_labels(labelled_frame(), sep=sep, encoding=encoding)
    labels = load_labels()
    assert list(labels.columns) == SHEET_COLUMNS
    assert labels["id"].tolist() == list(range(1, N_ROWS + 1))
    assert labels["label"].tolist()[:3] == ["positive", "negative", "neutral"]
    assert labels["title"].iloc[0] == 'Siemens Énergie order 0, "record" year'


def test_load_labels_ignores_empty_excel_rows(paths):
    """Empty rows that Excel adds at the end are dropped."""
    path = write_labels(labelled_frame(), sep=";")
    with open(path, "a", encoding="utf-8") as f:
        f.write(";;;;;\n;;;;;\n")
    assert len(load_labels()) == N_ROWS


def test_load_labels_missing_label_raises(paths):
    """Empty labels raise ValueError naming the ids."""
    df = labelled_frame()
    df.loc[df["id"].isin([3, 7]), "label"] = " "
    write_labels(df)
    with pytest.raises(ValueError, match=r"missing label for ids \[3, 7\]"):
        load_labels()


def test_load_labels_invalid_label_raises(paths):
    """A label outside config.LABELS raises ValueError naming id and value."""
    df = labelled_frame()
    df.loc[df["id"] == 12, "label"] = "postive"
    write_labels(df)
    with pytest.raises(ValueError, match=r"invalid label for ids \[12\] \(12: 'postive'\)"):
        load_labels()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df[df["id"] != 30], r"missing ids \[30\]"),
        (lambda df: df.assign(id=df["id"].replace(30, 31)), r"unexpected ids \[31\]"),
        (lambda df: df.assign(id=df["id"].replace(30, 29)), r"duplicated ids \[29\]"),
    ],
    ids=["missing", "unexpected", "duplicated"],
)
def test_load_labels_wrong_ids_raise(paths, change, message):
    """Ids must be exactly 1..30."""
    write_labels(change(labelled_frame()))
    with pytest.raises(ValueError, match=message):
        load_labels()


# --- freeze_labels / check_frozen ------------------------------------------


def test_freeze_and_check_detect_change(paths):
    """Freezing stores the file's SHA-256; a later edit is detected."""
    path = write_labels(labelled_frame())
    digest = freeze_labels()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config.VALIDATION_HASH.read_text(encoding="utf-8").split() == [digest, path.name]
    check_frozen()  # unchanged: no error
    assert freeze_labels() == digest  # freezing again is a no-op

    df = labelled_frame()
    df.loc[df["id"] == 1, "label"] = "negative"
    write_labels(df)
    with pytest.raises(ValueError, match="changed"):
        check_frozen()
    with pytest.raises(ValueError, match="already frozen"):
        freeze_labels()


def test_freeze_invalid_labels_writes_no_hash(paths):
    """Invalid labels are not frozen."""
    df = labelled_frame()
    df.loc[df["id"] == 5, "label"] = ""
    write_labels(df)
    with pytest.raises(ValueError):
        freeze_labels()
    assert not config.VALIDATION_HASH.exists()


# --- command line ----------------------------------------------------------


def test_cli_sheet_prints_rows_per_company(caches, capsys):
    """'sheet' writes the sheet and prints the row count per company."""
    assert validation.main(["sheet"]) == 0
    out = capsys.readouterr().out
    for company in COMPANIES:
        assert f"{company:<10}{config.VALIDATION_PER_COMPANY:>4}" in out
    assert config.VALIDATION_SHEET.exists()


def test_cli_freeze_prints_hash_and_counts(paths, capsys):
    """'freeze' prints the hash and the counts per class."""
    write_labels(labelled_frame())
    assert validation.main(["freeze"]) == 0
    out = capsys.readouterr().out
    assert config.VALIDATION_HASH.read_text(encoding="utf-8").split()[0] in out
    assert f"{'positive':<10}{10:>4}" in out


def test_cli_refusal_returns_1(caches, capsys):
    """A refused step prints an error and returns exit code 1."""
    config.VALIDATION_LABELLED.write_text("already labelled", encoding="utf-8")
    assert validation.main(["sheet"]) == 1
    assert "Error" in capsys.readouterr().err
