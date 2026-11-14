"""Tests for the command-line application (newsentiment.app).

All tests run offline: app.fetch_news is replaced by a fake that returns a small
fixture frame, and the cache and result directories point to tmp_path. The real
saved model is used for classification.
"""

import pandas as pd
import pytest

from newsentiment import app, config
from newsentiment.news_fetcher import NEWS_COLUMNS, NewsFetchError


def make_news(company: str, titles: list[str]) -> pd.DataFrame:
    """A frame shaped like the output of fetch_news, one row per title."""
    n = len(titles)
    df = pd.DataFrame(
        {
            "company": company,
            "title": [f"{t} - Example News" for t in titles],
            "source": "Example News",
            "published": pd.date_range("2026-11-01", periods=n, freq="D", tz="UTC"),
            "link": [f"https://example.com/{i}" for i in range(n)],
        },
        columns=NEWS_COLUMNS,
    )
    df["fetched_at"] = pd.Timestamp("2026-11-10", tz="UTC")
    return df


# 5 items: 3 on-topic, 1 off-topic, 1 duplicate of the first (different case)
TESLA_TITLES = [
    "Tesla profit rose sharply compared with the previous year",
    "Tesla will cut 2,000 jobs after quarterly losses widened",
    "Tesla opens new factory in Berlin",
    "Electric car sales slow across Europe",
    "TESLA PROFIT ROSE SHARPLY COMPARED WITH THE PREVIOUS YEAR",
]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Redirect the news cache and result directories to tmp_path."""
    news_dir, results_dir = tmp_path / "news", tmp_path / "results"
    monkeypatch.setattr(config, "NEWS_DIR", news_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    return news_dir, results_dir


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace app.fetch_news; returns the list of (company, days) calls made."""
    calls = []

    def fake(company, days=config.NEWS_DAYS):
        calls.append((company, days))
        return make_news(company, TESLA_TITLES)

    monkeypatch.setattr(app, "fetch_news", fake)
    return calls


def csv_files(directory):
    """Names of the CSV files in ``directory`` (empty list if it does not exist)."""
    return sorted(p.name for p in directory.glob("*.csv")) if directory.exists() else []


def test_no_save_prints_report(dirs, fake_fetch, capsys):
    assert app.main(["Tesla", "--no-save", "--days", "3"]) == 0
    out = capsys.readouterr().out
    assert "5 fetched -> 3 kept: 0 press releases, 0 quote pages, 1 off-topic, 1 duplicates" in out
    assert "Sentiment report: Tesla" in out
    assert "Headlines: 3" in out
    assert "Saved files" not in out
    assert fake_fetch == [("Tesla", 3)]
    # Nothing written with --no-save
    assert csv_files(dirs[0]) == [] and csv_files(dirs[1]) == []


def test_save_writes_cache_and_results(dirs, fake_fetch, capsys):
    news_dir, results_dir = dirs
    assert app.main(["Tesla"]) == 0
    out = capsys.readouterr().out
    cache = csv_files(news_dir)
    results = csv_files(results_dir)
    assert len(cache) == 1 and cache[0].startswith("tesla_")
    assert len(results) == 2
    assert any(name.endswith("_headlines.csv") for name in results)
    assert any(name.endswith("_summary.csv") for name in results)
    # All three paths are listed in the output
    assert "Saved files:" in out
    for name in cache + results:
        assert name in out
    # Default look-back window is used
    assert fake_fetch == [("Tesla", config.NEWS_DAYS)]


def test_run_returns_all_parts(dirs, fake_fetch):
    output = app.run("Tesla", save=False)
    assert set(output) == {"results", "summary", "preprocessing", "report_text", "files"}
    assert len(output["results"]) == output["summary"]["n"] == 3
    assert output["results"]["sentiment"].isin(config.LABELS).all()
    assert output["preprocessing"]["rows_out"] == 3
    assert output["files"] == []


def test_fetch_error_returns_1(dirs, monkeypatch, capsys):
    def failing(company, days=config.NEWS_DAYS):
        raise NewsFetchError("Could not fetch news for 'Tesla': ConnectionError")

    monkeypatch.setattr(app, "fetch_news", failing)
    assert app.main(["Tesla"]) == 1
    captured = capsys.readouterr()
    assert "Could not fetch news for 'Tesla'" in captured.err
    assert "Sentiment report" not in captured.out
    assert csv_files(dirs[0]) == [] and csv_files(dirs[1]) == []


def test_all_off_topic_reports_no_headlines(dirs, fake_fetch, capsys):
    # None of the Tesla fixture titles mentions Airbus, so preprocessing keeps nothing
    news_dir, results_dir = dirs
    assert app.main(["Airbus"]) == 0
    out = capsys.readouterr().out
    assert "No recent headlines found for Airbus" in out
    assert "5 fetched -> 0 kept" in out
    # The raw cache is kept, but no (empty) result files are written
    assert len(csv_files(news_dir)) == 1
    assert csv_files(results_dir) == []

    output = app.run("Airbus", save=False)
    assert output["summary"]["n"] == 0 and output["results"].empty


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "--no-save" in out and "--days" in out


def test_empty_company_is_rejected(fake_fetch, capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["   "])
    assert exc.value.code == 2
    assert "company name must not be empty" in capsys.readouterr().err
    assert fake_fetch == []
