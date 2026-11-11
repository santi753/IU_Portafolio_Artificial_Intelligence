"""Tests for the news fetcher (newsentiment.news_fetcher).

All tests run offline: requests.get is replaced by a fake, so no network is used.
"""

import pandas as pd
import pytest
import requests

from newsentiment import config, news_fetcher
from newsentiment.news_fetcher import NewsFetchError, fetch_news, parse_feed, save_news

# Two items in Google News style: the first has a <source>, the second has none
FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>"Tesla when:7d" - Google News</title>
    <item>
      <title>Tesla shares rise after delivery report - Reuters</title>
      <link>https://news.google.com/rss/articles/abc</link>
      <pubDate>Mon, 09 Nov 2026 14:30:00 GMT</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
    </item>
    <item>
      <title>Tesla opens new factory</title>
      <link>https://news.google.com/rss/articles/def</link>
      <pubDate>Sun, 08 Nov 2026 10:00:00 +0100</pubDate>
    </item>
  </channel>
</rss>
"""

EMPTY_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>empty</title></channel></rss>
"""


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_parse_feed_columns_and_values():
    """Both items are parsed with company, title, source, link and UTC dates."""
    df = parse_feed(FIXTURE, "Tesla")
    assert list(df.columns) == ["company", "title", "source", "published", "link"]
    assert len(df) == 2
    assert (df["company"] == "Tesla").all()
    assert df["title"].tolist() == ["Tesla shares rise after delivery report - Reuters",
                                    "Tesla opens new factory"]
    assert df["source"].tolist() == ["Reuters", ""]
    assert df["link"].iloc[0] == "https://news.google.com/rss/articles/abc"


def test_parse_feed_dates_are_utc():
    """Dates are timezone-aware UTC; the +0100 date is shifted to UTC."""
    df = parse_feed(FIXTURE, "Tesla")
    assert str(df["published"].dt.tz) == "UTC"
    assert df["published"].iloc[0] == pd.Timestamp("2026-11-09 14:30", tz="UTC")
    assert df["published"].iloc[1] == pd.Timestamp("2026-11-08 09:00", tz="UTC")


def test_parse_feed_missing_date_is_nat():
    """An item without pubDate gets NaT."""
    content = FIXTURE.replace(b"<pubDate>Sun, 08 Nov 2026 10:00:00 +0100</pubDate>", b"")
    df = parse_feed(content, "Tesla")
    assert pd.isna(df["published"].iloc[1])


def test_parse_feed_empty():
    """An empty feed gives an empty DataFrame with the expected columns."""
    df = parse_feed(EMPTY_FEED, "Tesla")
    assert df.empty
    assert list(df.columns) == ["company", "title", "source", "published", "link"]


def test_fetch_news_uses_query_and_settings(monkeypatch):
    """fetch_news sends the right query, params, timeout and User-Agent."""
    calls = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.update(url=url, params=params, headers=headers, timeout=timeout)
        return FakeResponse(FIXTURE)

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    df = fetch_news("Tesla", days=3)

    assert calls["url"] == config.NEWS_URL
    assert calls["params"]["q"] == "Tesla when:3d"
    assert calls["params"]["hl"] == "en-US"
    assert calls["params"]["ceid"] == "US:en"
    assert calls["headers"]["User-Agent"] == config.USER_AGENT
    assert calls["timeout"] == config.REQUEST_TIMEOUT
    assert len(df) == 2
    assert "fetched_at" in df.columns
    assert str(df["fetched_at"].dt.tz) == "UTC"


def test_fetch_news_timeout_raises(monkeypatch):
    """A timeout is turned into NewsFetchError."""
    def fake_get(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    with pytest.raises(NewsFetchError, match="Tesla"):
        fetch_news("Tesla")


def test_fetch_news_http_error_raises(monkeypatch):
    """An HTTP error status is turned into NewsFetchError."""
    monkeypatch.setattr(news_fetcher.requests, "get",
                        lambda *a, **k: FakeResponse(b"", status_code=503))
    with pytest.raises(NewsFetchError):
        fetch_news("Tesla")


@pytest.mark.parametrize("name", ["", "   "])
def test_fetch_news_empty_company_raises(monkeypatch, name):
    """An empty company name is rejected before any request is made."""
    def fake_get(*args, **kwargs):
        raise AssertionError("no request expected")

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    with pytest.raises(ValueError):
        fetch_news(name)


def test_save_news_writes_csv(monkeypatch, tmp_path):
    """save_news writes a UTF-8 CSV named <slug>_<YYYYMMDD-HHMM>.csv into NEWS_DIR."""
    monkeypatch.setattr(config, "NEWS_DIR", tmp_path)
    df = parse_feed(FIXTURE, "Rolls Royce")
    path = save_news(df, "Rolls Royce")

    assert path.parent == tmp_path
    assert path.exists()
    assert path.name.startswith("rolls-royce_")
    assert len(path.stem.split("_")[1]) == len("YYYYMMDD-HHMM")
    loaded = pd.read_csv(path, encoding="utf-8")
    assert loaded["title"].tolist() == df["title"].tolist()


def test_load_news_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NEWS_DIR", tmp_path)
    df = parse_feed(FIXTURE, "Tesla")
    df["fetched_at"] = pd.Timestamp("2026-11-10 12:00", tz="UTC")
    loaded = news_fetcher.load_news(save_news(df, "Tesla"))
    assert str(loaded["published"].dt.tz) == "UTC"
    assert str(loaded["fetched_at"].dt.tz) == "UTC"
    assert loaded["source"].tolist() == ["Reuters", ""]
    assert loaded["title"].tolist() == df["title"].tolist()
