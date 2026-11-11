"""Fetching of recent company news headlines from the Google News RSS search feed.

The feed is free and needs no API key. The English edition (``config.NEWS_PARAMS``)
is used because the sentiment model was trained on English sentences.

This module only downloads and parses the feed into a DataFrame. Headlines are
returned exactly as the feed delivers them; cleaning happens in the
preprocessing step.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests

from newsentiment import config

# Columns returned by parse_feed, in fixed order
NEWS_COLUMNS = ["company", "title", "source", "published", "link"]


class NewsFetchError(Exception):
    """Raised when the news feed cannot be downloaded."""


def parse_feed(content: bytes, company: str) -> pd.DataFrame:
    """Parse the raw RSS bytes into one row per news item.

    Args:
        content: Body of the RSS response.
        company: Company name the feed was requested for (stored in every row).

    Returns:
        DataFrame with columns ``company``, ``title``, ``source``, ``published``
        and ``link``. ``published`` is a UTC pandas datetime (NaT if the item has
        no date); ``source`` is the publisher name or "" if missing. An empty
        feed gives an empty DataFrame with the same columns.
    """
    feed = feedparser.parse(content)
    rows = []
    for entry in feed.entries:
        # feedparser converts the RSS date to a UTC time tuple
        parsed = entry.get("published_parsed")
        published = datetime(*parsed[:6], tzinfo=timezone.utc) if parsed else None
        source = entry.get("source")
        rows.append({
            "company": company,
            "title": entry.get("title", ""),
            "source": source.get("title", "") if source else "",
            "published": published,
            "link": entry.get("link", ""),
        })
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    # utc=True gives a datetime64[ns, UTC] column even when it is empty or all NaT
    df["published"] = pd.to_datetime(df["published"], utc=True)
    return df


def fetch_news(company: str, days: int = config.NEWS_DAYS) -> pd.DataFrame:
    """Download the news items of the last ``days`` days for one company.

    Args:
        company: Company name used as search query, e.g. "Tesla".
        days: Look-back window in days. Defaults to ``config.NEWS_DAYS``.

    Returns:
        The parse_feed DataFrame plus a ``fetched_at`` column (UTC time of the request).

    Raises:
        ValueError: If ``company`` is empty or only whitespace.
        NewsFetchError: If the request fails (network error, timeout) or the
            server answers with an HTTP error status.
    """
    company = company.strip() if company else ""
    if not company:
        raise ValueError("Company name must not be empty.")

    params = {"q": f"{company} when:{days}d", **config.NEWS_PARAMS}
    headers = {"User-Agent": config.USER_AGENT}
    fetched_at = pd.Timestamp.now(tz="UTC")
    try:
        response = requests.get(
            config.NEWS_URL, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise NewsFetchError(
            f"Could not fetch news for '{company}' from {config.NEWS_URL}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    df = parse_feed(response.content, company)
    df["fetched_at"] = fetched_at
    return df


def slugify(text: str) -> str:
    """Turn a company name into a lowercase file-name part, e.g. "Rolls Royce" -> "rolls-royce"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "company"


def save_news(df: pd.DataFrame, company: str) -> Path:
    """Save fetched news as a UTF-8 CSV in ``config.NEWS_DIR``.

    The file is named ``<company-slug>_<YYYYMMDD-HHMM>.csv`` (current UTC time).

    Args:
        df: DataFrame returned by fetch_news.
        company: Company name, used for the file name.

    Returns:
        Path of the written file.
    """
    news_dir = Path(config.NEWS_DIR)
    news_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    path = news_dir / f"{slugify(company)}_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_news(path: str | Path) -> pd.DataFrame:
    """Read a news cache CSV written by save_news.

    Args:
        path: Path of the CSV file.

    Returns:
        DataFrame with the saved columns. Text columns keep "" for missing values
        (not NaN); ``published`` and ``fetched_at`` are UTC pandas datetimes.
    """
    df = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    for column in ("published", "fetched_at"):
        if column in df.columns:
            # "" (missing date) becomes NaT
            df[column] = pd.to_datetime(df[column].replace("", None), utc=True, format="ISO8601")
    return df
