"""Cleaning of fetched news headlines before sentiment classification.

The Google News feed returns headlines with the publisher appended, and mixes
real news with items that say nothing about how the press sees a company
(own press releases, stock quote pages, off-topic hits, repeated stories).
``preprocess`` removes these with a few simple, documented rules and reports
how many rows each rule removed, so the filtering is transparent in the report.
"""

import re

import pandas as pd

from newsentiment import config

# Columns of the cleaned output; "title" is the headline the model classifies
OUTPUT_COLUMNS = ["company", "title", "source", "published", "link"]

# Keys of the report, in the order the rules are applied
REPORT_KEYS = (
    "rows_in",
    "empty_removed",
    "press_releases_removed",
    "quote_pages_removed",
    "off_topic_removed",
    "duplicates_removed",
    "rows_out",
)


def strip_source_suffix(title: str, source: str) -> str:
    """Remove a trailing " - <source>" from ``title`` if it matches ``source``.

    Only the row's own publisher is removed, so a headline that legitimately
    ends with " - something" is left alone.
    """
    title = title.strip()
    source = source.strip()
    suffix = f" - {source}"
    if source and title.endswith(suffix):
        title = title[: -len(suffix)]
    return title.strip()


def normalize_title(title: str) -> str:
    """Lowercase and collapse every run of non-alphanumeric characters into one space."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def preprocess(df: pd.DataFrame, company: str) -> tuple[pd.DataFrame, dict]:
    """Clean the news items of one company.

    Rules, applied in this order:
        a) remove " - <source>" from the end of the title; drop empty titles
        b) drop the company's own press releases (source equals the company name)
        c) drop stock quote pages (title contains a ``config.QUOTE_PAGE_PATTERNS`` entry)
        d) drop off-topic items (title does not mention the company as a whole word)
        e) drop duplicate titles (after normalisation), keeping the first

    Args:
        df: News items as returned by fetch_news / load_news. Not modified.
        company: Company name the news was fetched for, e.g. "Tesla".

    Returns:
        ``(clean, report)``: ``clean`` has the columns ``OUTPUT_COLUMNS`` and a
        fresh 0..n-1 index; ``report`` maps each key of ``REPORT_KEYS`` to a count.
    """
    company = company.strip()
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    out = out[OUTPUT_COLUMNS]
    out["title"] = out["title"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    report = {"rows_in": len(out)}

    # a) The appended publisher is not part of the headline and would leak the
    #    source name into the model input; an empty title has nothing to classify.
    # dtype="object" keeps the .str accessor working when the frame is empty
    out["title"] = pd.Series(
        [strip_source_suffix(t, s) for t, s in zip(out["title"], out["source"])],
        index=out.index,
        dtype="object",
    )
    keep = out["title"] != ""
    report["empty_removed"] = int((~keep).sum())
    out = out[keep]

    # b) A company's own press releases are self-promotion, not press coverage.
    #    Only an exact name match counts, so sites like "Teslarati" are kept.
    keep = out["source"].str.strip().str.lower() != company.lower()
    report["press_releases_removed"] = int((~keep).sum())
    out = out[keep]

    # c) Stock quote pages list prices, not news, and would inflate "neutral".
    lowered = out["title"].str.lower()
    is_quote = pd.Series(False, index=out.index)
    for pattern in config.QUOTE_PAGE_PATTERNS:
        is_quote |= lowered.str.contains(pattern.lower(), regex=False)
    report["quote_pages_removed"] = int(is_quote.sum())
    out = out[~is_quote]

    # d) The search also matches the article text, so some headlines are about
    #    something else; their sentiment says nothing about the company.
    #    \b makes it a whole word ("Tesla's" matches, "Teslarati" does not).
    mention = re.compile(rf"\b{re.escape(company)}\b", flags=re.IGNORECASE)
    keep = out["title"].map(lambda t: bool(mention.search(t))).astype(bool)
    report["off_topic_removed"] = int((~keep).sum())
    out = out[keep]

    # e) The same story syndicated by several outlets would be counted twice;
    #    normalising ignores differences in case and punctuation.
    duplicated = out["title"].map(normalize_title).duplicated(keep="first")
    report["duplicates_removed"] = int(duplicated.sum())
    out = out[~duplicated]

    out = out.reset_index(drop=True)
    report["rows_out"] = len(out)
    return out, report
