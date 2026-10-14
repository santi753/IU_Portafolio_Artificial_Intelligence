"""Phase 1 feasibility check of the Google News RSS search feed.

Fetches the feed for a few company keywords, describes the returned items and
writes the results into feasibility_report.md. Articles are not stored except
as a few sample titles in the report.
"""
import html
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests

REPORT = Path(__file__).resolve().parent / "feasibility_report.md"
URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
HEADERS = {"User-Agent": "IU-student-feasibility-check/0.1 (news sentiment course project)"}
KEYWORDS = ["Tesla", "Airbus", "Siemens"]


def write_section(key: str, body: str) -> None:
    """Insert or replace the section delimited by markers for `key` in the report."""
    start, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
    old = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Phase 1 Feasibility Report\n"
    block = f"{start}\n{body.strip()}\n{end}"
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    new = pattern.sub(lambda _: block, old) if pattern.search(old) else old.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(new, encoding="utf-8")


def fetch(query: str) -> tuple[int | None, list, str | None]:
    """Fetch and parse one feed; return (HTTP status, entries, error message)."""
    try:
        resp = requests.get(URL.format(q=quote_plus(query)), headers=HEADERS, timeout=15)
    except requests.RequestException as exc:
        return None, [], f"{type(exc).__name__}: {exc}"
    return resp.status_code, feedparser.parse(resp.content).entries, None


def pub_date(entry) -> datetime | None:
    """Return the entry's published date as an aware UTC datetime, if present."""
    t = entry.get("published_parsed")
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None


def strip_html(text: str) -> str:
    """Remove HTML tags and entities and collapse whitespace."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def describe(keyword: str, query: str) -> tuple[str, list[str]]:
    """Analyse one query; return a Markdown block and observation lines."""
    status, entries, err = fetch(query)
    head = f"#### Query `{query}`\n\n- HTTP status: {status}\n"
    if err:
        return head + f"- Network error: `{err}`\n", [f"`{query}`: request failed ({err})."]
    n = len(entries)
    fields = Counter(k for e in entries for k in e.keys())
    dates = sorted(d for d in map(pub_date, entries) if d)
    kw = sum(keyword.lower() in e.get("title", "").lower() for e in entries)
    src = [e.get("source", {}).get("title", "") for e in entries]
    suffix = sum(bool(s) and e.get("title", "").endswith(f" - {s}") for e, s in zip(entries, src))
    # Summary text left over after removing the headline and the source name.
    extra = sum(bool(strip_html(e.get("summary", "")).replace(e.get("title", "").rsplit(" - ", 1)[0], "")
                     .replace(s, "").strip()) for e, s in zip(entries, src))
    rng = f"{dates[0]:%Y-%m-%d %H:%M} to {dates[-1]:%Y-%m-%d %H:%M} UTC" if dates else "n/a"

    def pct(x: int) -> str:
        return f"{x} ({x / n:.0%})" if n else "0"
    md = head + (f"- Items: **{n}**\n- Published range: {rng}\n"
                 f"- Title contains '{keyword}' (case-insensitive): {pct(kw)}\n"
                 f"- Title ends with ' - <source name>': {pct(suffix)}\n"
                 f"- Summary contains text beyond title/source (HTML stripped): {pct(extra)}\n"
                 f"- Fields (items having it): " + ", ".join(f"`{k}` {v}" for k, v in fields.most_common()) + "\n\n"
                 "| # | title | source | published |\n|---|---|---|---|\n")
    for i, (e, s) in enumerate(zip(entries[:5], src[:5]), 1):
        md += f"| {i} | {e.get('title', '').replace('|', '/')} | {s} | {e.get('published', '')} |\n"
    if entries:
        md += f"\nSummary of item 1, HTML stripped: \"{strip_html(entries[0].get('summary', ''))[:200]}\"\n"
    obs = [f"`{query}`: {n} items, published {rng}; keyword in title {pct(kw)}; "
           f"source suffix {pct(suffix)}; summary with extra text {pct(extra)}."]
    return md, obs


def main() -> None:
    """Run all feed checks and write the report section."""
    out = [f"## 2. News feed check (Google News RSS)\n\nRun: {datetime.now():%Y-%m-%d %H:%M:%S} | "
           f"Python {platform.python_version()} | requests {requests.__version__} | "
           f"feedparser {feedparser.__version__}\n\nURL template: `{URL}`\n"]
    obs: list[str] = []
    for kw in KEYWORDS:
        md, o = describe(kw, f"{kw} when:7d")
        out.append(md)
        obs += o
    # Baseline without the recency operator: count and oldest date only.
    status, entries, err = fetch("Tesla")
    dates = sorted(d for d in map(pub_date, entries) if d)
    oldest = f"{dates[0]:%Y-%m-%d %H:%M} UTC" if dates else "n/a"
    out.append(f"#### Query `Tesla` (without when:7d)\n\n- HTTP status: {status}\n"
               f"- Items: **{len(entries)}**\n- Oldest published: {oldest}\n"
               + (f"- Network error: `{err}`\n" if err else ""))
    obs.append(f"`Tesla` without when:7d: {len(entries)} items, oldest {oldest}.")
    out.append("### Observations\n\n" + "\n".join(f"- {o}" for o in obs))
    write_section("news", "\n".join(out))
    print(f"News feed section written to {REPORT}")


if __name__ == "__main__":
    main()
