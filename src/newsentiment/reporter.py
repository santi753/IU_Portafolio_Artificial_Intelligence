"""Reporter component: summary statistics, a console report and CSV export.

Works on the output of :func:`newsentiment.classifier.classify`, i.e. a news
frame with the columns ``sentiment``, ``confidence`` and ``low_confidence``.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from newsentiment import config
from newsentiment.news_fetcher import slugify

# Column order of the per-headline results CSV
HEADLINE_COLUMNS = (
    "company", "published", "source", "title",
    "sentiment", "confidence", "low_confidence", "link",
)
# Maximum title length shown in the console report
TITLE_WIDTH = 90


def summarise(df: pd.DataFrame) -> dict:
    """Aggregate the classified headlines of one company.

    Args:
        df: Classified news frame.

    Returns:
        Dict with ``n``, ``count_<label>`` and ``share_<label>`` for every label
        in ``config.LABELS`` order, ``net_sentiment`` ((positive - negative) / n
        * 100, rounded to 1 decimal, range -100..+100), ``mean_confidence``
        (rounded to 3 decimals) and ``n_low_confidence``. For empty input ``n``
        is 0, counts are 0 and the shares, net sentiment and mean confidence are None.
    """
    n = len(df)
    counts = df["sentiment"].value_counts() if n else pd.Series(dtype=int)
    summary: dict = {"n": n}
    for label in config.LABELS:
        count = int(counts.get(label, 0))
        summary[f"count_{label}"] = count
        summary[f"share_{label}"] = round(count / n, 3) if n else None

    if n:
        net = (summary["count_positive"] - summary["count_negative"]) / n * 100
        summary["net_sentiment"] = round(net, 1)
        summary["mean_confidence"] = round(float(df["confidence"].mean()), 3)
        summary["n_low_confidence"] = int(df["low_confidence"].sum())
    else:
        summary["net_sentiment"] = None
        summary["mean_confidence"] = None
        summary["n_low_confidence"] = 0
    return summary


def _fmt_date(value) -> str:
    """Format a timestamp as YYYY-MM-DD, or "?" if it is missing."""
    return "?" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _cut(text: str, width: int = TITLE_WIDTH) -> str:
    """Shorten ``text`` to at most ``width`` characters, ending in "..." if cut."""
    return text if len(text) <= width else text[: width - 3] + "..."


def format_report(df: pd.DataFrame, summary: dict, company: str, top_n: int = 10) -> str:
    """Build a plain-text console report for one company.

    Args:
        df: Classified news frame.
        summary: Output of :func:`summarise` for ``df``.
        company: Company name shown in the header.
        top_n: Number of newest headlines to list.

    Returns:
        The report as one multi-line string.
    """
    lines = [f"Sentiment report: {company}", "=" * 60]
    n = summary["n"]
    if n == 0:
        lines.append("No headlines to report.")
        return "\n".join(lines)

    # Header: number of headlines and the covered date range
    dates = df["published"].dropna() if "published" in df.columns else pd.Series(dtype=object)
    if len(dates):
        date_range = f"{_fmt_date(dates.min())} to {_fmt_date(dates.max())}"
    else:
        date_range = "unknown"
    lines.append(f"Headlines: {n}   Period: {date_range}")
    lines.append("")

    # Class shares with counts, in the fixed label order
    for label in config.LABELS:
        share = summary[f"share_{label}"]
        lines.append(f"  {label:<9} {share:6.1%}  ({summary[f'count_{label}']})")
    lines.append("")
    lines.append(f"Net sentiment: {summary['net_sentiment']:+.1f}  (scale -100 to +100)")
    lines.append(
        f"Low-confidence headlines (confidence < {config.LOW_CONFIDENCE_THRESHOLD}): "
        f"{summary['n_low_confidence']}   Mean confidence: {summary['mean_confidence']:.3f}"
    )
    lines.append("")

    # Newest headlines first; rows without a date go last
    shown = df.sort_values("published", ascending=False, na_position="last").head(top_n)
    lines.append(f"Latest {len(shown)} headlines (* = low confidence):")
    for row in shown.itertuples(index=False):
        mark = "*" if row.low_confidence else " "
        lines.append(
            f"  {_fmt_date(row.published)}  {row.sentiment:<8} {row.confidence:.3f}{mark} "
            f"{_cut(str(row.title))}"
        )
    return "\n".join(lines)


def save_results(df: pd.DataFrame, summary: dict, company: str) -> tuple[Path, Path]:
    """Write the per-headline results and the one-row summary as CSV files.

    Both files go to ``config.RESULTS_DIR`` and are named
    ``<company-slug>_<YYYYMMDD-HHMM UTC>_headlines.csv`` and ``..._summary.csv``.

    Args:
        df: Classified news frame.
        summary: Output of :func:`summarise` for ``df``.
        company: Company name, used for the file names and the summary row.

    Returns:
        Paths of the headlines file and the summary file.
    """
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base = f"{slugify(company)}_{stamp}"
    headlines_path = results_dir / f"{base}_headlines.csv"
    summary_path = results_dir / f"{base}_summary.csv"

    # Keep only the documented columns (any that are missing are added empty)
    df.reindex(columns=list(HEADLINE_COLUMNS)).to_csv(headlines_path, index=False, encoding="utf-8")
    row = {"company": company, "created_utc": stamp, **summary}
    pd.DataFrame([row]).to_csv(summary_path, index=False, encoding="utf-8")
    return headlines_path, summary_path
