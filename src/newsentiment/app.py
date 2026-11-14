"""Command-line application: runs the whole pipeline for one company.

Usage::

    python -m newsentiment <company> [--days N] [--top N] [--no-save]

The four components are chained in a fixed order:
fetch_news -> save_news (raw cache) -> preprocess -> classify -> summarise,
and the result is printed as a plain-text report and saved as CSV files.
"""

import argparse
import sys

from sklearn.pipeline import Pipeline

from newsentiment import config
from newsentiment.classifier import classify
from newsentiment.news_fetcher import NewsFetchError, fetch_news, save_news
from newsentiment.preprocessing import preprocess
from newsentiment.reporter import format_report, save_results, summarise


def run(
    company: str,
    days: int = config.NEWS_DAYS,
    top_n: int = 10,
    save: bool = True,
    model: Pipeline | None = None,
) -> dict:
    """Fetch, clean, classify and summarise the recent headlines of one company.

    Args:
        company: Company name used as search query, e.g. "Tesla".
        days: Look-back window in days.
        top_n: Number of newest headlines listed in the report text.
        save: If True, write the raw news cache and the two result CSV files.
        model: Fitted pipeline passed to :func:`classify`; None loads the saved model.

    Returns:
        Dict with ``results`` (classified headlines), ``summary`` (output of
        :func:`summarise`), ``preprocessing`` (the preprocessing report),
        ``report_text`` (console report) and ``files`` (paths written, in order
        raw cache, headlines, summary). If no headline is left after
        preprocessing, ``results`` is empty, ``summary["n"]`` is 0 and no
        result files are written (the raw cache still is, if ``save``).

    Raises:
        NewsFetchError: If the news feed cannot be downloaded.
    """
    files = []
    raw = fetch_news(company, days=days)
    if save:
        # Raw cache first, so the fetched headlines are kept even if a later step fails
        files.append(save_news(raw, company))

    clean, prep_report = preprocess(raw, company)
    # classify() handles an empty frame without loading the model
    results = classify(clean, model=model)
    summary = summarise(results)
    report_text = format_report(results, summary, company, top_n=top_n)

    # An empty result has nothing worth exporting
    if save and summary["n"] > 0:
        files.extend(save_results(results, summary, company))

    return {
        "results": results,
        "summary": summary,
        "preprocessing": prep_report,
        "report_text": report_text,
        "files": files,
    }


def format_preprocessing(report: dict) -> str:
    """One-line summary of the preprocessing report.

    Example: "100 fetched -> 95 kept: 0 press releases, 0 quote pages, 3 off-topic, 2 duplicates".
    Empty titles are rare and only mentioned when there are any.
    """
    line = (
        f"{report['rows_in']} fetched -> {report['rows_out']} kept: "
        f"{report['press_releases_removed']} press releases, "
        f"{report['quote_pages_removed']} quote pages, "
        f"{report['off_topic_removed']} off-topic, "
        f"{report['duplicates_removed']} duplicates"
    )
    if report["empty_removed"]:
        line += f", {report['empty_removed']} empty"
    return line


def _utf8_console() -> None:
    """Make stdout and stderr write UTF-8 and replace unencodable characters.

    On Windows, output that is piped or redirected uses the legacy code page,
    and a headline with e.g. a curly quote or an emoji would raise
    UnicodeEncodeError. Streams without ``reconfigure`` are left as they are.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m newsentiment",
        description=(
            "Fetch recent Google News headlines for a company and classify each "
            "one as negative, neutral or positive."
        ),
    )
    parser.add_argument("company", help='company name used as search query, e.g. "Tesla"')
    parser.add_argument(
        "--days", type=int, default=config.NEWS_DAYS,
        help=f"look-back window in days (default: {config.NEWS_DAYS})",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="number of newest headlines listed in the report (default: 10)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="do not write the raw news cache and the result CSV files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m newsentiment``.

    Args:
        argv: Command-line arguments without the program name; None uses sys.argv.

    Returns:
        Exit code: 0 on success (also when no headlines were found), 1 if the
        news could not be fetched. Invalid arguments exit with code 2 (argparse).
    """
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    company = args.company.strip()
    if not company:
        parser.error("company name must not be empty")
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.top < 0:
        parser.error("--top must not be negative")

    try:
        output = run(company, days=args.days, top_n=args.top, save=not args.no_save)
    except NewsFetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Check the internet connection and try again.", file=sys.stderr)
        return 1

    print(format_preprocessing(output["preprocessing"]))
    print()
    if output["summary"]["n"] == 0:
        print(f"No recent headlines found for {company}")
    else:
        print(output["report_text"])

    if output["files"]:
        print()
        print("Saved files:")
        for path in output["files"]:
            print(f"  {path}")
    return 0
