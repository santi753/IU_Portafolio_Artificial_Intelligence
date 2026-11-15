"""Real-news validation set: labelling sheet, label loading and freezing.

The model is validated on a small set of real headlines labelled by hand.
The workflow keeps the labels independent of the model:

1. ``sheet``: sample ``config.VALIDATION_PER_COMPANY`` cleaned headlines per
   company from the latest news caches and write them to
   ``config.VALIDATION_SHEET`` with an empty ``label`` column. The sheet has
   no prediction or confidence columns, so the labelling is blind.
2. The labeller fills in the labels (see ``data/validation/LABELLING_GUIDE.md``)
   and saves the result as ``config.VALIDATION_LABELLED``.
3. ``freeze``: validate the labels and store the SHA-256 of the labelled file
   in ``config.VALIDATION_HASH``. :func:`check_frozen` later proves that the
   labels were not changed after the model was evaluated on them.

Usage::

    python -m newsentiment.validation sheet
    python -m newsentiment.validation freeze
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd

from newsentiment import config
from newsentiment.news_fetcher import load_news, slugify
from newsentiment.preprocessing import preprocess

# Companies of the validation set (the three case-study companies)
COMPANIES = ("Tesla", "Airbus", "Siemens")

# Columns of the labelling sheet, in fixed order
SHEET_COLUMNS = ["id", "company", "published", "source", "title", "label"]


def latest_cache(company: str) -> Path:
    """Return the newest news cache ``<NEWS_DIR>/<company-slug>_*.csv`` of a company.

    The file names end with a UTC time stamp (YYYYMMDD-HHMM), so the
    alphabetically last file is the newest.

    Raises:
        FileNotFoundError: If the company has no cache file.
    """
    files = sorted(Path(config.NEWS_DIR).glob(f"{slugify(company)}_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No news cache for '{company}' in {config.NEWS_DIR}. "
            f"Run 'python -m newsentiment {company}' first."
        )
    return files[-1]


def create_labelling_sheet(
    companies: tuple[str, ...] = COMPANIES,
    n: int = config.VALIDATION_PER_COMPANY,
) -> pd.DataFrame:
    """Sample headlines for labelling and write them to ``config.VALIDATION_SHEET``.

    For each company the latest cache is loaded and cleaned with
    :func:`preprocess`, and ``n`` headlines are sampled. All rows are then
    shuffled, so companies are mixed and the labeller sees no grouping.
    Sampling and shuffling use ``config.RANDOM_SEED``: the same caches always
    give the same sheet.

    Args:
        companies: Company names to sample from.
        n: Headlines per company.

    Returns:
        The sheet: columns ``SHEET_COLUMNS``, ids 1..len(companies)*n,
        ``published`` as date only (YYYY-MM-DD, "" if unknown) and an empty
        ``label`` column.

    Raises:
        FileExistsError: If ``config.VALIDATION_LABELLED`` already exists
            (a new sheet must not replace headlines that were already labelled).
        FileNotFoundError: If a company has no news cache.
        ValueError: If a company has fewer than ``n`` headlines after cleaning.
    """
    labelled = Path(config.VALIDATION_LABELLED)
    if labelled.exists():
        raise FileExistsError(
            f"{labelled} already exists; refusing to create a new labelling sheet."
        )

    parts = []
    for company in companies:
        clean, _ = preprocess(load_news(latest_cache(company)), company)
        if len(clean) < n:
            raise ValueError(
                f"Only {len(clean)} headlines for '{company}' after cleaning, {n} needed."
            )
        parts.append(clean.sample(n=n, random_state=config.RANDOM_SEED))

    # Shuffle across companies so the sheet is not grouped by company
    sheet = pd.concat(parts).sample(frac=1, random_state=config.RANDOM_SEED)
    sheet = sheet.reset_index(drop=True)
    sheet.insert(0, "id", range(1, len(sheet) + 1))
    # Date only: the time of day is not needed for labelling
    sheet["published"] = sheet["published"].dt.strftime("%Y-%m-%d").fillna("")
    sheet["label"] = ""
    sheet = sheet[SHEET_COLUMNS]

    path = Path(config.VALIDATION_SHEET)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig writes a byte order mark, so Excel recognises the file as UTF-8
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return sheet


def _read_text(path: Path) -> str:
    """Read a CSV file saved by Excel or pandas as text.

    "CSV UTF-8" (with or without BOM) is tried first; Excel's plain "CSV"
    format is saved in the Windows code page cp1252.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def load_labels(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the labelled sheet.

    The delimiter is detected from the header line: a comma (standard CSV) or
    a semicolon (Excel with e.g. French or German regional settings). Labels
    are stripped and lowercased, so " Positive" counts as "positive".

    Args:
        path: Labelled CSV file. None uses ``config.VALIDATION_LABELLED``.

    Returns:
        DataFrame with the columns ``SHEET_COLUMNS``, sorted by ``id`` (int).

    Raises:
        ValueError: If a column is missing, the ids are not exactly
            1..len(COMPANIES)*config.VALIDATION_PER_COMPANY, or a label is
            missing or not in ``config.LABELS``. The message names the ids.
    """
    path = Path(path if path is not None else config.VALIDATION_LABELLED)
    text = _read_text(path)
    header = text.splitlines()[0] if text else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing_columns = [c for c in SHEET_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name}: missing columns {missing_columns}")
    df = df[SHEET_COLUMNS]
    # Excel sometimes saves empty rows at the end (";;;;;"); they are not data
    df = df[(df.apply(lambda col: col.str.strip()) != "").any(axis=1)].copy()

    # Ids: exactly 1..N, each once
    expected = set(range(1, len(COMPANIES) * config.VALIDATION_PER_COMPANY + 1))
    ids = pd.to_numeric(df["id"].str.strip(), errors="coerce")
    if ids.isna().any():
        bad = df.loc[ids.isna(), "id"].tolist()
        raise ValueError(f"{path.name}: ids that are not numbers: {bad}")
    df["id"] = ids.astype(int)
    problems = []
    unexpected = sorted(set(df["id"]) - expected)
    absent = sorted(expected - set(df["id"]))
    duplicated = sorted(set(df.loc[df["id"].duplicated(), "id"]))
    if absent:
        problems.append(f"missing ids {absent}")
    if unexpected:
        problems.append(f"unexpected ids {unexpected}")
    if duplicated:
        problems.append(f"duplicated ids {duplicated}")
    if problems:
        raise ValueError(f"{path.name}: ids must be 1..{len(expected)}: " + "; ".join(problems))

    # Labels: one of config.LABELS for every row
    df["label"] = df["label"].str.strip().str.lower()
    empty = df.loc[df["label"] == "", "id"].sort_values().tolist()
    invalid = df[(df["label"] != "") & ~df["label"].isin(config.LABELS)].sort_values("id")
    problems = []
    if empty:
        problems.append(f"missing label for ids {empty}")
    if len(invalid):
        pairs = ", ".join(f"{i}: '{lab}'" for i, lab in zip(invalid["id"], invalid["label"]))
        problems.append(f"invalid label for ids {invalid['id'].tolist()} ({pairs})")
    if problems:
        raise ValueError(
            f"{path.name}: " + "; ".join(problems) + f". Allowed labels: {list(config.LABELS)}"
        )

    return df.sort_values("id").reset_index(drop=True)


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen_hash() -> str:
    """Hash stored in ``config.VALIDATION_HASH`` (first field, sha256sum format)."""
    return Path(config.VALIDATION_HASH).read_text(encoding="utf-8").split()[0]


def freeze_labels() -> str:
    """Validate the labelled file and store its SHA-256 in ``config.VALIDATION_HASH``.

    Freezing is done once. If a hash file already exists, the labels count as
    frozen: the call only succeeds if the file is unchanged, so labels cannot
    be edited and silently re-frozen.

    Returns:
        The SHA-256 hex digest of ``config.VALIDATION_LABELLED``.

    Raises:
        ValueError: If the labels are invalid (see :func:`load_labels`) or the
            labels were already frozen with a different hash.
    """
    labelled = Path(config.VALIDATION_LABELLED)
    load_labels(labelled)
    digest = file_sha256(labelled)

    hash_file = Path(config.VALIDATION_HASH)
    if hash_file.exists():
        if _frozen_hash() != digest:
            raise ValueError(
                f"{labelled.name} is already frozen and has changed since; "
                "frozen labels must not be edited."
            )
        return digest
    # Same format as the sha256sum tool: "<hash>  <file name>"
    hash_file.write_text(f"{digest}  {labelled.name}\n", encoding="utf-8", newline="\n")
    return digest


def check_frozen() -> None:
    """Check that the labelled file still has the frozen hash.

    Raises:
        FileNotFoundError: If the labelled file or the hash file is missing.
        ValueError: If the file's SHA-256 differs from the frozen one.
    """
    digest = file_sha256(config.VALIDATION_LABELLED)
    frozen = _frozen_hash()
    if digest != frozen:
        raise ValueError(
            f"{Path(config.VALIDATION_LABELLED).name} has changed since it was frozen "
            f"(SHA-256 {digest}, frozen {frozen})."
        )


def _print_sheet(sheet: pd.DataFrame, companies: tuple[str, ...]) -> None:
    """Print where the sheet came from and how many rows each company has."""
    print(f"Labelling sheet written: {config.VALIDATION_SHEET}")
    print("Sampled from: " + ", ".join(latest_cache(c).name for c in companies))
    print("Rows per company:")
    counts = sheet["company"].value_counts()
    for company in companies:
        print(f"  {company:<10}{counts.get(company, 0):>4}")
    print(f"  {'Total':<10}{len(sheet):>4}")


def _print_frozen(labels: pd.DataFrame, digest: str) -> None:
    """Print the hash and the label counts per class and per company."""
    print(f"Labels frozen: {config.VALIDATION_LABELLED}")
    print(f"SHA-256: {digest}")
    print()
    print("Labels per class:")
    counts = labels["label"].value_counts()
    for label in config.LABELS:
        print(f"  {label:<10}{counts.get(label, 0):>4}")
    print()
    print("Labels per company:")
    table = pd.crosstab(labels["company"], labels["label"], margins=True, margins_name="Total")
    table = table.reindex(columns=[*config.LABELS, "Total"], fill_value=0)
    print(table.to_string())


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m newsentiment.validation {sheet,freeze}``.

    Returns:
        Exit code: 0 on success, 1 if the step was refused or the input is invalid.
    """
    parser = argparse.ArgumentParser(
        prog="python -m newsentiment.validation",
        description="Real-news validation set: create the labelling sheet or freeze the labels.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sheet", help=f"write the label-free sheet to {config.VALIDATION_SHEET.name}")
    commands.add_parser("freeze", help=f"validate {config.VALIDATION_LABELLED.name} and store its hash")
    args = parser.parse_args(argv)

    try:
        if args.command == "sheet":
            sheet = create_labelling_sheet()
            _print_sheet(sheet, COMPANIES)
        else:
            digest = freeze_labels()
            _print_frozen(load_labels(), digest)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
