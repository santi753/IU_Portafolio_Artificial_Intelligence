"""Loading and cleaning of the labelled training data.

Source: "Sentiment Analysis for Financial News" on Kaggle
(https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news),
which redistributes the Financial PhraseBank of Malo et al. (2014):

    Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
    Good debt or bad debt: Detecting semantic orientations in economic texts.
    Journal of the Association for Information Science and Technology, 65(4), 782-796.

The file ``all-data.csv`` is latin-1 encoded, has no header row and two
columns: the sentiment label (negative / neutral / positive) and the sentence.

This module only loads and cleans the data. Splitting, exploration and
modelling happen elsewhere.
"""

from pathlib import Path

import pandas as pd

from newsentiment import config

KAGGLE_URL = "https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news"


def load_raw_training_data(path: Path = config.TRAINING_FILE) -> pd.DataFrame:
    """Read the raw training CSV into a DataFrame with columns ``label`` and ``text``.

    Args:
        path: Location of ``all-data.csv``. Defaults to ``config.TRAINING_FILE``.

    Returns:
        The rows exactly as stored in the file (no cleaning applied).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains a label outside ``config.LABELS``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {path}. Download 'all-data.csv' from "
            f"{KAGGLE_URL} and place it in {config.DATA_RAW_DIR}."
        )

    # The file has no header and is not UTF-8, so both must be stated explicitly.
    df = pd.read_csv(path, header=None, names=["label", "text"], encoding="latin-1")

    # Fail early on unexpected labels instead of training on a wrong class set.
    invalid = sorted(set(df["label"].dropna().unique()) - set(config.LABELS))
    if invalid or df["label"].isna().any():
        found = invalid + (["<missing>"] if df["label"].isna().any() else [])
        raise ValueError(
            f"Unexpected labels in {path}: {found}. Allowed labels: {list(config.LABELS)}."
        )
    return df


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw training data and report what was removed.

    Steps, in order:

    1. Strip leading/trailing whitespace from each text, so that texts that
       differ only in surrounding spaces are treated as the same sentence.
    2. Drop rows whose text is empty after stripping; they carry no signal.
    3. Drop *all* rows of any text that appears with more than one label.
       The annotations disagree, so there is no reliable ground truth and
       keeping either version would teach the model a guess.
    4. Drop exact duplicate rows (same label and same text), keeping the
       first. Duplicates would be over-weighted in training and could leak
       across a later train/test split.
    5. Reset the index to 0..n-1.

    Garbled non-ASCII characters (about 84 rows, mostly in company names,
    caused by the original encoding) are deliberately *not* repaired: the
    affected words are rare names that matter little for sentiment, and a
    guessed repair could introduce new errors.

    The input DataFrame is not modified.

    Args:
        df: Raw data with columns ``label`` and ``text``.

    Returns:
        A tuple ``(cleaned_df, report)``. ``report`` has the keys
        ``rows_in``, ``empty_removed``, ``conflicting_texts``,
        ``conflicting_rows_removed``, ``duplicate_rows_removed``,
        ``rows_out`` and ``class_counts`` (counts per label in
        ``config.LABELS`` order).
    """
    out = df.copy()
    rows_in = len(out)

    # 1. Strip whitespace (missing texts become empty strings).
    out["text"] = out["text"].fillna("").astype(str).str.strip()

    # 2. Drop empty texts.
    empty_mask = out["text"] == ""
    empty_removed = int(empty_mask.sum())
    out = out[~empty_mask]

    # 3. Drop every row of a text that occurs with more than one label.
    labels_per_text = out.groupby("text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    conflict_mask = out["text"].isin(conflicting)
    conflicting_rows_removed = int(conflict_mask.sum())
    out = out[~conflict_mask]

    # 4. Drop exact duplicates, keeping the first occurrence.
    dup_mask = out.duplicated(subset=["label", "text"], keep="first")
    duplicate_rows_removed = int(dup_mask.sum())
    out = out[~dup_mask]

    # 5. Fresh index.
    out = out.reset_index(drop=True)

    counts = out["label"].value_counts()
    report = {
        "rows_in": rows_in,
        "empty_removed": empty_removed,
        "conflicting_texts": int(len(conflicting)),
        "conflicting_rows_removed": conflicting_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_out": len(out),
        "class_counts": {label: int(counts.get(label, 0)) for label in config.LABELS},
    }
    return out, report


def load_training_data(path: Path = config.TRAINING_FILE) -> tuple[pd.DataFrame, dict]:
    """Load and clean the training data in one call.

    Args:
        path: Location of ``all-data.csv``. Defaults to ``config.TRAINING_FILE``.

    Returns:
        ``(cleaned_df, report)`` as returned by :func:`clean_training_data`.
    """
    return clean_training_data(load_raw_training_data(path))
