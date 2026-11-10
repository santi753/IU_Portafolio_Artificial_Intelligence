"""Train/test split and evaluation framework.

Methodology:

* The cleaned data is split once into a stratified 80 % training part and a
  20 % test part. The split is saved as row positions in ``data/split.json``
  together with a fingerprint of the data, so it is reproducible and cannot be
  silently applied to different data.
* All model choices (features, hyperparameters, model type) are made with
  stratified k-fold cross-validation on the training part only.
* The test part is evaluated exactly once, for the final model.
* Main metric is macro-F1, because the classes are imbalanced; accuracy,
  per-class precision/recall/F1 and the confusion matrix are reported as well.
* Every modelling iteration is logged as one row in ``iterations.csv``.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from newsentiment import config

# Column order of the iterations log
ITERATION_COLUMNS = [
    "iteration",
    "model",
    "description",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "params",
    "logged_at",
]


def data_fingerprint(df: pd.DataFrame) -> str:
    """Return a SHA-256 hash of the ``label`` and ``text`` columns.

    Row order is part of the hash, because the stored split refers to row
    positions: the same rows in a different order would need a different split.

    Args:
        df: Data with columns ``label`` and ``text``.

    Returns:
        The hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    for label, text in zip(df["label"].astype(str), df["text"].astype(str)):
        # A tab separates label and text, a newline separates rows; neither
        # occurs inside the sentences, so the encoding is unambiguous.
        digest.update(f"{label}\t{text}\n".encode("utf-8"))
    return digest.hexdigest()


def create_split(df: pd.DataFrame, path: Path = config.SPLIT_FILE, force: bool = False) -> dict:
    """Create the stratified train/test split and save it as JSON.

    If the split file already exists and ``force`` is False, it is not
    overwritten and its content is returned unchanged. This keeps the test set
    fixed across notebook re-runs.

    Args:
        df: The cleaned data with columns ``label`` and ``text``.
        path: Where to store the split. Defaults to ``config.SPLIT_FILE``.
        force: Overwrite an existing split file.

    Returns:
        The split as a dict with the keys ``seed``, ``test_size``, ``n_rows``,
        ``fingerprint``, ``train_idx`` and ``test_idx`` (sorted row positions).
    """
    path = Path(path)
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    # Split row positions (not the data itself), stratified by label
    positions = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        positions,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df["label"],
    )
    split = {
        "seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "n_rows": len(df),
        "fingerprint": data_fingerprint(df),
        "train_idx": sorted(int(i) for i in train_idx),
        "test_idx": sorted(int(i) for i in test_idx),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # One key per line, index lists kept on a single line each (compact but readable)
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in split.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return split


def load_split(df: pd.DataFrame, path: Path = config.SPLIT_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the stored split to ``df``.

    Args:
        df: The cleaned data the split was created from.
        path: Location of the split file. Defaults to ``config.SPLIT_FILE``.

    Returns:
        ``(train_df, test_df)``, each with a fresh 0..n-1 index.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If ``df`` has a different row count or fingerprint than
            the data the split was created from.
    """
    split = json.loads(Path(path).read_text(encoding="utf-8"))

    # Refuse to apply the split to different data: the row positions would
    # point to other sentences and could mix test rows into training.
    if split["n_rows"] != len(df):
        raise ValueError(
            f"Split was created for {split['n_rows']} rows, but the data has {len(df)} rows."
        )
    if split["fingerprint"] != data_fingerprint(df):
        raise ValueError("Data fingerprint does not match the split file; the data has changed.")

    train_df = df.iloc[split["train_idx"]].reset_index(drop=True)
    test_df = df.iloc[split["test_idx"]].reset_index(drop=True)
    return train_df, test_df


def evaluate(y_true, y_pred) -> dict:
    """Compute the evaluation metrics for one set of predictions.

    All per-class results and the confusion matrix follow ``config.LABELS``.
    Classes without predictions get precision 0 instead of a warning.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        A dict with ``accuracy``, ``macro_f1``, ``per_class`` (label ->
        precision/recall/f1/support), ``labels`` and ``confusion_matrix``
        (nested list; rows = true class, columns = predicted class).
    """
    labels = list(config.LABELS)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _cv_splitter() -> StratifiedKFold:
    """Return the cross-validation splitter shared by all CV functions.

    Shuffled, stratified folds with a fixed seed: every model is compared on
    exactly the same folds, and the out-of-fold predictions of
    :func:`cross_val_predictions` come from the same folds as the scores of
    :func:`cross_validate_model`.
    """
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


def cross_validate_model(estimator, X, y) -> dict:
    """Run stratified k-fold cross-validation on the training data.

    Args:
        estimator: An unfitted scikit-learn estimator or pipeline.
        X: Training inputs (e.g. a Series of texts).
        y: Training labels.

    Returns:
        A dict with ``accuracy_mean``, ``accuracy_std``, ``macro_f1_mean``,
        ``macro_f1_std``, ``fold_accuracy`` and ``fold_macro_f1`` (lists with
        one score per fold).
    """
    scores = cross_validate(estimator, X, y, cv=_cv_splitter(), scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"})

    result: dict[str, Any] = {}
    for name in ("accuracy", "macro_f1"):
        fold_scores = scores[f"test_{name}"]
        result[f"{name}_mean"] = float(np.mean(fold_scores))
        result[f"{name}_std"] = float(np.std(fold_scores))
        result[f"fold_{name}"] = [float(s) for s in fold_scores]
    return result


def cross_val_predictions(estimator, X, y) -> np.ndarray:
    """Return out-of-fold predictions for every training row.

    Uses the same stratified folds as :func:`cross_validate_model`. Every
    prediction comes from a model that did not see that sentence during
    training, so the predictions can be passed to :func:`evaluate` for honest
    per-class metrics without touching the test set.

    Args:
        estimator: An unfitted scikit-learn estimator or pipeline.
        X: Training inputs (e.g. a Series of texts).
        y: Training labels.

    Returns:
        An array with one predicted label per row of ``X``, in input order.
    """
    return cross_val_predict(estimator, X, y, cv=_cv_splitter())


def tune_model(estimator, param_grid: dict, X, y, *, n_jobs: int = -1) -> tuple[dict, pd.DataFrame]:
    """Grid search over ``param_grid`` with cross-validation on the training data.

    Every candidate is scored on the same stratified folds as every other
    iteration (:func:`_cv_splitter`), with macro-F1 as the selection metric and
    accuracy as a secondary metric.

    Note: picking the best of many cross-validation scores makes that best
    score slightly optimistic (some of its lead over the other candidates is
    luck on these particular folds). The unbiased estimate of the chosen
    model comes later, from the locked test set.

    Args:
        estimator: An unfitted scikit-learn estimator or pipeline.
        param_grid: Parameter grid in scikit-learn's ``step__param`` notation,
            e.g. ``{"clf__C": [0.1, 1, 10]}``.
        X: Training inputs (e.g. a Series of texts).
        y: Training labels.
        n_jobs: Parallel jobs for the search; ``-1`` uses all CPU cores.

    Returns:
        ``(best_params, results)``: the parameters of the candidate with the
        highest mean macro-F1, and a DataFrame with one row per candidate
        (one column per parameter, ``mean_macro_f1``, ``std_macro_f1``,
        ``mean_accuracy``, ``std_accuracy``, ``rank``), sorted by rank.
    """
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=_cv_splitter(),
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit="macro_f1",
        n_jobs=n_jobs,
    )
    search.fit(X, y)

    cv = search.cv_results_
    results = pd.DataFrame(list(cv["params"]))
    for name in ("macro_f1", "accuracy"):
        results[f"mean_{name}"] = cv[f"mean_test_{name}"]
        results[f"std_{name}"] = cv[f"std_test_{name}"]
    results["rank"] = cv["rank_test_macro_f1"]
    # Stable sort keeps the grid order among tied candidates
    results = results.sort_values("rank", kind="stable").reset_index(drop=True)
    return dict(search.best_params_), results


def log_iteration(
    iteration: str,
    model: str,
    description: str,
    cv_result: dict,
    params: dict,
    path: Path = config.ITERATIONS_FILE,
) -> pd.DataFrame:
    """Add or replace one row in the iterations log.

    A row with the same ``iteration`` name is replaced, so re-running a
    notebook does not create duplicate rows.

    Args:
        iteration: Short unique name of the iteration, e.g. ``"baseline"``.
        model: Human-readable model name.
        description: What was changed in this iteration and why.
        cv_result: Output of :func:`cross_validate_model`.
        params: Model parameters, stored as a JSON string.
        path: Location of the CSV log. Defaults to ``config.ITERATIONS_FILE``.

    Returns:
        The full log after the update.
    """
    path = Path(path)
    row = {
        "iteration": iteration,
        "model": model,
        "description": description,
        "cv_accuracy_mean": round(cv_result["accuracy_mean"], 4),
        "cv_accuracy_std": round(cv_result["accuracy_std"], 4),
        "cv_macro_f1_mean": round(cv_result["macro_f1_mean"], 4),
        "cv_macro_f1_std": round(cv_result["macro_f1_std"], 4),
        # default=str keeps non-JSON values (e.g. tuples, None-like objects) readable
        "params": json.dumps(params, sort_keys=True, default=str),
        "logged_at": datetime.now().isoformat(timespec="seconds"),
    }

    if path.is_file():
        log = pd.read_csv(path)
        positions = np.flatnonzero(log["iteration"].astype(str) == iteration)
        if len(positions) > 0:
            # Upsert: put the new row where the old one was (keeps the log
            # order) and drop every row with the same name.
            first = int(positions[0])
            others = log[log["iteration"].astype(str) != iteration]
            log = pd.concat(
                [others.iloc[:first], pd.DataFrame([row]), others.iloc[first:]], ignore_index=True
            )
        else:
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])

    log = log[ITERATION_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log
