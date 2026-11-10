"""Tests for the split and evaluation framework (newsentiment.evaluation)."""

import json

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from newsentiment import config
from newsentiment.evaluation import (
    create_split,
    cross_validate_model,
    data_fingerprint,
    evaluate,
    load_split,
    log_iteration,
    tune_model,
)
from newsentiment.model import build_pipeline


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Imbalanced synthetic data: 120 neutral, 50 positive, 30 negative rows."""
    counts = {"negative": 30, "neutral": 120, "positive": 50}
    rows = [(label, f"{label} sentence {i}") for label, n in counts.items() for i in range(n)]
    # Shuffle with a fixed seed so the classes are not in blocks
    return pd.DataFrame(rows, columns=["label", "text"]).sample(frac=1, random_state=0).reset_index(drop=True)


def test_split_sizes_and_stratification(synthetic_df, tmp_path):
    """The split is 80/20 and keeps each class share within one row."""
    split = create_split(synthetic_df, path=tmp_path / "split.json")
    n = len(synthetic_df)
    assert len(split["test_idx"]) == round(n * config.TEST_SIZE)
    assert len(split["train_idx"]) + len(split["test_idx"]) == n

    full_counts = synthetic_df["label"].value_counts()
    test_counts = synthetic_df.iloc[split["test_idx"]]["label"].value_counts()
    for label in config.LABELS:
        expected = full_counts[label] * config.TEST_SIZE
        assert abs(test_counts[label] - expected) <= 1


def test_split_disjoint_complete_and_reproducible(synthetic_df, tmp_path):
    """Train and test do not overlap, cover all rows and are the same on every run."""
    a = create_split(synthetic_df, path=tmp_path / "a.json")
    b = create_split(synthetic_df, path=tmp_path / "b.json")
    assert set(a["train_idx"]).isdisjoint(a["test_idx"])
    assert sorted(a["train_idx"] + a["test_idx"]) == list(range(len(synthetic_df)))
    assert a["train_idx"] == sorted(a["train_idx"])
    assert a == b


def test_existing_split_not_overwritten(synthetic_df, tmp_path):
    """An existing split file is kept unless force=True."""
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"marker": "old"}), encoding="utf-8")

    assert create_split(synthetic_df, path=path) == {"marker": "old"}
    assert json.loads(path.read_text(encoding="utf-8")) == {"marker": "old"}

    new = create_split(synthetic_df, path=path, force=True)
    assert "marker" not in new
    assert json.loads(path.read_text(encoding="utf-8")) == new


def test_load_split_returns_rows(synthetic_df, tmp_path):
    """load_split returns the rows at the stored positions."""
    path = tmp_path / "split.json"
    split = create_split(synthetic_df, path=path)
    train_df, test_df = load_split(synthetic_df, path=path)
    assert len(train_df) == len(split["train_idx"])
    assert test_df["text"].tolist() == synthetic_df.iloc[split["test_idx"]]["text"].tolist()


def test_fingerprint_mismatch_raises(synthetic_df, tmp_path):
    """Changed data with the same row count is rejected."""
    path = tmp_path / "split.json"
    create_split(synthetic_df, path=path)
    changed = synthetic_df.copy()
    changed.loc[0, "text"] = "a different sentence"
    assert data_fingerprint(changed) != data_fingerprint(synthetic_df)
    with pytest.raises(ValueError, match="fingerprint"):
        load_split(changed, path=path)


def test_row_count_mismatch_raises(synthetic_df, tmp_path):
    """Data with a different row count is rejected."""
    path = tmp_path / "split.json"
    create_split(synthetic_df, path=path)
    with pytest.raises(ValueError, match="rows"):
        load_split(synthetic_df.iloc[:-1], path=path)


def test_fingerprint_depends_on_row_order(synthetic_df):
    """Reordering the rows changes the fingerprint."""
    reordered = synthetic_df.iloc[::-1].reset_index(drop=True)
    assert data_fingerprint(reordered) != data_fingerprint(synthetic_df)


def test_evaluate_hand_made_example():
    """Metrics match values computed by hand.

    true:  neg neg neu neu neu pos
    pred:  neg neu neu neu pos pos

    negative: P=1/1, R=1/2, F1=2/3
    neutral:  P=2/3, R=2/3, F1=2/3
    positive: P=1/2, R=1/1, F1=2/3
    """
    y_true = ["negative", "negative", "neutral", "neutral", "neutral", "positive"]
    y_pred = ["negative", "neutral", "neutral", "neutral", "positive", "positive"]
    result = evaluate(y_true, y_pred)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["labels"] == list(config.LABELS)
    assert result["per_class"]["negative"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )
    assert result["per_class"]["neutral"] == pytest.approx(
        {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3}
    )
    assert result["per_class"]["positive"] == pytest.approx(
        {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1}
    )
    # rows = true class, columns = predicted class
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 1], [0, 0, 1]]


def test_evaluate_zero_division():
    """A class that is never predicted gets precision 0 (no error)."""
    result = evaluate(["negative", "neutral", "positive"], ["neutral"] * 3)
    assert result["per_class"]["negative"]["precision"] == 0.0
    assert result["per_class"]["positive"]["f1"] == 0.0


def test_cross_validate_model_keys(synthetic_df):
    """Cross-validation returns mean, std and one score per fold for each metric."""
    result = cross_validate_model(
        DummyClassifier(strategy="most_frequent"), synthetic_df["text"], synthetic_df["label"]
    )
    assert set(result) == {
        "accuracy_mean", "accuracy_std", "fold_accuracy",
        "macro_f1_mean", "macro_f1_std", "fold_macro_f1",
    }
    assert len(result["fold_accuracy"]) == config.CV_FOLDS
    assert len(result["fold_macro_f1"]) == config.CV_FOLDS
    # The majority class is "neutral" with 120 of 200 rows
    assert result["accuracy_mean"] == pytest.approx(0.6)


def _cv(acc: float, f1: float) -> dict:
    """Minimal cross-validation result for logging tests."""
    return {"accuracy_mean": acc, "accuracy_std": 0.01, "macro_f1_mean": f1, "macro_f1_std": 0.02}


def test_log_iteration_upserts(tmp_path):
    """A row with the same iteration name is replaced in place, others are kept."""
    path = tmp_path / "iterations.csv"
    log_iteration("baseline", "Majority", "first", _cv(0.5, 0.2), {"a": 1}, path=path)
    log_iteration("tfidf", "LogReg", "second", _cv(0.8, 0.7), {}, path=path)
    log_iteration("baseline", "Majority", "re-run", _cv(0.123456, 0.25), {"a": 2}, path=path)

    log = pd.read_csv(path)
    assert log["iteration"].tolist() == ["baseline", "tfidf"]
    baseline = log.iloc[0]
    assert baseline["description"] == "re-run"
    assert baseline["cv_accuracy_mean"] == 0.1235  # rounded to 4 decimals
    assert json.loads(baseline["params"]) == {"a": 2}
    assert list(log.columns) == [
        "iteration", "model", "description", "cv_accuracy_mean", "cv_accuracy_std",
        "cv_macro_f1_mean", "cv_macro_f1_std", "params", "logged_at",
    ]


def test_tune_model_returns_grid_results():
    """tune_model picks best parameters from the grid and returns one row per candidate."""
    words = {
        "negative": ["loss fell down", "profit decreased sharply", "sales dropped"],
        "neutral": ["company is based in", "the board met", "report was published"],
        "positive": ["profit rose up", "sales increased strongly", "earnings grew"],
    }
    rows = [(label, f"{words[label][i % 3]} in quarter {i}") for label in config.LABELS for i in range(20)]
    df = pd.DataFrame(rows, columns=["label", "text"]).sample(frac=1, random_state=0)

    grid = {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.1, 1.0]}
    # Single process: keeps the unit test independent of worker start-up on Windows
    best_params, results = tune_model(build_pipeline("nb"), grid, df["text"], df["label"], n_jobs=1)

    assert set(best_params) == set(grid)
    for name, value in best_params.items():
        assert value in grid[name]
    assert len(results) == 4
    assert {"tfidf__ngram_range", "clf__alpha", "mean_macro_f1", "std_macro_f1",
            "mean_accuracy", "std_accuracy", "rank"} <= set(results.columns)
    assert results["rank"].is_monotonic_increasing
    assert results.iloc[0]["mean_macro_f1"] == results["mean_macro_f1"].max()
