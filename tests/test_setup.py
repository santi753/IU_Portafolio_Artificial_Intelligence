"""Smoke tests for the project skeleton and configuration."""

from pathlib import Path

import pytest

from newsentiment import config


def test_package_imports():
    """The package imports and exposes a version string."""
    import newsentiment

    assert newsentiment.__version__ == "0.1.0"


def test_project_root_contains_pyproject():
    """PROJECT_ROOT points to the repository root."""
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "name",
    [
        "DATA_RAW_DIR",
        "NEWS_DIR",
        "VALIDATION_DIR",
        "MODELS_DIR",
        "METRICS_DIR",
        "FIGURES_DIR",
        "RESULTS_DIR",
    ],
)
def test_config_directories_exist(name):
    """Every directory defined in config exists on disk."""
    path = getattr(config, name)
    assert isinstance(path, Path)
    assert path.is_dir(), f"{name} does not exist: {path}"


def test_labels_have_three_classes():
    """LABELS contains exactly the three sentiment classes."""
    assert len(config.LABELS) == 3
    assert len(set(config.LABELS)) == 3


def test_training_file_exists():
    """The training CSV is present (skipped if the dataset is not downloaded)."""
    if not config.TRAINING_FILE.is_file():
        pytest.skip(
            f"Training file not found at {config.TRAINING_FILE}; "
            "download the Kaggle Financial PhraseBank into data/raw/."
        )
    assert config.TRAINING_FILE.stat().st_size > 0


def test_every_label_has_a_colour():
    """CLASS_COLORS defines a colour for every label in LABELS."""
    for label in config.LABELS:
        assert label in config.CLASS_COLORS, f"No colour for label {label!r}"
