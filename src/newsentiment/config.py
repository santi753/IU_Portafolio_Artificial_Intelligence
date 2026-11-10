"""Central configuration: project paths, training data location and constants.

All paths are derived from this file's location, so they work regardless of
the current working directory.
"""

from pathlib import Path

# Repository root (src/newsentiment/config.py -> two levels up from the package)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
NEWS_DIR = PROJECT_ROOT / "data" / "news"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

# Model and output directories
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
# Error-analysis files with dataset sentences; git-ignored (not redistributed)
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"

# Training data (Kaggle Financial PhraseBank, combined file)
TRAINING_FILE = DATA_RAW_DIR / "all-data.csv"

# Reproducibility
RANDOM_SEED = 42

# Sentiment classes, in fixed order
LABELS = ("negative", "neutral", "positive")

# Class colours used by every figure: a diverging red-grey-blue scheme for
# negative-neutral-positive. Red/blue (not red/green) stays distinguishable for
# colour-blind readers, and the grey middle marks "neutral" as the absence of tone.
CLASS_COLORS = {"negative": "#e34948", "neutral": "#8c8c88", "positive": "#2a78d6"}

# Train/test split and evaluation
# The split is stored once as row positions, so every notebook and script
# evaluates on exactly the same rows.
SPLIT_FILE = PROJECT_ROOT / "data" / "split.json"
# One row per modelling iteration (cross-validation scores and parameters)
ITERATIONS_FILE = METRICS_DIR / "iterations.csv"
# Share of the data held back as the final, untouched test set
TEST_SIZE = 0.2
# Number of stratified cross-validation folds on the training part
CV_FOLDS = 5

# Annotator-agreement files of the original Financial PhraseBank (latin-1, one
# "sentence@label" per line). Key = agreement level in percent. The files are
# cumulative: the 50 % file contains every sentence with at least 50 % agreement,
# the 100 % file only those all annotators agreed on.
AGREEMENT_FILES = {
    50: DATA_RAW_DIR / "Sentences_50Agree.txt",
    66: DATA_RAW_DIR / "Sentences_66Agree.txt",
    75: DATA_RAW_DIR / "Sentences_75Agree.txt",
    100: DATA_RAW_DIR / "Sentences_AllAgree.txt",
}

# Colour for single-series figures that are not about the sentiment classes
ACCENT_COLOR = "#6250d6"
