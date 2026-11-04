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
