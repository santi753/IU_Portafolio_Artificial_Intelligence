"""Shared matplotlib style and figure saving for all project figures.

Every figure uses the same look (see :func:`apply_style`) and the class
colours from ``config.CLASS_COLORS``. Figures carry no titles; captions are
added on the slides instead.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from newsentiment import config

TEXT_COLOR = "#333333"
GRID_COLOR = "#e6e6e6"
# Confusion-matrix cells above this row percentage are dark: their text is white
DARK_CELL_PCT = 50


def apply_style() -> None:
    """Set the matplotlib rcParams shared by all figures.

    White background, font size 10, no top/right spines, a light grey grid on
    the value (y) axis only, drawn behind the data, dark grey text and a
    default figure size of 6.4 x 3.6 inches.
    """
    mpl.rcParams.update(
        {
            # Canvas
            "figure.figsize": (6.4, 3.6),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Text
            "font.size": 10,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            # Frame: only left and bottom spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid on the value axis only, behind the data
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def confusion_heatmap(cm, labels) -> plt.Figure:
    """Draw a confusion matrix as a heatmap coloured by row percentage.

    Same look as ``fig_confusion_test`` in notebook 02: rows = true class,
    columns = predicted class; each cell shows "count\\n(row %)"; the "Blues"
    colour shows the row percentage (0-100), so classes of different size are
    comparable; text on dark cells is white; no colour bar. Calls
    :func:`apply_style` first.

    Args:
        cm: Square matrix of counts (nested list or array), e.g. the
            ``confusion_matrix`` of :func:`newsentiment.evaluation.evaluate`.
        labels: Class names in the row/column order of ``cm``.

    Returns:
        The figure (not saved; pass it to :func:`save_figure`).
    """
    cm = np.asarray(cm)
    # Row percentages; a class with no true rows gets 0 % instead of a division by zero
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm * 100.0, row_sums, out=np.zeros(cm.shape), where=row_sums > 0)

    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)  # colour = row percentage
    for i in range(len(labels)):
        for j in range(len(labels)):
            dark = row_pct[i, j] > DARK_CELL_PCT
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f} %)", ha="center", va="center",
                    color="white" if dark else TEXT_COLOR)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG to ``config.FIGURES_DIR`` and close it.

    Args:
        fig: The figure to save.
        name: File name without extension, e.g. ``"fig_class_distribution"``.

    Returns:
        The path of the written PNG file.
    """
    path = config.FIGURES_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
