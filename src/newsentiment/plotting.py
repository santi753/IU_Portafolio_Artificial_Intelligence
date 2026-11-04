"""Shared matplotlib style and figure saving for all project figures.

Every figure uses the same look (see :func:`apply_style`) and the class
colours from ``config.CLASS_COLORS``. Figures carry no titles; captions are
added on the slides instead.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from newsentiment import config

TEXT_COLOR = "#333333"
GRID_COLOR = "#e6e6e6"


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
