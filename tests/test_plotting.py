"""Tests for the shared figure helpers (newsentiment.plotting)."""

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt  # noqa: E402

from newsentiment.plotting import confusion_heatmap  # noqa: E402


def test_confusion_heatmap_cells_and_axes():
    """Every cell shows "count\\n(row %)", white on dark cells; axis labels set; no colour bar."""
    labels = ["negative", "neutral", "positive"]
    cm = [[3, 1, 0], [0, 4, 4], [0, 0, 0]]  # last row empty: 0 % instead of an error
    fig = confusion_heatmap(cm, labels)
    try:
        assert len(fig.axes) == 1  # no colour bar axis
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Predicted class"
        assert ax.get_ylabel() == "True class"
        assert [t.get_text() for t in ax.get_xticklabels()] == labels
        assert [t.get_text() for t in ax.get_yticklabels()] == labels

        cells = {t.get_text(): t.get_color() for t in ax.texts}
        assert len(ax.texts) == 9
        assert cells["3\n(75.0 %)"] == "white"  # dark cell (> 50 %)
        assert cells["1\n(25.0 %)"] != "white"
        assert cells["4\n(50.0 %)"] != "white"  # exactly 50 % is not dark
        assert "0\n(0.0 %)" in cells
    finally:
        plt.close(fig)
