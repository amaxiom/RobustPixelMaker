"""Shared figure style for every benchmark plot.

Two house rules are enforced here rather than left to each script, so a new plot cannot
quietly break them:

  * **No bold anywhere inside a figure.** Titles, axis labels, tick labels and legends
    all stay at normal weight. Matplotlib's defaults already are normal, but a style
    sheet, a seaborn import or a future edit could change that silently, so the weights
    are pinned explicitly.
  * **Viridis** as the colour map, and categorical series sampled from it rather than
    from the default categorical cycle.

Call :func:`apply_style` at the top of any plotting function, and use
:func:`series_colours` for per-method or per-series colours.
"""
from __future__ import annotations

import numpy as np

#: every weight-bearing rcParam, pinned to normal
_NO_BOLD = {
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "figure.titleweight": "normal",
}


def apply_style():
    """Pin the house style. Safe to call repeatedly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_NO_BOLD)
    plt.rcParams["image.cmap"] = "viridis"
    return plt


def series_colours(n: int, lo: float = 0.0, hi: float = 0.9):
    """`n` visually distinct colours sampled evenly from viridis."""
    import matplotlib.pyplot as plt

    return plt.cm.viridis(np.linspace(lo, hi, max(n, 1)))
