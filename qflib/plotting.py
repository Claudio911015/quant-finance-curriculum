"""Shared matplotlib style and plotting helpers for the curriculum."""
import matplotlib.pyplot as plt
import numpy as np

_STYLE = {
    "figure.figsize": (9.0, 5.0),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "lines.linewidth": 1.4,
}


def apply_style() -> None:
    """Apply the repo-wide matplotlib style."""
    plt.rcParams.update(_STYLE)


def plot_paths(t: np.ndarray, paths: np.ndarray, ax=None, **kwargs):
    """Plot simulated paths (one line per row of `paths`) against time grid `t`."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(t, np.asarray(paths).T, alpha=kwargs.pop("alpha", 0.6), **kwargs)
    ax.set_xlabel("t")
    return ax
