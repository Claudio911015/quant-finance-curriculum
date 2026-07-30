import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qflib.plotting import apply_style, plot_paths


def test_apply_style_sets_rcparams():
    apply_style()
    assert plt.rcParams["figure.figsize"] == [9.0, 5.0]
    assert plt.rcParams["axes.grid"] is True


def test_plot_paths_returns_axes():
    t = np.linspace(0.0, 1.0, 50)
    paths = np.random.default_rng(42).standard_normal((10, 50)).cumsum(axis=1)
    ax = plot_paths(t, paths)
    assert len(ax.lines) == 10
