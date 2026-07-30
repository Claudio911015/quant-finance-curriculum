import numpy as np
import pytest
from qflib.market import nelson_siegel_zero, nelson_siegel_df, svi_total_variance


def test_ns_zero_short_end_limit():
    assert nelson_siegel_zero(0.0) == pytest.approx(0.045 - 0.01)


def test_ns_zero_long_end_limit():
    assert nelson_siegel_zero(1e6) == pytest.approx(0.045, abs=1e-4)


def test_ns_df_at_zero_is_one():
    assert nelson_siegel_df(0.0) == pytest.approx(1.0)


def test_ns_df_decreasing():
    t = np.linspace(0.0, 30.0, 121)
    df = nelson_siegel_df(t)
    assert np.all(np.diff(df) < 0)


def test_svi_positive_and_convex_wings():
    k = np.linspace(-2.0, 2.0, 201)
    w = svi_total_variance(k)
    assert np.all(w > 0)
    assert w[0] > w[100] and w[-1] > w[100]  # smile shape with rho=-0.4, m=0
