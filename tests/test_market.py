import numpy as np
import pytest
from qflib.market import nelson_siegel_zero, nelson_siegel_df, svi_total_variance, par_swap_rate


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


def test_par_swap_rate_hand_computed():
    # 2 payment dates, hand-picked discount factors -> par rate by the annuity formula
    payment_times = np.array([1.0, 2.0])
    dfs = {1.0: 0.95, 2.0: 0.90}
    curve_fn = lambda t: np.array([dfs[float(x)] for x in np.atleast_1d(t)])
    rate = par_swap_rate(curve_fn, payment_times)
    tau = np.array([1.0, 1.0])
    annuity = np.sum(tau * np.array([0.95, 0.90]))
    expected = (1.0 - 0.90) / annuity
    assert rate == pytest.approx(expected, abs=1e-12)


def test_par_swap_rate_reprices_to_zero():
    # by construction, the par rate makes the swap NPV (float leg - fixed leg) exactly zero
    payment_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    curve_fn = nelson_siegel_df
    K = par_swap_rate(curve_fn, payment_times)
    dfs = curve_fn(payment_times)
    tau = np.diff(np.concatenate([[0.0], payment_times]))
    float_leg = 1.0 - dfs[-1]
    fixed_leg = K * np.sum(tau * dfs)
    assert float_leg == pytest.approx(fixed_leg, abs=1e-12)


def test_par_swap_rate_flat_curve_recovers_annual_yield():
    y = 0.04
    curve_fn = lambda t: np.exp(-y * np.asarray(t, dtype=float))
    payment_times = np.arange(1, 21)  # 20 annual payments
    rate = par_swap_rate(curve_fn, payment_times)
    # discrete annual-pay par rate on a flat continuously-compounded curve converges to
    # the annually-compounded equivalent of y, not y itself -- check against that, not y directly
    annual_equivalent = np.exp(y) - 1.0
    assert rate == pytest.approx(annual_equivalent, rel=1e-6)
