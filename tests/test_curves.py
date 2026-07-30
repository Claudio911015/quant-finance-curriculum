import numpy as np
import pytest

from qflib.curves import DiscountCurve


def test_df_at_exact_nodes_matches_given_values():
    times = np.array([1.0, 2.0, 5.0])
    dfs = np.array([0.95, 0.90, 0.78])
    curve = DiscountCurve(times, dfs)
    np.testing.assert_allclose(curve.df(times), dfs, atol=1e-12)


def test_df_and_zero_rate_hand_computed_between_nodes():
    # log-linear interpolation between two nodes: ln(df) is linear in t.
    times = np.array([1.0, 3.0])
    dfs = np.array([0.95, 0.80])
    curve = DiscountCurve(times, dfs)
    t_mid = 2.0
    expected_ln_df = 0.5 * np.log(0.95) + 0.5 * np.log(0.80)
    expected_df = np.exp(expected_ln_df)
    assert curve.df(t_mid) == pytest.approx(expected_df, abs=1e-12)
    expected_zero = -expected_ln_df / t_mid
    assert curve.zero_rate(t_mid) == pytest.approx(expected_zero, abs=1e-12)


def test_forward_rate_hand_computed():
    times = np.array([1.0, 2.0, 3.0])
    dfs = np.array([0.95, 0.90, 0.85])
    curve = DiscountCurve(times, dfs)
    t1, t2 = 1.0, 2.0
    expected = (0.95 / 0.90 - 1.0) / (t2 - t1)
    assert curve.forward_rate(t1, t2) == pytest.approx(expected, abs=1e-12)


def test_df_accepts_array_input():
    times = np.array([1.0, 2.0, 3.0])
    dfs = np.array([0.95, 0.90, 0.85])
    curve = DiscountCurve(times, dfs)
    out = curve.df(np.array([1.0, 3.0]))
    np.testing.assert_allclose(out, [0.95, 0.85], atol=1e-12)


def test_non_increasing_times_raises():
    with pytest.raises(ValueError):
        DiscountCurve(np.array([2.0, 1.0, 3.0]), np.array([0.9, 0.95, 0.8]))


def test_times_not_starting_positive_raises():
    with pytest.raises(ValueError):
        DiscountCurve(np.array([0.0, 1.0]), np.array([1.0, 0.95]))


def test_unknown_interpolator_raises():
    with pytest.raises(ValueError):
        DiscountCurve(np.array([1.0, 2.0]), np.array([0.95, 0.90]), interpolator="bogus")
