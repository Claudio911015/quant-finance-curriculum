import numpy as np
import pytest

from qflib.lsm import longstaff_schwartz


def test_single_exercise_date_matches_simulated_european():
    # m=1: the only "policy" is to wait until T, so LSM must reproduce the
    # discounted terminal payoff exactly on the same paths (no regression can change it).
    rng = np.random.default_rng(0)
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    Z = rng.standard_normal(5000)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    paths = np.column_stack([np.full(5000, S0), S_T])

    payoff = lambda S: np.maximum(K - S, 0.0)
    price, se, exercise_time = longstaff_schwartz(paths, payoff, np.array([np.exp(-r * T)]), degree=3)

    expected = (np.exp(-r * T) * payoff(S_T)).mean()
    assert price == pytest.approx(expected, abs=1e-12)
    assert np.all(exercise_time == 1)


def test_hand_computed_two_date_snell_value():
    # 4 deterministic paths, K=100, r=0 (discount factors = 1), degree=0 (constant
    # regression = sample mean over ITM paths), itm_only=True.
    #
    # Path 1: 100 -> 90  -> 70   (intrinsic at t=1: 10; at t=2: 30)
    # Path 2: 100 -> 90  -> 110  (intrinsic at t=1: 10; at t=2: 0)
    # Path 3: 100 -> 110 -> 130  (never ITM)
    # Path 4: 100 -> 110 -> 60   (intrinsic at t=1: 0; at t=2: 40)
    #
    # At t=1, ITM paths are {1,2}; continuation estimate (mean of cashflow at t=2
    # among ITM paths) = mean(30, 0) = 15 > intrinsic (10) for both -> continue.
    # Final cashflows: [30, 0, 0, 40] (r=0, no discounting) -> price = 17.5.
    paths = np.array([
        [100.0, 90.0, 70.0],
        [100.0, 90.0, 110.0],
        [100.0, 110.0, 130.0],
        [100.0, 110.0, 60.0],
    ])
    payoff = lambda S: np.maximum(100.0 - S, 0.0)
    discount_factors = np.array([1.0, 1.0])

    price, se, exercise_time = longstaff_schwartz(paths, payoff, discount_factors, degree=0, itm_only=True)

    assert price == pytest.approx(17.5, abs=1e-12)
    assert np.all(exercise_time == 2)  # none exercise early


def test_itm_only_true_and_false_agree_when_all_paths_are_itm():
    # A call struck at K=1 is (numerically) always in the money for a GBM started at S0=100:
    # itm_only True/False must then regress over the same set of paths and agree exactly.
    rng = np.random.default_rng(1)
    S0, K, r, sigma, T = 100.0, 1.0, 0.05, 0.2, 1.0
    n_paths, n_dates = 4000, 10
    dt = T / n_dates
    Z = rng.standard_normal((n_paths, n_dates))
    log_paths = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    paths = np.column_stack([np.full(n_paths, S0), np.exp(log_paths)])

    payoff = lambda S: np.maximum(S - K, 0.0)
    assert np.all(payoff(paths) > 0.0)  # sanity: every path is ITM at every date with this K

    discount_factors = np.full(n_dates, np.exp(-r * dt))
    price_itm, se_itm, _ = longstaff_schwartz(paths, payoff, discount_factors, degree=3, itm_only=True)
    price_all, se_all, _ = longstaff_schwartz(paths, payoff, discount_factors, degree=3, itm_only=False)

    assert price_itm == pytest.approx(price_all, abs=1e-12)


def test_wrong_discount_factors_length_raises():
    paths = np.zeros((10, 4))  # n_dates = 3
    payoff = lambda S: np.maximum(100.0 - S, 0.0)
    with pytest.raises(ValueError):
        longstaff_schwartz(paths, payoff, np.array([1.0, 1.0]), degree=2)  # wrong length


def test_unsupported_basis_raises():
    paths = np.zeros((10, 2))
    payoff = lambda S: np.maximum(100.0 - S, 0.0)
    with pytest.raises(ValueError):
        longstaff_schwartz(paths, payoff, np.array([1.0]), basis="laguerre")
