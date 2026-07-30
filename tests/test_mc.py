import numpy as np
import pytest
from scipy.stats import qmc

from qflib.mc import gbm_paths, ou_paths, cir_paths, mc_estimate, normal_sampler


# --- shapes ---

def test_gbm_paths_shape():
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 10, rng)
    assert paths.shape == (10, 51)


def test_ou_paths_shape():
    rng = np.random.default_rng(0)
    paths = ou_paths(0.02, 2.0, 0.04, 0.1, 1.0, 50, 10, rng)
    assert paths.shape == (10, 51)


def test_cir_paths_shape():
    rng = np.random.default_rng(0)
    paths = cir_paths(0.03, 2.0, 0.04, 0.1, 1.0, 50, 10, rng)
    assert paths.shape == (10, 51)


# --- initial values ---

def test_gbm_paths_start_at_s0():
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 20, 5, rng)
    assert np.all(paths[:, 0] == 100.0)


def test_ou_paths_start_at_x0():
    rng = np.random.default_rng(0)
    paths = ou_paths(0.02, 2.0, 0.04, 0.1, 1.0, 20, 5, rng)
    assert np.all(paths[:, 0] == 0.02)


def test_cir_paths_start_at_x0():
    rng = np.random.default_rng(0)
    paths = cir_paths(0.03, 2.0, 0.04, 0.1, 1.0, 20, 5, rng)
    assert np.all(paths[:, 0] == 0.03)


# --- GBM: lognormal terminal mean within 3 SE ---

def test_gbm_terminal_mean_within_3se():
    rng = np.random.default_rng(42)
    S0, mu, sigma, T = 100.0, 0.08, 0.3, 1.0
    n_paths = 50_000
    paths = gbm_paths(S0, mu, sigma, T, 100, n_paths, rng)
    S_T = paths[:, -1]

    analytic_mean = S0 * np.exp(mu * T)
    se = S_T.std(ddof=1) / np.sqrt(n_paths)

    assert abs(S_T.mean() - analytic_mean) <= 3 * se


# --- OU: analytic moments with fixed seed within tolerance ---

def test_ou_moments_within_3se():
    rng = np.random.default_rng(123)
    x0, kappa, theta, sigma, T = 0.0, 2.0, 0.04, 0.1, 2.0
    n_paths = 20_000
    paths = ou_paths(x0, kappa, theta, sigma, T, 200, n_paths, rng)
    X_T = paths[:, -1]

    mean_analytic = x0 * np.exp(-kappa * T) + theta * (1 - np.exp(-kappa * T))
    var_analytic = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * T))

    se_mean = X_T.std(ddof=1) / np.sqrt(n_paths)
    assert abs(X_T.mean() - mean_analytic) <= 3 * se_mean

    se_var = X_T.var(ddof=1) * np.sqrt(2 / (n_paths - 1))
    assert abs(X_T.var(ddof=1) - var_analytic) <= 3 * se_var


# --- CIR: positivity of truncated full-truncation scheme ---

def test_cir_paths_nonnegative():
    rng = np.random.default_rng(7)
    # Feller-violating params (sigma large relative to kappa*theta) to actually exercise truncation
    paths = cir_paths(0.03, 2.0, 0.04, 0.5, 2.0, 200, 5_000, rng)
    assert np.all(paths >= 0.0)


def test_cir_paths_feller_ok_mostly_positive():
    rng = np.random.default_rng(7)
    paths = cir_paths(0.03, 2.0, 0.04, 0.1, 2.0, 200, 5_000, rng)
    assert np.all(paths >= 0.0)
    # under Feller, essentially never truncated to exactly 0
    assert np.mean(paths == 0.0) < 0.001


# --- mc_estimate ---

def test_mc_estimate_matches_hand_computed_stats():
    payoffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, se, (lo, hi) = mc_estimate(payoffs, confidence=0.95)
    n = payoffs.size
    expected_mean = payoffs.mean()
    expected_se = payoffs.std(ddof=1) / np.sqrt(n)
    assert mean == pytest.approx(expected_mean)
    assert se == pytest.approx(expected_se)
    assert lo == pytest.approx(expected_mean - 1.959963984540054 * expected_se)
    assert hi == pytest.approx(expected_mean + 1.959963984540054 * expected_se)
    assert lo < mean < hi


def test_mc_estimate_narrower_ci_for_lower_confidence():
    rng = np.random.default_rng(0)
    payoffs = rng.standard_normal(1000)
    _, _, (lo95, hi95) = mc_estimate(payoffs, confidence=0.95)
    _, _, (lo80, hi80) = mc_estimate(payoffs, confidence=0.80)
    assert (hi80 - lo80) < (hi95 - lo95)


def test_mc_estimate_ci_coverage():
    # 2000 replicas of n=200 iid N(0,1): the nominal-95% CI should contain the
    # true mean (0) in roughly 95% of replicas (Monte Carlo error on the estimate
    # itself is sqrt(0.95*0.05/2000) ~ 0.005, so [0.93, 0.97] is a ~4-sigma band).
    rng = np.random.default_rng(1)
    hits = 0
    reps = 2000
    for _ in range(reps):
        sample = rng.standard_normal(200)
        _, _, (lo, hi) = mc_estimate(sample, confidence=0.95)
        hits += int(lo <= 0.0 <= hi)
    coverage = hits / reps
    assert 0.93 <= coverage <= 0.97


# --- normal_sampler ---

def test_normal_sampler_pseudo_shape():
    rng = np.random.default_rng(0)
    Z = normal_sampler(1000, 4, rng, method="pseudo")
    assert Z.shape == (1000, 4)


def test_normal_sampler_antithetic_sums_to_zero_per_column():
    rng = np.random.default_rng(0)
    Z = normal_sampler(2000, 3, rng, method="pseudo", antithetic=True)
    assert Z.shape == (2000, 3)
    np.testing.assert_allclose(Z.sum(axis=0), 0.0, atol=1e-10)


def test_normal_sampler_antithetic_requires_even_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        normal_sampler(1001, 2, rng, method="pseudo", antithetic=True)


def test_normal_sampler_sobol_no_scramble_matches_scipy():
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    Z = normal_sampler(64, 2, rng, method="sobol", scramble=False)
    ref = norm.ppf(qmc.Sobol(d=2, scramble=False, seed=None).random(64))
    np.testing.assert_allclose(Z, ref)


def test_normal_sampler_sobol_scramble_changes_with_seed_but_stays_standardized():
    Z1 = normal_sampler(2**14, 1, np.random.default_rng(1), method="sobol", scramble=True)
    Z2 = normal_sampler(2**14, 1, np.random.default_rng(2), method="sobol", scramble=True)
    assert not np.allclose(Z1, Z2)
    for Z in (Z1, Z2):
        assert abs(Z.mean()) < 0.05
        assert abs(Z.var(ddof=1) - 1.0) < 0.05


def test_normal_sampler_unknown_method_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        normal_sampler(100, 1, rng, method="bogus")
