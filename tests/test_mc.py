import numpy as np
import pytest

from qflib.mc import gbm_paths, ou_paths, cir_paths


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
