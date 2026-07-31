"""Monte Carlo path generators and estimator helpers.

Path generators for canonical SDEs (GBM, OU, CIR): see
notebooks/01-procesos-estocasticos/01.4-sdes-simulation.ipynb for the
derivations behind each scheme.

Estimator helpers (mc_estimate, normal_sampler): see
notebooks/04-metodos-numericos/04.1-monte-carlo-fundamentals.ipynb (mean/SE/CI)
and 04.3-variance-reduction-ii.ipynb (antithetic/Sobol sampling) for the
derivations. Reused from M6 onward wherever a plain MC price and its
confidence interval are needed.
"""
import numpy as np
from scipy.stats import norm, qmc


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Simulate geometric Brownian motion paths with the exact log-Euler scheme.

    dS_t = mu*S_t*dt + sigma*S_t*dW_t

    Exact per step (no discretization bias): S_{k+1} = S_k * exp((mu - sigma^2/2)*h + sigma*sqrt(h)*Z).

    Returns an array of shape (n_paths, n_steps + 1), column 0 = S0.
    """
    h = T / n_steps
    Z = rng.standard_normal(size=(n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z
    cum_log_increments = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(cum_log_increments)
    return paths


def ou_paths(x0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    """Simulate Ornstein-Uhlenbeck paths with the exact Gaussian transition scheme.

    dX_t = kappa*(theta - X_t)*dt + sigma*dW_t

    Exact per step: X_{k+1} = X_k*e^{-kappa*h} + theta*(1-e^{-kappa*h}) + sqrt(sigma^2/(2*kappa)*(1-e^{-2*kappa*h}))*Z.

    Returns an array of shape (n_paths, n_steps + 1), column 0 = x0.
    """
    h = T / n_steps
    decay = np.exp(-kappa * h)
    step_sd = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * h)))

    Z = rng.standard_normal(size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for k in range(n_steps):
        paths[:, k + 1] = paths[:, k] * decay + theta * (1 - decay) + step_sd * Z[:, k]
    return paths


def cir_paths(x0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    """Simulate CIR paths with the full-truncation Euler scheme (Lord, Koekkoek & van Dijk, 2010).

    dX_t = kappa*(theta - X_t)*dt + sigma*sqrt(X_t)*dW_t

    Auxiliary process may go transiently negative; coefficients use X^+ = max(X, 0),
    and the reported path at every step is X^+ (never negative).

    Returns an array of shape (n_paths, n_steps + 1), column 0 = x0, all entries >= 0.
    """
    h = T / n_steps
    Z = rng.standard_normal(size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for k in range(n_steps):
        x_pos = np.maximum(paths[:, k], 0.0)
        paths[:, k + 1] = paths[:, k] + kappa * (theta - x_pos) * h + sigma * np.sqrt(x_pos * h) * Z[:, k]
    return np.maximum(paths, 0.0)


def heston_paths(S0, v0, kappa, theta, xi, rho, r, q, T, n_steps, n_paths, rng):
    """Simulate Heston (1993) paths with the full-truncation Euler scheme for variance.

    dS_t = S_t*((r-q)*dt + sqrt(v_t)*dW_t^S)
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_t^v,  corr(dW^S, dW^v) = rho

    Same full-truncation idea as cir_paths (Lord, Koekkoek & van Dijk, 2010): the drift
    and diffusion coefficients of v use v^+ = max(v, 0), and both v and log S are stepped
    with v^+ substituted wherever v appears under a square root.

    Returns (S_T, v_T), each shape (n_paths,) -- only the terminal values, since option
    pricing here only needs S_T (see notebooks/08-volatilidad/08.3-heston-model.ipynb).
    """
    dt = T / n_steps
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(1.0 - rho**2)]])
    S = np.full(n_paths, S0, dtype=float)
    v = np.full(n_paths, v0, dtype=float)
    for _ in range(n_steps):
        Z = rng.standard_normal((n_paths, 2))
        dW = (Z @ chol.T) * np.sqrt(dt)
        v_pos = np.maximum(v, 0.0)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW[:, 0])
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW[:, 1]
    return S, np.maximum(v, 0.0)


def mc_estimate(payoffs, confidence=0.95):
    """Mean, standard error and confidence interval for an array of iid discounted payoffs.

    Returns (mean, standard_error, (ci_low, ci_high)), using the asymptotic
    normal approximation of the CLT (see 04.1 Section 1).
    """
    payoffs = np.asarray(payoffs, dtype=float)
    n = payoffs.size
    mean = payoffs.mean()
    se = payoffs.std(ddof=1) / np.sqrt(n)
    z = norm.ppf(0.5 + confidence / 2.0)
    return mean, se, (mean - z * se, mean + z * se)


def normal_sampler(n_paths, dim, rng, method="pseudo", antithetic=False, scramble=True):
    """Return an (n_paths, dim) array of standard normal draws.

    method: "pseudo" uses rng.standard_normal; "sobol" uses a scrambled Sobol'
    sequence (scipy.stats.qmc.Sobol) mapped through the inverse normal CDF
    (see 04.3 Section 3).

    antithetic=True returns n_paths draws built as n_paths/2 draws and their
    negatives (requires n_paths even); only supported with method="pseudo".

    scramble applies only to method="sobol": scrambling (Owen) makes the
    sequence honest for confidence intervals across independent replicas
    (a fresh `rng` draws the Sobol seed, so different rng -> different
    scrambling); scramble=False reproduces the unscrambled scipy sequence.
    """
    if antithetic:
        if method != "pseudo":
            raise ValueError("antithetic=True is only supported with method='pseudo'")
        if n_paths % 2 != 0:
            raise ValueError(f"antithetic=True requires an even n_paths, got {n_paths}")
        half = rng.standard_normal(size=(n_paths // 2, dim))
        return np.concatenate([half, -half], axis=0)

    if method == "pseudo":
        return rng.standard_normal(size=(n_paths, dim))
    elif method == "sobol":
        seed = int(rng.integers(0, 2**31 - 1)) if scramble else None
        sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        return norm.ppf(sampler.random(n_paths))
    else:
        raise ValueError(f"unknown method: {method!r}")
