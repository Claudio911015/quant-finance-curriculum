"""Monte Carlo path generators for canonical SDEs (GBM, OU, CIR).

See notebooks/01-procesos-estocasticos/01.4-sdes-simulation.ipynb for the
derivations behind each scheme.
"""
import numpy as np


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
