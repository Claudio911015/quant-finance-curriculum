"""Black-Scholes-Merton analytic pricing, Greeks and implied volatility.

Continuous dividend yield `q` (Merton, 1973) generalizes the plain Black-Scholes
(1973) formula; setting q=0 recovers the original. All Greeks are per unit of
their underlying parameter (e.g. vega is dPrice/dsigma, not per 1% move) and
theta is dPrice/dt (calendar time), matching QuantLib's `AnalyticEuropeanEngine`
conventions.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, r, q, sigma, T):
    S, K, r, q, sigma, T = map(float, (S, K, r, q, sigma, T))
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be strictly positive")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, r, q, sigma, T, kind="call"):
    """Black-Scholes-Merton price of a European option."""
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif kind == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_delta(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dS."""
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        return disc_q * norm.cdf(d1)
    elif kind == "put":
        return disc_q * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_gamma(S, K, r, q, sigma, T):
    """d^2 Price/dS^2 -- identical for calls and puts."""
    S, sigma, T = float(S), float(sigma), float(T)
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, r, q, sigma, T):
    """dPrice/dsigma -- identical for calls and puts."""
    S, T = float(S), float(T)
    d1, _ = _d1_d2(S, K, r, q, sigma, T)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dt (calendar time, i.e. -dPrice/dT), per year."""
    S, K, r, q, sigma, T = map(float, (S, K, r, q, sigma, T))
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    decay = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if kind == "call":
        return decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    elif kind == "put":
        return decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_rho(S, K, r, q, sigma, T, kind="call"):
    """dPrice/dr."""
    K, r, T = float(K), float(r), float(T)
    _, d2 = _d1_d2(S, K, r, q, sigma, T)
    disc_r = np.exp(-r * T)
    if kind == "call":
        return K * T * disc_r * norm.cdf(d2)
    elif kind == "put":
        return -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def implied_vol(price, S, K, r, q, T, kind="call", lo=1e-6, hi=5.0):
    """Implied volatility solving bs_price(..., sigma) = price via Brent's method.

    Raises ValueError if `price` violates static no-arbitrage bounds (in which
    case no sigma in (lo, hi) can reproduce it) or if the root is not bracketed
    within [lo, hi].
    """
    S, K, r, q, T, price = map(float, (S, K, r, q, T, price))
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        lower_bound = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q
    elif kind == "put":
        lower_bound = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    if not (lower_bound <= price <= upper_bound):
        raise ValueError(
            f"price {price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}] for a European {kind}"
        )

    def objective(sigma):
        return bs_price(S, K, r, q, sigma, T, kind) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"implied_vol: root not bracketed in [{lo}, {hi}] "
            f"(f_lo={f_lo:.6g}, f_hi={f_hi:.6g})"
        )
    return brentq(objective, lo, hi, xtol=1e-12, rtol=1e-14)
