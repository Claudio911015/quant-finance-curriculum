"""Synthetic market data generators (parametric curves and vol surfaces)."""
import numpy as np


def nelson_siegel_zero(t, beta0=0.045, beta1=-0.01, beta2=0.01, tau=2.0):
    """Nelson-Siegel continuously-compounded zero rate at maturity t (years)."""
    t = np.asarray(t, dtype=float)
    x = t / tau
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.where(x > 0, (1 - np.exp(-x)) / x, 1.0)
    curvature = loading - np.exp(-x)  # -> 0 as t -> 0
    z = beta0 + beta1 * loading + beta2 * curvature
    return z if z.shape else float(z)


def nelson_siegel_df(t, **params):
    """Discount factor exp(-z(t) * t) from the Nelson-Siegel zero curve."""
    t = np.asarray(t, dtype=float)
    df = np.exp(-nelson_siegel_zero(t, **params) * t)
    return df if df.shape else float(df)


def svi_total_variance(k, a=0.03, b=0.12, rho=-0.4, m=0.0, sigma=0.25):
    """Raw SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(k, dtype=float)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    return w if w.shape else float(w)


def par_swap_rate(curve_fn, payment_times):
    """Par fixed rate of a vanilla swap (single-curve assumption).

    curve_fn: callable times -> discount factors (e.g. nelson_siegel_df or a
    DiscountCurve.df bound method). payment_times: increasing array of fixed-leg
    payment times (years from today). Floating leg (par, no spread) values at
    1 - df(payment_times[-1]); the par rate is that over the annuity.
    """
    payment_times = np.asarray(payment_times, dtype=float)
    dfs = curve_fn(payment_times)
    tau = np.diff(np.concatenate([[0.0], payment_times]))
    annuity = np.sum(tau * dfs)
    return (1.0 - dfs[-1]) / annuity
