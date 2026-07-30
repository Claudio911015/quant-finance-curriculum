"""Discount curve built from bootstrapped nodes.

See notebooks/05-curvas/05.2-single-curve-bootstrapping.ipynb for the
bootstrapping algorithm and notebooks/05-curvas/05.4-interpolation-methods.ipynb
for the log_cubic and monotone_convex interpolators.
"""
import numpy as np


def _log_linear_df(t, times, log_dfs):
    return np.exp(np.interp(t, times, log_dfs))


_INTERPOLATORS = {
    "log_linear": _log_linear_df,
}


class DiscountCurve:
    """Discount curve P(0,t) from nodes (times, discount_factors)."""

    def __init__(self, times, discount_factors, interpolator="log_linear"):
        times = np.asarray(times, dtype=float)
        discount_factors = np.asarray(discount_factors, dtype=float)
        if times.ndim != 1 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if times[0] <= 0.0:
            raise ValueError("times must start strictly positive")
        if interpolator not in _INTERPOLATORS:
            raise ValueError(f"unknown interpolator: {interpolator!r}")
        self._times = times
        self._log_dfs = np.log(discount_factors)
        self._interp = _INTERPOLATORS[interpolator]

    def df(self, t):
        """Discount factor P(0,t), scalar or array."""
        return self._interp(t, self._times, self._log_dfs)

    def zero_rate(self, t):
        """Continuously-compounded zero rate z(t) = -ln(df(t))/t."""
        t = np.asarray(t, dtype=float)
        return -np.log(self.df(t)) / t

    def forward_rate(self, t1, t2):
        """Simple forward rate over [t1,t2]: (df(t1)/df(t2) - 1) / (t2 - t1)."""
        return (self.df(t1) / self.df(t2) - 1.0) / (t2 - t1)
