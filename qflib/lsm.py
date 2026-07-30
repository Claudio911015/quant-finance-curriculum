"""Longstaff-Schwartz least-squares Monte Carlo for Bermudan/American options.

See notebooks/04-metodos-numericos/04.4-longstaff-schwartz.ipynb for the
derivation: the continuation value is a conditional expectation, approximated
by its L2 projection onto a finite basis of the underlying, estimated by
ordinary least squares over the simulated paths.
"""
import numpy as np

from qflib.mc import mc_estimate


def longstaff_schwartz(paths, payoff_fn, discount_factors, basis="poly", degree=3, itm_only=True):
    """Least-squares Monte Carlo price of a Bermudan option (Longstaff & Schwartz, 2001).

    paths: (n_paths, n_dates+1) array of underlying values, column 0 = t=0.
    payoff_fn: callable S -> intrinsic value, applied elementwise.
    discount_factors: (n_dates,) array, discount factor from t_i to t_{i-1}, i=1..n_dates.
    basis: "poly" (monomials 1, S, S^2, ..., S^degree). Only "poly" is currently supported.
    itm_only: regress only on in-the-money paths (Longstaff-Schwartz's own recommendation).

    Returns (price, standard_error, exercise_dates_used).
    """
    if basis != "poly":
        raise ValueError(f"unsupported basis: {basis!r}")
    n_paths, n_cols = paths.shape
    n_dates = n_cols - 1
    discount_factors = np.asarray(discount_factors, dtype=float)
    if discount_factors.shape != (n_dates,):
        raise ValueError(f"discount_factors must have shape ({n_dates},), got {discount_factors.shape}")

    intrinsic = payoff_fn(paths)               # (n_paths, n_dates+1)
    cashflow = intrinsic[:, -1].copy()          # realized cashflow per path, updated backward
    exercise_time = np.full(n_paths, n_dates)

    for i in range(n_dates - 1, 0, -1):
        cashflow = cashflow * discount_factors[i]   # discount one step: t_{i+1} -> t_i
        S_i = paths[:, i]
        itm = intrinsic[:, i] > 0.0
        mask = itm if itm_only else np.ones(n_paths, dtype=bool)
        if mask.sum() > degree:                     # enough points to fit the basis
            design = np.vander(S_i[mask], degree + 1, increasing=True)
            coeffs, *_ = np.linalg.lstsq(design, cashflow[mask], rcond=None)
            continuation = design @ coeffs
            exercise_now = intrinsic[mask, i] > continuation
            idx = np.flatnonzero(mask)[exercise_now]
            cashflow[idx] = intrinsic[idx, i]
            exercise_time[idx] = i

    cashflow = cashflow * discount_factors[0]        # discount t_1 -> t_0
    mean, se, _ = mc_estimate(cashflow)
    return mean, se, exercise_time
