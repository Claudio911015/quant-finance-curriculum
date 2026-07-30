import numpy as np
import pytest

from qflib.black import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    implied_vol,
)

# --- QuantLib reference values (AnalyticEuropeanEngine, eval_date=ql.Date(29,7,2026),
#     Actual365Fixed, flat continuous curves) ---
# Each case: (S, K, r, q, sigma, T, kind) -> dict of QuantLib outputs.
QL_CASES = [
    (
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "call"),
        {
            "price": 9.227005508154061,
            "delta": 0.5868511461347647,
            "gamma": 0.018950578755008714,
            "vega": 37.90115751001742,
            "theta": -5.089318913998339,
            "rho": 49.45810910532238,
        },
    ),
    (
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "put"),
        {
            "price": 6.3300806275499175,
            "delta": -0.3933475271719908,
            "gamma": 0.018950578755008714,
            "vega": 37.90115751001742,
            "theta": -2.293569138108272,
            "rho": -45.66483334474904,
        },
    ),
    (
        (100.0, 110.0, 0.03, 0.00, 0.25, 0.4986301369863014, "call"),
        {
            "price": 3.888204153810179,
            "delta": 0.3568497133065905,
            "gamma": 0.02112762253816076,
            "vega": 26.337173300994934,
            "theta": -7.556285058480708,
            "rho": 15.854826373113688,
        },
    ),
    (
        (100.0, 110.0, 0.03, 0.00, 0.25, 0.4986301369863014, "put"),
        {
            "price": 12.25497084740773,
            "delta": -0.6431502866934095,
            "gamma": 0.02112762253816076,
            "vega": 26.337173300994934,
            "theta": -4.305282057672766,
            "rho": -38.1801093480774,
        },
    ),
    (
        (100.0, 90.0, 0.04, 0.01, 0.30, 2.0, "call"),
        {
            "price": 23.80259779394481,
            "delta": 0.7119932702084922,
            "gamma": 0.007689914632018797,
            "vega": 46.13948779211282,
            "theta": -4.644337483276143,
            "rho": 94.79345845380885,
        },
    ),
    (
        (100.0, 90.0, 0.04, 0.01, 0.30, 2.0, "put"),
        {
            "price": 8.863201638066494,
            "delta": -0.2682054030982631,
            "gamma": 0.007689914632018797,
            "vega": 46.13948779211282,
            "theta": -2.301317309591009,
            "rho": -71.36748389578558,
        },
    ),
]

ATOL = 1e-8


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_price_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    price = bs_price(S, K, r, q, sigma, T, kind)
    np.testing.assert_allclose(price, ref["price"], atol=ATOL)


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_delta_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    delta = bs_delta(S, K, r, q, sigma, T, kind)
    np.testing.assert_allclose(delta, ref["delta"], atol=ATOL)


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_gamma_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    gamma = bs_gamma(S, K, r, q, sigma, T)
    np.testing.assert_allclose(gamma, ref["gamma"], atol=ATOL)


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_vega_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    vega = bs_vega(S, K, r, q, sigma, T)
    np.testing.assert_allclose(vega, ref["vega"], atol=ATOL)


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_theta_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    theta = bs_theta(S, K, r, q, sigma, T, kind)
    np.testing.assert_allclose(theta, ref["theta"], atol=ATOL)


@pytest.mark.parametrize("params, ref", QL_CASES)
def test_bs_rho_matches_quantlib(params, ref):
    S, K, r, q, sigma, T, kind = params
    rho = bs_rho(S, K, r, q, sigma, T, kind)
    np.testing.assert_allclose(rho, ref["rho"], atol=ATOL)


# --- put-call parity: C - P = S*exp(-qT) - K*exp(-rT) ---


@pytest.mark.parametrize(
    "S, K, r, q, sigma, T",
    [
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0),
        (100.0, 110.0, 0.03, 0.00, 0.25, 0.5),
        (100.0, 90.0, 0.04, 0.01, 0.30, 2.0),
        (80.0, 120.0, 0.02, 0.03, 0.35, 0.1),
    ],
)
def test_put_call_parity(S, K, r, q, sigma, T):
    call = bs_price(S, K, r, q, sigma, T, "call")
    put = bs_price(S, K, r, q, sigma, T, "put")
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


# --- implied_vol: roundtrip implied_vol(bs_price(sigma)) == sigma ---


@pytest.mark.parametrize(
    "S, K, r, q, sigma, T, kind",
    [
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "call"),
        (100.0, 100.0, 0.05, 0.02, 0.20, 1.0, "put"),
        (100.0, 110.0, 0.03, 0.00, 0.25, 0.5, "call"),
        (100.0, 90.0, 0.04, 0.01, 0.30, 2.0, "put"),
        (100.0, 80.0, 0.06, 0.00, 0.45, 0.1, "call"),
        (100.0, 130.0, 0.01, 0.02, 0.55, 3.0, "put"),
    ],
)
def test_implied_vol_roundtrip(S, K, r, q, sigma, T, kind):
    price = bs_price(S, K, r, q, sigma, T, kind)
    iv = implied_vol(price, S, K, r, q, T, kind)
    np.testing.assert_allclose(iv, sigma, atol=1e-8)


def test_implied_vol_raises_on_arbitrage_violating_price():
    # a call price above the S*exp(-qT) upper bound has no valid implied vol
    S, K, r, q, T = 100.0, 100.0, 0.05, 0.0, 1.0
    with pytest.raises(ValueError):
        implied_vol(S * np.exp(-q * T) + 1.0, S, K, r, q, T, "call")
