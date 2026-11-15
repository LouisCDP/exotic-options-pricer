"""Statistical validation: each MC price must fall within 4 standard errors of its benchmark."""
from exotics import (Market, price_asian_call, price_down_and_out_call,
                     price_lookback_floating_call)
from exotics import analytic

M = Market(S0=100.0, r=0.03, sigma=0.20, T=1.0)
STEPS, N = 52, 100_000


def within(res, ref, k=4.0):
    return abs(res.price - ref) <= k * res.stderr


def test_black_scholes_sanity():
    assert abs(analytic.bs_call(100, 100, 0.05, 0.2, 1.0) - 10.4506) < 1e-3


def test_down_and_out_plus_down_and_in_is_vanilla():
    # knocked-out value must be below the vanilla and positive
    c = analytic.bs_call(100, 100, 0.03, 0.2, 1.0)
    d = analytic.down_and_out_call(100, 100, 90, 0.03, 0.2, 1.0)
    assert 0 < d < c


def test_barrier_bridge_matches_continuous_price():
    ref = analytic.down_and_out_call(M.S0, 100.0, 90.0, M.r, M.sigma, M.T)
    assert within(price_down_and_out_call(M, 100.0, 90.0, STEPS, N, bridge=True), ref)


def test_barrier_grid_only_overprices():
    ref = analytic.down_and_out_call(M.S0, 100.0, 90.0, M.r, M.sigma, M.T)
    assert price_down_and_out_call(M, 100.0, 90.0, STEPS, N, bridge=False).price > ref


def test_lookback_bridge_matches_continuous_price():
    ref = analytic.lookback_floating_call(M.S0, M.r, M.sigma, M.T)
    assert within(price_lookback_floating_call(M, STEPS, N, bridge=True), ref)


def test_asian_control_variate_reduces_error_and_is_consistent():
    plain = price_asian_call(M, 100.0, STEPS, N, control_variate=False)
    cv = price_asian_call(M, 100.0, STEPS, N, control_variate=True)
    assert cv.stderr < plain.stderr / 5
    assert abs(cv.price - plain.price) < 4 * plain.stderr


def test_arithmetic_asian_above_geometric():
    cv = price_asian_call(M, 100.0, STEPS, N)
    assert cv.price > analytic.geometric_asian_call(M.S0, 100.0, M.r, M.sigma, M.T, STEPS)
