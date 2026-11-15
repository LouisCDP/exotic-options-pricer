"""Closed-form benchmarks (zero dividend yield) used to validate the Monte Carlo prices."""
from math import erf, exp, log, sqrt


def N(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S0, K, r, sigma, T):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * N(d1) - K * exp(-r * T) * N(d2)


def geometric_asian_call(S0, K, r, sigma, T, n_obs):
    """Discretely monitored geometric Asian call (observation times iT/n, i=1..n).

    ln G is Gaussian, so the price has a Black-Scholes-like closed form.
    It is used as a control variate for the arithmetic Asian option.
    """
    n = n_obs
    m = log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2.0 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6.0 * n**2)
    d2 = (m - log(K)) / sqrt(v)
    d1 = d2 + sqrt(v)
    return exp(-r * T) * (exp(m + 0.5 * v) * N(d1) - K * N(d2))


def down_and_out_call(S0, K, H, r, sigma, T):
    """Continuously monitored down-and-out call, barrier H <= K, H < S0 (Reiner-Rubinstein)."""
    if not (H < S0 and H <= K):
        raise ValueError("requires H < S0 and H <= K")
    lam = (r + 0.5 * sigma**2) / sigma**2
    sT = sigma * sqrt(T)
    y = log(H**2 / (S0 * K)) / sT + lam * sT
    down_in = S0 * (H / S0) ** (2 * lam) * N(y) - K * exp(-r * T) * (H / S0) ** (2 * lam - 2) * N(y - sT)
    return bs_call(S0, K, r, sigma, T) - down_in


def lookback_floating_call(S0, r, sigma, T):
    """Continuously monitored floating-strike lookback call, payoff S_T - min S_t, started at inception."""
    sT = sigma * sqrt(T)
    a1 = (r + 0.5 * sigma**2) * T / sT
    a2 = a1 - sT
    a3 = (-r + 0.5 * sigma**2) * T / sT
    k = sigma**2 / (2.0 * r)
    return S0 * N(a1) - S0 * k * N(-a1) - S0 * exp(-r * T) * (N(a2) - k * N(-a3))
