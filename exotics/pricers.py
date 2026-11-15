"""Exotic option pricers built on the Monte Carlo engine."""
import numpy as np

from . import analytic
from .mc import Market, Result, run_mc


def price_asian_call(m: Market, K: float, steps: int = 252, n_paths: int = 200_000,
                     seed: int = 0, control_variate: bool = True) -> Result:
    """Arithmetic-average Asian call, discretely monitored `steps` times.

    The geometric Asian call (closed form) is used as control variate: the two payoffs are
    almost perfectly correlated, which cuts the variance by one to two orders of magnitude.
    """
    disc = np.exp(-m.r * m.T)

    def payoff(S, rng):
        return disc * np.maximum(S[:, 1:].mean(axis=1) - K, 0.0)

    def control(S, rng):
        return disc * np.maximum(np.exp(np.log(S[:, 1:]).mean(axis=1)) - K, 0.0)

    cv_mean = analytic.geometric_asian_call(m.S0, K, m.r, m.sigma, m.T, steps)
    return run_mc(payoff, m, steps, n_paths, seed,
                  control=control if control_variate else None,
                  control_mean=cv_mean if control_variate else None)


def price_down_and_out_call(m: Market, K: float, H: float, steps: int = 252,
                            n_paths: int = 200_000, seed: int = 0, bridge: bool = True) -> Result:
    """Down-and-out call.

    A naive simulation only checks the barrier on the grid, so it misses crossings between
    dates and overprices the option (the knock-out is under-detected).
    With bridge=True, each step is weighted by the probability that the Brownian bridge between
    two grid points stayed above H, which recovers the continuously monitored price.
    """
    disc = np.exp(-m.r * m.T)
    dt = m.T / steps

    def payoff(S, rng):
        vanilla = np.maximum(S[:, -1] - K, 0.0)
        logs = np.log(S / H)
        alive = (logs > 0.0).all(axis=1)
        if not bridge:
            return disc * vanilla * alive
        p_hit = np.exp(-2.0 * logs[:, :-1] * logs[:, 1:] / (m.sigma**2 * dt))
        survive = np.where(alive[:, None], 1.0 - p_hit, 0.0).prod(axis=1)
        return disc * vanilla * survive

    return run_mc(payoff, m, steps, n_paths, seed)


def price_lookback_floating_call(m: Market, steps: int = 252, n_paths: int = 200_000,
                                 seed: int = 0, bridge: bool = True) -> Result:
    """Floating-strike lookback call, payoff S_T - min_t S_t.

    With bridge=True the minimum of each Brownian bridge between grid points is sampled exactly,
    so the estimate converges to the continuously monitored price. The grid minimum alone is
    always above the true minimum, so it underprices the option.
    """
    disc = np.exp(-m.r * m.T)
    dt = m.T / steps

    def payoff(S, rng):
        if bridge:
            a = np.log(S[:, :-1])
            b = np.log(S[:, 1:])
            u = rng.random(a.shape)
            low = 0.5 * (a + b - np.sqrt((b - a) ** 2 - 2.0 * m.sigma**2 * dt * np.log(u)))
            running_min = np.exp(low.min(axis=1))
        else:
            running_min = S.min(axis=1)
        return disc * (S[:, -1] - running_min)

    return run_mc(payoff, m, steps, n_paths, seed)
