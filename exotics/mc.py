"""Generic, memory-bounded Monte Carlo engine for path-dependent payoffs under GBM."""
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class Market:
    S0: float
    r: float
    sigma: float
    T: float


@dataclass(frozen=True)
class Result:
    price: float
    stderr: float
    n_paths: int

    @property
    def ci95(self):
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)

    def __str__(self):
        return f"{self.price:.4f} +/- {1.96 * self.stderr:.4f}"


def iter_gbm_paths(m: Market, steps: int, n_paths: int, rng: np.random.Generator,
                   antithetic: bool = True, chunk: int = 20_000) -> Iterator[np.ndarray]:
    """Yield chunks of exact GBM paths, shape (chunk, steps + 1), column 0 = S0.

    Paths are generated chunk by chunk, so memory is O(chunk * steps), independent of n_paths.
    With antithetic=True, the second half of each chunk uses the negated normals of the first half.
    """
    dt = m.T / steps
    drift = (m.r - 0.5 * m.sigma**2) * dt
    vol = m.sigma * np.sqrt(dt)
    remaining = n_paths
    while remaining > 0:
        size = min(chunk, remaining)
        if antithetic:
            size -= size % 2
            z = rng.standard_normal((size // 2, steps))
            z = np.concatenate([z, -z])
        else:
            z = rng.standard_normal((size, steps))
        log_incr = np.empty((size, steps + 1))
        log_incr[:, 0] = np.log(m.S0)
        log_incr[:, 1:] = drift + vol * z
        yield np.exp(np.cumsum(log_incr, axis=1))
        remaining -= size


def run_mc(
    payoff: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    m: Market,
    steps: int,
    n_paths: int,
    seed: Optional[int] = 0,
    antithetic: bool = True,
    control: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    control_mean: Optional[float] = None,
    chunk: int = 20_000,
) -> Result:
    """Estimate E[payoff] with optional antithetic pairing and a control variate.

    `payoff` and `control` return *discounted* per-path values.
    Standard errors are computed on antithetic pair averages, so the confidence
    interval stays valid despite the negative correlation inside each pair.
    """
    rng = np.random.default_rng(seed)
    n = sx = sxx = sy = syy = sxy = 0.0

    for paths in iter_gbm_paths(m, steps, n_paths, rng, antithetic, chunk):
        x = payoff(paths, rng)
        y = control(paths, rng) if control is not None else None
        if antithetic:
            h = len(x) // 2
            x = 0.5 * (x[:h] + x[h:])
            if y is not None:
                y = 0.5 * (y[:h] + y[h:])
        n += len(x)
        sx += x.sum()
        sxx += (x * x).sum()
        if y is not None:
            sy += y.sum()
            syy += (y * y).sum()
            sxy += (x * y).sum()

    mean_x = sx / n
    var_x = max(sxx / n - mean_x**2, 0.0)
    if control is None:
        return Result(mean_x, np.sqrt(var_x / n), int(n_paths))

    mean_y = sy / n
    var_y = max(syy / n - mean_y**2, 1e-300)
    cov = sxy / n - mean_x * mean_y
    beta = cov / var_y
    est = mean_x - beta * (mean_y - control_mean)
    var = max(var_x - 2 * beta * cov + beta**2 * var_y, 0.0)
    return Result(est, np.sqrt(var / n), int(n_paths))
