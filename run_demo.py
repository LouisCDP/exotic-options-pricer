"""Prices three exotic options and compares them with analytic benchmarks."""
import time

from exotics import (Market, price_asian_call, price_down_and_out_call,
                     price_lookback_floating_call)
from exotics import analytic

m = Market(S0=100.0, r=0.03, sigma=0.20, T=1.0)
K, H, STEPS, N = 100.0, 90.0, 252, 400_000


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    res = fn(*a, **k)
    return res, time.perf_counter() - t0


print(f"Market: S0={m.S0}, r={m.r}, sigma={m.sigma}, T={m.T} | {N:,} paths, {STEPS} steps\n")

print("1) Arithmetic Asian call (K=100): control-variate effect")
plain, t1 = timed(price_asian_call, m, K, STEPS, N, control_variate=False)
cv, t2 = timed(price_asian_call, m, K, STEPS, N, control_variate=True)
print(f"   plain MC            : {plain}   (stderr {plain.stderr:.5f}, {t1:.1f}s)")
print(f"   geometric control   : {cv}   (stderr {cv.stderr:.5f}, {t2:.1f}s)")
print(f"   variance reduction  : x{(plain.stderr / cv.stderr) ** 2:.0f}")
print(f"   geometric Asian (closed form, for reference): {analytic.geometric_asian_call(m.S0, K, m.r, m.sigma, m.T, STEPS):.4f}\n")

print(f"2) Down-and-out call (K={K:.0f}, barrier H={H:.0f}), continuous-monitoring benchmark")
ref = analytic.down_and_out_call(m.S0, K, H, m.r, m.sigma, m.T)
naive, _ = timed(price_down_and_out_call, m, K, H, STEPS, N, bridge=False)
br, _ = timed(price_down_and_out_call, m, K, H, STEPS, N, bridge=True)
print(f"   analytic            : {ref:.4f}")
print(f"   grid check only     : {naive}   (bias {naive.price - ref:+.4f})")
print(f"   Brownian bridge     : {br}   (bias {br.price - ref:+.4f})\n")

print("3) Floating-strike lookback call, continuous-monitoring benchmark")
ref = analytic.lookback_floating_call(m.S0, m.r, m.sigma, m.T)
naive, _ = timed(price_lookback_floating_call, m, STEPS, N, bridge=False)
br, _ = timed(price_lookback_floating_call, m, STEPS, N, bridge=True)
print(f"   analytic            : {ref:.4f}")
print(f"   grid minimum        : {naive}   (bias {naive.price - ref:+.4f})")
print(f"   Brownian bridge min : {br}   (bias {br.price - ref:+.4f})")
