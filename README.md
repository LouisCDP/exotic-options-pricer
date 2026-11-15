# Exotic Options Pricing: Monte Carlo under Black-Scholes (Python)

Vectorised, memory-bounded Monte Carlo pricing of path-dependent options, validated against
closed-form benchmarks, with a focus on **variance reduction** and **discretisation bias**.

| Option | Technique | Benchmark |
|---|---|---|
| Arithmetic Asian call | Geometric-Asian control variate + antithetic variates | Closed-form geometric Asian |
| Down-and-out barrier call | Brownian-bridge survival weights (corrects discrete monitoring) | Reiner-Rubinstein |
| Floating-strike lookback call | Exact sampling of the bridge minimum between dates | Goldman-Sosin-Gatto |

## Quick start
```
pip install -r requirements.txt
python run_demo.py
pytest
```

## Design
- `exotics/mc.py`: generic engine. Exact GBM paths generated in chunks (memory independent of the
  number of paths), antithetic pairing, optional control variate. Standard errors are computed on
  pair averages so confidence intervals remain valid.
- `exotics/pricers.py`: one function per option; payoffs are plain NumPy functions of the paths.
- `exotics/analytic.py`: closed-form benchmarks.
- `tests/`: each MC price must land within 4 standard errors of its benchmark.

## Results
S0=100, r=3%, sigma=20%, T=1y, 252 monitoring dates, 400,000 paths (95% confidence intervals):

| Product | Monte Carlo | Benchmark | Comment |
|---|---|---|---|
| Arithmetic Asian call (K=100), plain | 5.3000 +/- 0.0175 | n/a | |
| Arithmetic Asian call, control variate | 5.3000 +/- 0.0007 | n/a | variance reduced about 700x |
| Down-and-out call (K=100, H=90), grid check only | 7.9559 +/- 0.0355 | 7.7402 | biased +0.22 |
| Down-and-out call, Brownian bridge | 7.7202 +/- 0.0355 | 7.7402 | consistent with benchmark |
| Lookback floating call, grid minimum | 15.6925 +/- 0.0258 | 16.2986 | biased -0.61 |
| Lookback floating call, bridge minimum | 16.3174 +/- 0.0257 | 16.2986 | consistent with benchmark |

Takeaways: checking a barrier or minimum only on the simulation grid misses crossings between
dates, which overprices knock-outs and underprices lookbacks. Sampling the Brownian bridge removes
that bias without increasing the number of time steps.

## Possible extensions
Heston stochastic volatility, Greeks by pathwise/likelihood-ratio methods, American options via
Longstaff-Schwartz, basket/rainbow options with correlated assets.
