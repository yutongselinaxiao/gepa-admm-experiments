#### AG News: per-arm results (10 seeds)

| Arm | 256 examples: mean ± sd | 95% CI | 2,000 examples: mean ± sd | 95% CI | Spread (sd) change |
|---|---:|---:|---:|---:|---:|
| *Seed instruction* | *30.86* | | *28.90* | | |
| Reflective GEPA (12 proposals) | 34.57 ± 2.45 | ±1.75 | 32.95 ± 1.98 | ±1.42 | -19% |
| Matched reflective (30, held-out) | 35.70 ± 1.71 | ±1.22 | 34.09 ± 1.21 | ±0.86 | -29% |
| Reflective GEPA (40 proposals) | 35.59 ± 3.23 | ±2.31 | 34.59 ± 2.79 | ±1.99 | -14% |
| GEPA + top-p ADMM proposer | 36.88 ± 2.23 | ±1.60 | 35.59 ± 2.63 | ±1.88 | +18% |
| GEPA + beam ADMM (S used twice) | 38.32 ± 3.00 | ±2.15 | 36.98 ± 2.70 | ±1.93 | -10% |
| GEPA + beam ADMM (fair S') | 35.70 ± 3.71 | ±2.65 | 35.02 ± 2.66 | ±1.90 | -28% |

#### AG News: comparisons (10 seeds each, Welch's t-test)

| Comparison | 256 examples: difference | p | 2,000 examples: difference | p |
|---|---:|---:|---:|---:|
| GEPA + beam ADMM (fair S') − Matched reflective (30, held-out) | +0.00 | 1.000 | +0.93 | 0.333 |
| GEPA + beam ADMM (fair S') − Reflective GEPA (12 proposals) | +1.13 | 0.433 | +2.06 | 0.066 |
| GEPA + top-p ADMM proposer − Reflective GEPA (12 proposals) | +2.30 | 0.041 | +2.64 | 0.021 |

#### Emotion: per-arm results (10 seeds)

| Arm | 256 examples: mean ± sd | 95% CI | 2,004 examples: mean ± sd | 95% CI | Spread (sd) change |
|---|---:|---:|---:|---:|---:|
| *Seed instruction* | *35.94* | | *37.03* | | |
| Reflective GEPA (12 proposals) | 35.73 ± 0.36 | ±0.26 | 36.81 ± 0.41 | ±0.29 | +12% |
| Matched reflective (30, held-out) | 36.20 ± 0.65 | ±0.46 | 36.79 ± 0.66 | ±0.47 | +2% |
| GEPA + top-p ADMM proposer | 36.09 ± 0.22 | ±0.16 | 37.03 ± 0.07 | ±0.05 | -70% |
| GEPA + beam ADMM (S used twice) | 37.32 ± 1.69 | ±1.21 | 37.23 ± 0.46 | ±0.33 | -73% |
| GEPA + beam ADMM (fair S') | 37.84 ± 1.98 | ±1.42 | 37.92 ± 0.99 | ±0.71 | -50% |

#### Emotion: comparisons (10 seeds each, Welch's t-test)

| Comparison | 256 examples: difference | p | 2,004 examples: difference | p |
|---|---:|---:|---:|---:|
| GEPA + beam ADMM (fair S') − Matched reflective (30, held-out) | +1.64 | 0.030 | +1.13 | 0.009 |
| GEPA + beam ADMM (fair S') − Reflective GEPA (12 proposals) | +2.11 | 0.008 | +1.11 | 0.007 |
| GEPA + top-p ADMM proposer − Reflective GEPA (12 proposals) | +0.36 | 0.016 | +0.21 | 0.134 |

