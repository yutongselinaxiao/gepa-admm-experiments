# Seed-count comparison

## AG News

Seed instruction: 30.86% on the 256-example half, 28.90% on the 2,000-example set.

Mean ± sd [95% CI half-width] using the first n seeds:

| Arm | Measure | n = 3 | n = 5 | n = 10 |
|---|---|---|---|---|
| Reflective GEPA (12 proposals) | 256 ex. | 36.07 ± 3.63 [±9.02] | 35.62 ± 2.64 [±3.28] | 34.57 ± 2.45 [±1.75] |
|  | 2k ex. | 33.45 ± 2.08 [±5.18] | 33.74 ± 1.53 [±1.90] | 32.95 ± 1.98 [±1.42] |
| Matched reflective (30 proposals, held-out) | 256 ex. | 34.64 ± 0.98 [±2.44] | 35.78 ± 2.04 [±2.53] | 35.70 ± 1.71 [±1.22] |
|  | 2k ex. | 33.52 ± 1.55 [±3.85] | 33.94 ± 1.25 [±1.55] | 34.09 ± 1.21 [±0.86] |
| Reflective GEPA (40 proposals) | 256 ex. | 35.16 ± 1.70 [±4.23] | 35.16 ± 1.24 [±1.53] | 35.59 ± 3.23 [±2.31] |
|  | 2k ex. | 34.33 ± 1.26 [±3.13] | 34.09 ± 1.03 [±1.28] | 34.59 ± 2.79 [±1.99] |
| GEPA + top-p ADMM proposer | 256 ex. | 37.63 ± 3.32 [±8.25] | 37.11 ± 2.96 [±3.68] | 36.88 ± 2.23 [±1.60] |
|  | 2k ex. | 36.55 ± 3.75 [±9.31] | 35.62 ± 3.18 [±3.95] | 35.59 ± 2.63 [±1.88] |
| GEPA + beam ADMM proposer (S used twice) | 256 ex. | 40.76 ± 3.71 [±9.22] | 40.08 ± 3.00 [±3.72] | 38.32 ± 3.00 [±2.15] |
|  | 2k ex. | 37.97 ± 2.67 [±6.64] | 37.67 ± 2.15 [±2.67] | 36.98 ± 2.70 [±1.93] |
| GEPA + beam ADMM proposer (fair S') | 256 ex. | 38.93 ± 3.18 [±7.90] | 37.66 ± 3.01 [±3.74] | 35.70 ± 3.71 [±2.65] |
|  | 2k ex. | 37.78 ± 2.78 [±6.91] | 36.43 ± 2.82 [±3.50] | 35.02 ± 2.66 [±1.90] |

Differences (Welch p) using the first n seeds:

| Comparison | Measure | n = 3 | n = 5 | n = 10 |
|---|---|---|---|---|
| GEPA + beam ADMM proposer (fair S') − Matched reflective (30 proposals, held-out) | 256 ex. | +4.30 (p = 0.134) | +1.88 (p = 0.286) | +0.00 (p = 1.000) |
|  | 2k ex. | +4.27 (p = 0.099) | +2.49 (p = 0.126) | +0.93 (p = 0.333) |
| GEPA + beam ADMM proposer (fair S') − Reflective GEPA (12 proposals) | 256 ex. | +2.86 (p = 0.363) | +2.03 (p = 0.290) | +1.13 (p = 0.433) |
|  | 2k ex. | +4.33 (p = 0.102) | +2.69 (p = 0.109) | +2.06 (p = 0.066) |
| GEPA + beam ADMM proposer (S used twice) − Reflective GEPA (12 proposals) | 256 ex. | +4.69 (p = 0.193) | +4.45 (p = 0.038) | +3.75 (p = 0.007) |
|  | 2k ex. | +4.52 (p = 0.086) | +3.93 (p = 0.012) | +4.02 (p = 0.001) |

## Emotion

Seed instruction: 35.94% on the 256-example half, 37.03% on the 2,004-example set.

Mean ± sd [95% CI half-width] using the first n seeds:

| Arm | Measure | n = 3 | n = 5 | n = 10 |
|---|---|---|---|---|
| Reflective GEPA (12 proposals) | 256 ex. | 35.59 ± 0.60 [±1.49] | 35.52 ± 0.44 [±0.54] | 35.73 ± 0.36 [±0.26] |
|  | 2k ex. | 36.61 ± 0.72 [±1.79] | 36.60 ± 0.51 [±0.63] | 36.81 ± 0.41 [±0.29] |
| Matched reflective (30 proposals, held-out) | 256 ex. | 36.37 ± 0.75 [±1.87] | 36.30 ± 0.77 [±0.96] | 36.20 ± 0.65 [±0.46] |
|  | 2k ex. | 37.36 ± 0.58 [±1.43] | 36.89 ± 0.81 [±1.01] | 36.79 ± 0.66 [±0.47] |
| GEPA + top-p ADMM proposer | 256 ex. | 36.02 ± 0.15 [±0.37] | 36.04 ± 0.14 [±0.18] | 36.09 ± 0.22 [±0.16] |
|  | 2k ex. | 37.06 ± 0.06 [±0.14] | 37.07 ± 0.05 [±0.07] | 37.03 ± 0.07 [±0.05] |
| GEPA + beam ADMM proposer (S used twice) | 256 ex. | 36.63 ± 0.80 [±1.98] | 36.46 ± 0.64 [±0.79] | 37.32 ± 1.69 [±1.21] |
|  | 2k ex. | 37.29 ± 0.64 [±1.59] | 37.22 ± 0.47 [±0.58] | 37.23 ± 0.46 [±0.33] |
| GEPA + beam ADMM proposer (fair S') | 256 ex. | 37.15 ± 1.05 [±2.61] | 37.08 ± 1.05 [±1.31] | 37.84 ± 1.98 [±1.42] |
|  | 2k ex. | 37.44 ± 0.42 [±1.05] | 37.50 ± 0.51 [±0.64] | 37.92 ± 0.99 [±0.71] |

Differences (Welch p) using the first n seeds:

| Comparison | Measure | n = 3 | n = 5 | n = 10 |
|---|---|---|---|---|
| GEPA + beam ADMM proposer (fair S') − Matched reflective (30 proposals, held-out) | 256 ex. | +0.78 (p = 0.360) | +0.78 (p = 0.221) | +1.64 (p = 0.030) |
|  | 2k ex. | +0.08 (p = 0.851) | +0.62 (p = 0.195) | +1.13 (p = 0.009) |
| GEPA + beam ADMM proposer (fair S') − Reflective GEPA (12 proposals) | 256 ex. | +1.56 (p = 0.107) | +1.56 (p = 0.026) | +2.11 (p = 0.008) |
|  | 2k ex. | +0.83 (p = 0.177) | +0.91 (p = 0.023) | +1.11 (p = 0.007) |
| GEPA + beam ADMM proposer (S used twice) − Reflective GEPA (12 proposals) | 256 ex. | +1.04 (p = 0.150) | +0.94 (p = 0.030) | +1.59 (p = 0.016) |
|  | 2k ex. | +0.68 (p = 0.289) | +0.62 (p = 0.081) | +0.42 (p = 0.046) |

