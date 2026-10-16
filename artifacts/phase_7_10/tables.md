# Phases 7-10: all arms on the 2k report sets

Mean ± sd over seeds; differences are Welch's t-tests against C (fair beam ADMM) and B (matched reflective). Many comparisons at n = 10: treat single p ≈ 0.03-0.05 with care.

## AG News (seed instruction 28.90)

| Arm | n | Mean ± sd | vs. C (p) | vs. B (p) | ADMM forward passes per run |
|---|---:|---:|---:|---:|---:|
| A  reflective GEPA | 10 | 32.95 ± 1.98 | -2.06 (0.066) | -1.13 (0.143) | — |
| A' reflective + held-out judge | 10 | 33.74 ± 1.58 | -1.27 (0.215) | -0.34 (0.597) | — |
| B  matched reflective (30) | 10 | 34.09 ± 1.21 | -0.93 (0.333) | — | — |
| E  fair reflect-then-polish | 10 | 36.18 ± 2.43 | +1.16 (0.321) | +2.09 (0.030) | 9,349 |
| C  fair beam ADMM | 10 | 35.02 ± 2.66 | — | +0.93 (0.333) | 7,008 |
| I  multi-start (3 runs) | 10 | 36.15 ± 1.99 | +1.13 (0.298) | +2.06 (0.014) | 20,974 |
| H  fixed beam, 36 proposals | 10 | 36.32 ± 1.87 | +1.30 (0.223) | +2.23 (0.006) | 21,008 |
| D  multi-length, LM insertion | 10 | 35.59 ± 1.72 | +0.58 (0.575) | +1.51 (0.038) | 20,221 |
| F  multi-length, gradient ins. | 10 | 38.26 ± 2.56 | +3.24 (0.013) | +4.17 (0.000) | 23,634 |
| G  gated | 10 | 36.62 ± 3.26 | +1.60 (0.245) | +2.53 (0.041) | 7,861 |

Polish effect (E − A', same judge and proposal count): +2.44 pp (p = 0.018).

## Emotion (seed instruction 37.03)

| Arm | n | Mean ± sd | vs. C (p) | vs. B (p) | ADMM forward passes per run |
|---|---:|---:|---:|---:|---:|
| A  reflective GEPA | 10 | 36.81 ± 0.41 | -1.11 (0.007) | +0.02 (0.937) | — |
| B  matched reflective (30) | 10 | 36.79 ± 0.66 | -1.13 (0.009) | — | — |
| E  fair reflect-then-polish | 10 | 37.00 ± 0.53 | -0.92 (0.021) | +0.21 (0.446) | 9,084 |
| C  fair beam ADMM | 10 | 37.92 ± 0.99 | — | +1.13 (0.009) | 7,262 |
| I  multi-start (3 runs) | 10 | 38.05 ± 0.81 | +0.12 (0.761) | +1.26 (0.001) | 21,725 |
| H  fixed beam, 36 proposals | 10 | 37.72 ± 1.10 | -0.20 (0.675) | +0.93 (0.037) | 21,766 |
| D  multi-length, LM insertion | 10 | 37.05 ± 1.24 | -0.88 (0.098) | +0.25 (0.576) | 21,254 |
| F  multi-length, gradient ins. | 10 | 37.09 ± 0.70 | -0.83 (0.045) | +0.30 (0.340) | 24,375 |
| G  gated | 10 | 37.86 ± 0.93 | -0.06 (0.891) | +1.07 (0.009) | 8,201 |

## Proposal bench (6 parents per task)

| Task | Method | Mean net | Per parent | s / proposal | ΔNLL per token |
|---|---|---:|---|---:|---:|
| agnews | gated | +6.5 | [9, 21, 2, -3, 11, -1] | 114 | +0.11 |
| agnews | ml_grad | +7.0 | [15, 19, -13, 12, 0, 9] | 260 | +0.78 |
| agnews | ml_lm | +6.8 | [31, 21, -6, -14, 2, 7] | 203 | +0.58 |
| agnews | multistart | +7.0 | [10, -5, 0, -2, 20, 19] | 215 | -0.37 |
| agnews | fixed | +3.0 | [8, 0, -10, 0, 20, 0] | 71 | -0.14 |
| emotion | gated | -5.2 | [4, 4, -18, -2, -6, -13] | 116 | -0.11 |
| emotion | ml_grad | -1.2 | [-17, -5, 11, 0, -12, 16] | 263 | +0.34 |
| emotion | ml_lm | -8.3 | [-10, -1, -45, -4, 0, 10] | 212 | +0.24 |
| emotion | multistart | -9.0 | [2, 18, -45, -3, -6, -20] | 204 | -0.01 |
| emotion | fixed | -7.7 | [2, 5, -45, -3, 6, -11] | 75 | +0.03 |

## Multi-length variants and polish outcomes (all runs; validation net vs. parent, of 512)

| Arm | Task | Variant | Proposed | Accepted on S' | Mean validation net | Share better |
|---|---|---:|---:|---:|---:|---:|
| D | agnews | -2 tokens | 71 | 47 | +13.8 | 68% |
| D | agnews | +0 tokens | 27 | 12 | -5.1 | 30% |
| D | agnews | +2 tokens | 16 | 7 | -4.5 | 31% |
| D | emotion | -2 tokens | 67 | 39 | -7.7 | 19% |
| D | emotion | +0 tokens | 30 | 10 | -4.2 | 40% |
| D | emotion | +2 tokens | 22 | 9 | -7.8 | 23% |
| F | agnews | -2 tokens | 11 | 7 | +11.8 | 73% |
| F | agnews | +0 tokens | 4 | 1 | +0.5 | 50% |
| F | agnews | +2 tokens | 101 | 65 | +9.2 | 74% |
| F | emotion | -2 tokens | 23 | 12 | -8.1 | 17% |
| F | emotion | +0 tokens | 9 | 6 | +4.9 | 67% |
| F | emotion | +2 tokens | 88 | 47 | -14.0 | 31% |

| Task | Polish outcome on reflection's draft | Count | Share |
|---|---|---:|---:|
| agnews | proposed | 64 | 53% |
| agnews | no_selection_improvement | 36 | 30% |
| agnews | no_token_change | 18 | 15% |
| agnews | no_exact_roundtrip | 2 | 2% |
| emotion | proposed | 88 | 73% |
| emotion | no_selection_improvement | 23 | 19% |
| emotion | no_token_change | 7 | 6% |
| emotion | no_exact_roundtrip | 2 | 2% |

