# Cross-task comparison (report sets; mean ± sd over 10 seeds; differences vs. C, Welch)

| Arm | AG News, letter answers | Emotion | Yahoo Answers topics | AG News, label-name answers | GoEmotions, example F1 | GoEmotions, micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seed instruction | 28.90 | 37.03 | 19.55 | 80.00 | 37.03 | 45.99 |
| A: reflection, minibatch rule | 32.95 (-2.06, p=0.066) | 36.81 (**-1.11**, p=0.007) | 22.96 (-0.74, p=0.243) | 80.00 (-0.11, p=0.501) | 43.38 (**+4.68**, p=0.000) | 46.07 (-0.41, p=0.251) |
| B: reflection, S′ judge (30) | 34.09 (-0.93, p=0.333) | 36.79 (**-1.13**, p=0.009) | 24.45 (+0.75, p=0.241) | 80.03 (-0.07, p=0.659) | 44.09 (**+5.39**, p=0.000) | 46.35 (-0.12, p=0.738) |
| C: fixed-length ADMM | **35.02** (reference) | **37.92** (reference) | **23.70** (reference) | **80.11** (reference) | **38.70** (reference) | **46.48** (reference) |
| E: reflection + ADMM polish | 36.18 (+1.16, p=0.321) | 37.00 (**-0.92**, p=0.021) | 24.59 (+0.89, p=0.117) | 80.62 (+0.52, p=0.179) | 43.44 (**+4.74**, p=0.000) | 45.93 (-0.54, p=0.180) |
| D: multi-length, LM insertion | 35.59 (+0.58, p=0.575) | 37.05 (-0.88, p=0.098) | — | — | — | — |
| F: multi-length, gradient insertion | 38.26 (**+3.24**, p=0.013) | 37.09 (**-0.83**, p=0.045) | 26.93 (**+3.23**, p=0.000) | 80.04 (-0.06, p=0.903) | 43.23 (**+4.52**, p=0.000) | 48.08 (**+1.60**, p=0.009) |
| G: gated variable length | 36.62 (+1.60, p=0.245) | 37.86 (-0.06, p=0.891) | 26.67 (**+2.97**, p=0.001) | 80.28 (+0.17, p=0.424) | 41.37 (**+2.67**, p=0.031) | 46.73 (+0.25, p=0.279) |
| I: multi-start (3 runs) | 36.15 (+1.13, p=0.298) | 38.05 (+0.12, p=0.761) | 24.44 (+0.74, p=0.333) | 80.48 (+0.38, p=0.106) | 40.34 (**+1.64**, p=0.025) | 46.79 (+0.31, p=0.172) |
| H: fixed-length ADMM, 36 proposals | 36.32 (+1.30, p=0.223) | 37.72 (-0.20, p=0.675) | 24.93 (**+1.23**, p=0.048) | 80.38 (+0.27, p=0.261) | 40.68 (+1.98, p=0.109) | 46.90 (+0.42, p=0.271) |

Cells: mean, then (difference vs. C, Welch p); bold = p < 0.05.

## GoEmotions: what the final prompts do on the report set

True emotions per comment: 1.52.

| Arm | Example F1 | Micro F1 | Macro F1 | Emotions predicted per comment | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Seed instruction | 37.0 | 46.0 | 48.0 | 2.22 | 38.7 | 56.6 |
| A: reflection, minibatch rule | 43.4 | 46.1 | 47.6 | 3.24 | 33.9 | 72.0 |
| B: reflection, S′ judge (30) | 44.1 | 46.4 | 48.0 | 3.29 | 34.0 | 73.3 |
| E: reflection + ADMM polish | 43.4 | 45.9 | 47.4 | 3.28 | 33.7 | 72.4 |
| C: fixed-length ADMM | 38.7 | 46.5 | 48.5 | 2.32 | 38.5 | 58.7 |
| I: multi-start (3 runs) | 40.3 | 46.8 | 48.9 | 2.51 | 37.6 | 62.1 |
| H: fixed-length ADMM, 36 proposals | 40.7 | 46.9 | 48.9 | 2.53 | 37.7 | 62.6 |
| G: gated variable length | 41.4 | 46.7 | 48.6 | 2.75 | 36.8 | 65.6 |
| F: multi-length, gradient insertion | 43.2 | 48.1 | 49.9 | 2.75 | 37.7 | 67.4 |

F vs. B, C and the equal-compute controls I, H (Welch):

- example_f1 F-B: -0.86 (p=0.351)
- example_f1 F-C: +4.52 (p=0.000)
- example_f1 F-I: +2.88 (p=0.004)
- example_f1 F-H: +2.54 (p=0.065)
- micro_f1 F-B: +1.73 (p=0.010)
- micro_f1 F-C: +1.60 (p=0.009)
- micro_f1 F-I: +1.29 (p=0.028)
- micro_f1 F-H: +1.18 (p=0.064)
- macro_f1 F-B: +1.96 (p=0.010)
- macro_f1 F-C: +1.44 (p=0.031)
- macro_f1 F-I: +1.03 (p=0.107)
- macro_f1 F-H: +0.99 (p=0.156)

## AG News with label-name answers: how often runs leave the seed instruction

| Arm | Runs whose final prompt is the seed | Accepted children per run |
|---|---:|---:|
| A: reflection, minibatch rule | 10/10 | 1.3 |
| B: reflection, S′ judge (30) | 9/10 | 4.7 |
| C: fixed-length ADMM | 3/10 | 6.8 |
| E: reflection + ADMM polish | 6/10 | 2.7 |
| F: multi-length, gradient insertion | 2/10 | 4.4 |
| G: gated variable length | 3/10 | 4.9 |
| H: fixed-length ADMM, 36 proposals | 2/10 | 19.2 |
| I: multi-start (3 runs) | 2/10 | 6.3 |


## F and G vs. the equal-compute controls (difference, Welch p)

| Setting | F − I | F − H | G − I | G − H |
|---|---:|---:|---:|---:|
| AG News, letter answers | +2.11 (p=0.056) | +1.94 (p=0.071) | +0.47 (p=0.703) | +0.30 (p=0.808) |
| Emotion | **-0.96** (p=0.011) | -0.63 (p=0.146) | -0.18 (p=0.641) | +0.14 (p=0.763) |
| Yahoo Answers topics | **+2.49** (p=0.008) | **+2.00** (p=0.010) | **+2.23** (p=0.019) | **+1.74** (p=0.030) |
| AG News, label-name answers | -0.44 (p=0.415) | -0.33 (p=0.539) | -0.21 (p=0.352) | -0.10 (p=0.667) |
| GoEmotions, example F1 | **+2.88** (p=0.004) | +2.54 (p=0.065) | +1.03 (p=0.395) | +0.69 (p=0.650) |
| GoEmotions, micro F1 | **+1.29** (p=0.028) | +1.18 (p=0.064) | -0.06 (p=0.806) | -0.18 (p=0.655) |
