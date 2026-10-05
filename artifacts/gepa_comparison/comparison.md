# GEPA minibatch comparison

| Run | Minibatch | Accepted | GEPA accuracy | Post-hoc accuracy | Rank disagreements |
|---|---:|---:|---:|---:|---:|
| mb4 | 4 | 0 | 0.3047 | 0.3555 | 12/66 |
| mb8 | 8 | 4 | 0.3672 | 0.3672 | 20/66 |
| mb16 | 16 | 6 | 0.3398 | 0.3398 | 23/78 |

Best fixed-seed GEPA selection: **mb8** at 0.3672 validation accuracy.

Each setting has one fixed-seed run. This comparison diagnoses minibatch sensitivity but does not estimate between-seed uncertainty.

Official test examples unused in every run: True.
