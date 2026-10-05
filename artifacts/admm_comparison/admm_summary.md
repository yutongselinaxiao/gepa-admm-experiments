# ADMM pilot summary

| Rho | Best iteration | Discrete accuracy | Discrete loss | Changed tokens | Round-trip |
|---:|---:|---:|---:|---:|---|
| 1 | 20 | 0.3262 | 8.1086 | 2 | True |
| 10 | 0 | 0.3047 | 9.2584 | 0 | True |
| 100 | 0 | 0.3047 | 9.2584 | 0 | True |

Selected configuration: rho=1, iteration 20, validation accuracy 0.3262.

The continuous relaxation improves substantially, but projection usually returns the seed. The low-rho run crosses token boundaries with a modest discrete gain and a large primal residual, so projection/feasibility is the dominant bottleneck.

## Selected projected instruction

> Carefully determine the article's main topic from the label mapping, ignores incidental details
 and output only the corresponding capital letter.

## Token changes

- Position 13: ` ignore` (10034) → ` ignores` (48278)
- Position 16: `,` (11) → `
` (198)

Official test examples unused in every run: True.
