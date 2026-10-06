# Projection diagnostics: what the ADMM Z-step does

Produced by [`src/gepa_pilot/projection_diagnostics.py`](../../src/gepa_pilot/projection_diagnostics.py).
Full numbers are in [`report.json`](report.json).

Sources:
- 18 saved iterates (iterations 5–30) from the three standalone ρ=1 runs:
  `admm_rho1`, `repeat_original_admm_rho1`, `parity_refactor_admm_rho1`.
- All 117 token edits the ADMM proposer made inside GEPA (`artifacts/phase1*`).

All losses use a fixed 32-example training batch. Validation data is not used.
On this batch the seed instruction has loss 8.75 and accuracy 25%.

## 1. Continuous gains are lost entirely at projection, and the Lipschitz bound is vacuous

The **projection gap** is F(E(z)) − F(e): the loss of the projected discrete
prompt minus the loss of the continuous iterate. The Lipschitz bound is
L_seg · ‖e − E(z)‖, where L_seg is the largest gradient norm at 9 points on
the segment from e to E(z).

| Quantity | Range over the 18 iterates |
|---|---|
| Continuous loss F(e) | 1.8–7.5 (usually about 2) |
| Discrete loss F(E(z)) | 6.8–14.2 (usually 8.75 = the seed, since z is still the seed) |
| Projection gap | −0.2 to 11.9 nats (typically about 6.4) |
| Primal residual ‖e − E(z)‖ | 3.6–10.2 |
| L_seg | 178–1646 |
| Bound L_seg · r | 654–10,996 nats |

- **The bound exceeds the actual gap by 100–1000×.** It can never certify that
  projection preserves loss.
- **The loss surface is very steep along the segment.** That is why ADMM's
  usual residual-based reasoning ("small residual, small damage") gives no
  guarantee here: with these gradient norms, a residual would have to be tiny.

![Lipschitz bound vs actual gap](../figures/fig1_gap_vs_bound.png)

*Figure 1. Each pair of points is one saved iterate: the Lipschitz bound
(squares) against the actual loss change from projection (circles), on a log
scale. One iterate with a gap of −0.24 is plotted at |gap|.*

## 2. Tokens are far apart; ADMM's projection targets are farther still

| Quantity | Median | 10th–90th percentile |
|---|---:|---:|
| Token embedding norm | 1.58 | 1.33–1.77 |
| Distance from a token to its nearest other token | 1.37 | 0.95–1.70 |
| Distance from an ADMM projection target e + u to its chosen token | 3–10 per iterate | up to 16.7 |

Random displacements of the 25 seed-token embeddings, then Euclidean projection
(200 samples per radius):

| Radius | Share of positions keeping their seed token | Distinct replacement tokens | ASCII share of replacements | Most common replacements |
|---:|---:|---:|---:|---|
| 1–5 | 100% | 0 | — | — |
| 7.5 | 99.5% | 19 | 96% | `output`, ` Letter`, ` Output`, ` detail` |
| 10 | 94.7% | 185 | 77% | ` Mapping`, `输出` ("output"), `\tlabel`, ` determines`, ` Capital` |
| 15 | 48.7% | 2,187 | 72% | ` Capital`, `\tlabel`, `output`, ` Letter`, `和` |
| 20 | 16.8% | 3,927 | 70% | ` a`, `.`, `\n`, ` in`, `label` |

Reading:
- **Each token has a very large region of embedding space that projects back
  to it.** A random move needs a radius of about 10, roughly 6× a token's norm,
  to change even 5% of tokens. Projection is therefore extremely "sticky",
  which explains why ADMM so often returns the seed unchanged.
- **Random moves land on reasonable tokens.** Before radius 20, the
  replacements are mostly case, whitespace or language variants of the
  original word: `output` → `Output` / `输出`, `label` → `\tlabel`. So
  Euclidean projection itself is fairly well behaved semantically.

![Random displacement](../figures/fig2_random_displacement.png)

*Figure 2. Seed-instruction embeddings displaced in random directions (200
samples per radius), then projected to the nearest allowed token. The shaded
band is where ADMM's actual projection targets sit (median distance per
iterate, 3–10).*

## 3. ADMM's garbage tokens come from the gradient, not from the geometry

Random moves land on sensible variants (§2), yet the Euclidean ADMM proposer's
edits are mostly unreadable: CJK characters, code identifiers, emoji. The
difference is the **direction**: the task gradient pushes e + u toward tokens
that change the model's prediction, whatever they look like. Gradient-based
discrete prompt search such as GCG produces the same kind of adversarial
tokens.

**Correction (fluency analysis, below):** an earlier version of this section
attributed the garbage to *low-norm*, under-trained "glitch" tokens. The
Euclidean edits' new tokens do sit low in the norm distribution (median 20th
percentile; 24 of 117 below the 10th). But the **fluent** edits from the top-p
and beam projections sit nearly as low (median 27th and 24th percentile), so
embedding norm does **not** distinguish garbage from fluent edits in this
vocabulary; ordinary English words also have below-median norms. What *does*
separate them is the LM's NLL of the inserted token in context: a median of
21.9 nats for Euclidean edits, against 9.3 for top-p and 5.9 for beam (AG
News). See [`../fluency_analysis/`](../fluency_analysis/) and
[`../prompt_examples/README.md`](../prompt_examples/README.md).

![Filter keep rate](../figures/fig3_filter_keep_rate.png)

*Figure 3. For each projection, the share of ADMM edits whose most surprising
inserted token stays under a given NLL threshold, i.e. what a post-hoc
perplexity filter at that threshold would keep. At 12 nats it keeps about 10%
of Euclidean edits and nearly all top-p and beam edits.*

## 4. Changing the distance used for projection doesn't help

The same 18 projection targets e + u were projected under each distance:

| Projection | Mean loss | Mean accuracy | Tokens changed from seed (mean, max) | ASCII share of changes |
|---|---:|---:|---|---:|
| Euclidean (current) | 8.77 | 26.0% | 0.5, 2 | 75% |
| Euclidean, ASCII-only vocabulary | 8.45 | 26.4% | 0.4, 2 | 100% |
| Cosine | 8.50 | 25.7% | 0.6, 3 | 70% |
| Whitened (Mahalanobis) | 9.73 | 25.3% | 2.7, 8 | 36% |
| All-but-the-top (mean and top 20 directions removed, then cosine) | 9.28 | 25.7% | 1.2, 5 | 16% |

- Whitening and all-but-the-top change more tokens, but almost all into
  garbage, and they score worse.
- Cosine and ASCII-only Euclidean are about the same as the current Euclidean
  projection.
- The problem is *where e + u sits*, far off the token manifold and in an
  exploitative direction, not how distance is measured.

![Alternative projections](../figures/fig4_alternative_projections.png)

*Figure 4. The same 18 projection targets e + u, projected under each
distance.*

## 5. Bug found: control tokens leaked into the allowed vocabulary

The whitened projection selected `<tool_call>`. Qwen3 registers 12 control
tokens with `special=False`. The original filter in `build_allowed_token_ids`
caught only those spelled `<|…|>`, so 6 of them leaked through: `<tool_call>`,
`</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`.

The filter now excludes every added token, which takes the allowed vocabulary
from 151,613 to 151,607. **No reported ADMM prompt contained any of these
tokens**, so no earlier result changes.

## Implications

1. To get sensible tokens, **constrain where the iterate goes**, rather than
   changing the distance used for projection: a context top-p candidate set, a
   perplexity term in the Z-step, or a larger or adaptive ρ that keeps e near
   real tokens. (A norm-based blocklist would not work: fluent edits also
   use low-norm tokens. Constraining NLL does.) The top-p and beam projections
   that were built afterwards confirm this: see Figure 3 and
   [`CLASSIFICATION_RESULTS.md`](../../CLASSIFICATION_RESULTS.md) §5.8–5.9.
2. **Lowering cross-entropy doesn't raise accuracy here**, even for the
   continuous soft prompt. In the pilot's ρ=1 run, the soft prompt reached
   validation loss 1.6 (seed: 9.26) while its accuracy stayed at 25.6–33.2%
   (seed: 30.5%). The loss falls mostly because the model becomes less
   confidently wrong, not more often right. Choosing a better surrogate (the
   margin and class-balance options) matters at least as much as fixing
   projection.

## Reproduce

```bash
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=6 python -m gepa_pilot.projection_diagnostics   # GPUs 4–7 only
```
