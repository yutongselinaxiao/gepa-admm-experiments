# GSM8K surrogate and gradient diagnostics

Produced by [`src/gepa_pilot/gsm8k_diagnostics.py`](../../src/gepa_pilot/gsm8k_diagnostics.py)
via [`scripts/run_gsm8k_diagnostics.sh`](../../scripts/run_gsm8k_diagnostics.sh).

**Setup.**
- Model: Qwen3-1.7B, non-thinking mode.
- Problems: 200 GSM8K *training* problems for ranking and 64 others for
  gradients. The official test split is never loaded.
- Instructions: 30, ranging from helpful through neutral to deliberately
  harmful.
- Metric: greedy exact-match accuracy, up to 384 new tokens.
- Sampled signals: k = 4 samples at temperature 0.7.
- Raw outputs: `rank_s*.json`, `rank_summary.json`, `grad.json`.

## 1. Accuracy barely depends on the instruction

The 30 instructions span 45.5–88.0% greedy accuracy, but 26 of the 30 fall
between 77.5% and 88.0%.

- **The best instructions are the shortest:** "Solve the problem." (88.0%) and
  "Answer the question." (86.5%).
- **Detailed reasoning instructions do slightly worse** (77–81%), and their
  outputs are longer.
- **Only instructions that suppress reasoning hurt badly:** "Answer
  immediately with only the final number" scores 45.5%, and "Give only a
  rough estimate" 60.5%.
- **Wrong advice is mostly ignored:** "Always add all the numbers together"
  still scores 80.0%.

## 2. Which signals rank instructions like accuracy?

Higher is oriented as better for every signal. A "disagreement" is a pair of
instructions with different accuracy that the signal orders the other way;
there are 423 such pairs.

| Signal | Needs labels? | Needs generation? | Spearman with accuracy | Disagreements |
|---|---|---|---:|---:|
| Reference solution NLL, teacher-forced (prompt-tuning loss) | yes | no | **−0.20** | 237 (56%) |
| Gold-answer NLL given the reference reasoning | yes | no | **−0.13** | 236 (56%) |
| Gold-answer NLL given the model's **own** reasoning | yes | yes | 0.74 | 81 (19%) |
| Own-answer NLL (confidence) | no | yes | 0.49 | 126 (30%) |
| Self-consistency of 4 samples | no | yes | 0.84 | 63 (15%) |
| Sampled accuracy, k = 4 (the RL objective) | yes | yes | **0.88** | 53 (13%) |

**Teacher-forced reference likelihood is useless here; it is slightly
anti-correlated with accuracy.** Only signals computed on the model's *own*
generations track accuracy.

## 3. Gradients with respect to the instruction-slot embeddings

Each instruction was measured on 4 disjoint batches of 16 problems.
- **Consistency** is the mean cosine similarity between batch gradients.
- **Alignment** is the mean cosine with a high-sample RLOO policy gradient
  (64 problems × 8 samples), used as the proxy for the true reward gradient.

| Instruction (accuracy) | Problems where all samples tie | Ref. NLL: consistency / alignment | Answer NLL | RLOO | RWR |
|---|---:|---|---|---|---|
| Seed, step-by-step (≈81%) | 77% | 0.96 / 0.06 | 0.80 / 0.01 | −0.02 / 0.01 | 0.54 / 0.05 |
| "Read the problem twice…" (77.5%) | 78% | 0.94 / 0.24 | 0.72 / 0.13 | −0.13 / 0.01 | 0.34 / 0.00 |
| "Answer immediately…" (45.5%) | 67% | 0.92 / 0.31 | 0.62 / 0.17 | **0.44 / 0.56** | 0.02 / −0.08 |

- **Likelihood objectives** give very consistent gradients (0.92–0.96) that are
  nearly orthogonal to the reward gradient. They reliably point in a direction
  that doesn't improve accuracy.
- **RLOO policy gradient** is pure noise for good instructions, because about
  77% of problems have all samples correct (or all wrong) and contribute no
  signal. It becomes consistent and aligned (0.44 / 0.56) only for the poor
  instruction, where the rewards vary.
- **Reward-weighted regression (RWR)** is moderately consistent but not aligned
  with the reward gradient.
- **Caveat:** the reference policy gradient is itself a finite-sample estimate,
  so the alignment numbers are bounded by its noise.

## Implications for generative ADMM

1. **Don't use reference-likelihood losses**, i.e. standard prompt-tuning NLL,
   as the ADMM objective for this kind of task. They rank prompts wrongly, and
   their gradients point somewhere else.
2. **Reward-weighted EM (variant b) as designed won't work** near the model's
   ceiling: most problems produce tied rewards, so the gradient is noise.
3. **A viable variant needs informative problems:**
   - keep only problems whose k samples disagree, resampling until a batch is
     full (the "dynamic sampling" fix from RL for LLM reasoning);
   - use larger k;
   - or focus on hard problems, where accuracy is mid-range.
4. **The instruction slot has little leverage on GSM8K for this model.** The
   accuracy range among reasonable instructions is about 10 points, and the
   minimal instruction is best. As on the emotion task, this limits what *any*
   prompt optimizer can gain.
