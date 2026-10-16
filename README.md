# GEPA and ADMM discrete-prompt pilot

> **Results log by date:** [`RESULTS_LOG.md`](RESULTS_LOG.md).
>
> **Later results:** ADMM running *inside* GEPA (proposer, projections, acceptance, matched baselines, a second task) is consolidated in [`CLASSIFICATION_RESULTS.md`](CLASSIFICATION_RESULTS.md). GSM8K surrogate diagnostics: [`artifacts/gsm8k_diagnostics/README.md`](artifacts/gsm8k_diagnostics/README.md).

This repository is an implementation and experimental record for the pilot in
[`gepa_admm_pilot_plan.tex`](gepa_admm_pilot_plan.tex). It compares two ways to
optimize a natural-language classification instruction while keeping the
language model frozen:

- **GEPA:** proposes textual prompt revisions using local reflection and accepts
  them according to minibatch task accuracy.
- **ADMM:** optimizes a fixed-length continuous prompt embedding and repeatedly
  projects it back to valid vocabulary tokens.

The task is four-way AG News classification. The evaluator scores only the next
token at the answer position, restricted to `A`, `B`, `C`, or `D`. The main
finding so far is that **GEPA with reflection minibatch size 8 produced the best
observed validation accuracy (36.72%)**, compared with **30.47% for the seed**
and **32.62% for the best ADMM projection**. These are pilot results from one
fixed seed and are not yet estimates of between-run uncertainty.

## Experiment status

| Stage | Purpose | Status | Main result |
|---|---|---:|---|
| 0 | Build one common evaluator and pass sanity gates | Complete | All correctness gates passed; BF16 raw logits are batch-shape sensitive |
| 0B | Establish seed and manual-prompt baselines | Complete | Best manual prompt: 30.86%; seed: 30.47% |
| I-A | Smoke-test the GEPA reflection loop | Complete | Unconstrained reflection was invalid; constrained reflection was valid |
| I-B | Compare GEPA reflection minibatches 4, 8, and 16 | Complete | Minibatch 8 won at 36.72% validation accuracy |
| II | Audit agreement between loss and accuracy | Complete | Every GEPA pool contained ranking disagreements |
| III-A | Calibrate and smoke-test ADMM | Complete | Continuous optimization worked, but projection usually returned the seed |
| III-B | Run the full ADMM penalty sweep | Complete | \(\rho=1\) reached 32.62%; \(\rho=10,100\) stayed at the seed |
| IV | Multi-seed confirmation and final held-out test | Not run | Official AG News test data remains locked and unused |

## Fixed experimental contract

| Component | Setting |
|---|---|
| Dataset | `fancyzhx/ag_news` |
| Development source | Official **train** split only |
| Optimization subset | 256 examples, stratified: 64 per class |
| Validation subset | 512 examples, stratified: 128 per class |
| Data seed | `20261002` |
| Official test split | Locked in code and unused in all reported runs |
| Label mapping | `A=World`, `B=Sports`, `C=Business`, `D=Sci/Tech` |
| Frozen model | `Qwen/Qwen3-1.7B` |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Precision | BF16 |
| Decoding mode | Thinking disabled; next-token label scoring |
| Validation batching | Fixed order, right padding, batch size 32 |
| Seed instruction length | 25 tokens |
| Software | Python 3.10.12, PyTorch 2.11.0+cu128, Transformers 5.5.4, GEPA 0.1.4 |
| Recorded accelerator | NVIDIA H100 80GB HBM3 |

The prepared files are content-addressed in
[`artifacts/data/data_manifest.json`](artifacts/data/data_manifest.json). The
SHA-256 hashes are:

| File | SHA-256 |
|---|---|
| `train.json` | `f57e1b3e5b6e868b1b89014879b1dcb66aff7f96b6c5125761f39f8ca4eb7f9e` |
| `validation.json` | `a386d5a54361463646be686cda0388010bacd7b5808d55e017044907b1dada57` |

## Stage 0 — common evaluator and sanity gates

### Why this experiment

GEPA and ADMM can only be compared if prompt composition, label scoring, data
order, and metrics are identical. This stage tested those invariants before any
optimization. It also checked that gradients flow through soft prompt embeddings
without making the frozen model trainable.

### Procedure

```text
load the pinned tokenizer and frozen model
assert A, B, C, and D are each exactly one token
compose every prompt from system text + instruction + mapping + article
verify instruction insertion is compositional and article-independent
verify encode -> decode -> encode preserves the seed instruction
reject any attempt to load the official test split without an explicit override

for each sanity example:
    score the four label-token logits at the answer position
    predict argmax(A, B, C, D)
    compute cross-entropy against the gold label

repeat evaluation and require deterministic outputs
compare input_ids evaluation with inputs_embeds evaluation
backpropagate through a soft instruction and require finite nonzero gradients
compare singleton and batched BF16 evaluation as a diagnostic
```

### Results

| Gate | Result |
|---|---:|
| Label token IDs | `A=32`, `B=33`, `C=34`, `D=35`; all single-token |
| Seed instruction length | 25 tokens |
| Compositional insertion | Pass |
| Token round trip | Exact |
| Article independence | Pass |
| Test-split lock | Pass |
| Repeated evaluation | Deterministic |
| `input_ids` vs. `inputs_embeds` prediction agreement | 100% |
| `input_ids` vs. `inputs_embeds` maximum label-score difference | 0.0 |
| Soft-prompt gradient norm | `9.0419e-4`, finite and nonzero |
| Trainable model parameters | 0 |
| Singleton vs. batch prediction agreement | 100% |
| Singleton vs. batch maximum raw label-score difference | 1.5 |
| Prespecified raw-score tolerance | 0.05 — diagnostic exceeded |
| Unit tests | 15/15 passed |

The BF16 batch diagnostic did **not** show prediction changes, but raw scores
changed by more than the tolerance. Consequently, all comparisons after this
stage use the same batch size, padding policy, and example order. Raw loss values
from differently batched runs should not be compared directly.

## Stage 0B — seed and manual-prompt baselines

### Why this experiment

Baselines establish whether either optimizer improves over the starting prompt
and whether a few hand-written alternatives already solve the problem. Per-class
metrics also expose degenerate label preferences that overall accuracy can hide.

### Procedure

```text
instructions = [seed, manual_1, manual_2, manual_3]
for instruction in instructions:
    evaluate all 512 validation examples in fixed batches of 32
    record mean label cross-entropy, accuracy, per-class accuracy,
           confusion matrix, and prediction counts
select prompts by validation accuracy; use loss only as a diagnostic/tie-breaker
```

### Results

| Prompt | Accuracy | Mean loss | World (A) | Sports (B) | Business (C) | Sci/Tech (D) | Prediction counts A/B/C/D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed | 30.47% | 9.2584 | 98.44% | 7.03% | 1.56% | 14.84% | 460 / 9 / 2 / 41 |
| Manual 1 | **30.86%** | **4.5807** | 78.91% | 10.94% | 1.56% | 32.03% | 328 / 47 / 2 / 135 |
| Manual 2 | 30.08% | 8.5297 | 96.88% | 10.16% | 1.56% | 11.72% | 439 / 37 / 2 / 34 |
| Manual 3 | 30.47% | 10.6016 | 98.44% | 4.69% | 3.12% | 15.62% | 452 / 6 / 4 / 50 |

- Manual 1 had the best manual accuracy and much lower loss than the seed.
- All four prompts were strongly biased toward `A` (World).
- Business (`C`) was the persistent failure mode: no prompt exceeded 3.12% on
  that class.
- The four-prompt run took 50.95 seconds.

## Stage I — GEPA textual prompt optimization

### Why this experiment

GEPA provides a discrete, readable prompt-search baseline. The experiment asks
whether local error reflection can improve validation classification while
preserving the immutable label mapping and output contract. The minibatch sweep
tests whether proposal acceptance is overly sensitive to a small reflection
sample.

### GEPA loop

```text
start from the seed instruction
freeze the model, system prompt, label mapping, and response format

for proposal = 1 .. 12:
    sample the fixed-seed reflection minibatch
    evaluate the current instruction with binary example-level accuracy
    give local Qwen the errors and a constrained reflection prompt
    require one instruction of at most 90 words
    evaluate the candidate through the common evaluator
    let GEPA accept or reject the candidate from its minibatch score

evaluate every unique accepted and rejected candidate on all 512 val examples
report GEPA's selected candidate separately from the post-hoc validation winner
```

### Stage I-A: reflection smoke tests

- The first unconstrained smoke test sometimes hallucinated a new label mapping
  and produced truncated output.
- The reflection template was tightened to declare the mapping immutable, demand
  only the replacement instruction, and cap it at 90 words.
- The constrained smoke test produced a valid candidate. It tied the incumbent
  on its small evaluation and was correctly rejected rather than silently
  treated as an improvement.

### Stage I-B: fixed-seed minibatch comparison

Each setting used 12 proposals, merge disabled, local Qwen reflection, fixed
validation batching, and a post-hoc full-validation audit of every unique
candidate.

| Reflection minibatch | Accepted proposals | GEPA-selected accuracy | Selected loss | Post-hoc best accuracy | Best candidate accepted? | Ranking disagreements | Pearson loss vs. accuracy | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 30.47% | 9.2584 | 35.55% | No | 12 / 66 | -0.8160 | 89.3 s |
| 8 | 4 | **36.72%** | **6.4863** | **36.72%** | Yes | 20 / 66 | -0.6174 | 132.5 s |
| 16 | 6 | 33.98% | 6.9231 | 33.98% | Yes | 23 / 78 | -0.2082 | 145.6 s |

The best observed GEPA instruction, from minibatch 8, was:

> Carefully analyze the article to determine the primary subject and context,
> focusing on the main event or issue, and ignore incidental details. Select the
> corresponding capital letter from the label mapping.

Its class-level behavior was:

| Metric | World (A) | Sports (B) | Business (C) | Sci/Tech (D) |
|---|---:|---:|---:|---:|
| Per-class accuracy | 79.69% | 5.47% | 2.34% | 59.38% |
| Prediction count | 319 | 8 | 4 | 181 |

- Relative to the seed, the minibatch-8 prompt improved overall accuracy by
  6.25 percentage points, mostly by trading some World accuracy for Sci/Tech.
- Sports and Business remained poorly recognized, so the higher total accuracy
  did not resolve class imbalance in the model's predictions.
- Minibatch 4 never accepted a proposal even though one rejected proposal later
  scored 35.55% on full validation. This is direct evidence of noisy local
  acceptance.
- Minibatch 16 accepted more proposals but did not outperform minibatch 8.
- There is only one run per minibatch size. The table diagnoses sensitivity but
  does not establish statistical superiority.

## Stage II — objective-alignment audit

### Why this experiment

The common evaluator exposes both discrete accuracy and differentiable label
cross-entropy. GEPA acts on binary example scores, whereas ADMM requires a smooth
loss. Before comparing optimizers, this stage tested whether the two objectives
rank the observed textual candidates consistently.

### Procedure

```text
collect every unique GEPA candidate, including rejected candidates
evaluate each candidate on the full fixed validation set
record accuracy, mean label loss, per-class accuracy, and predictions

for every pair of candidates:
    compare the accuracy ordering with the loss ordering
    count disagreement when higher accuracy also has worse loss

compute Pearson(loss, accuracy) and Spearman(loss rank, accuracy rank)
save disagreement examples and a scatter plot
```

### Results

| GEPA pool | Pairwise disagreements | Pearson correlation | Spearman rank correlation | Highest-accuracy candidate also lowest-loss? |
|---|---:|---:|---:|---:|
| Minibatch 4 | 12 / 66 | -0.8160 | -0.7881 | No |
| Minibatch 8 | 20 / 66 | -0.6174 | -0.5965 | No |
| Minibatch 16 | 23 / 78 | -0.2082 | -0.4133 | No |

- Lower loss generally correlated with higher accuracy, but the relationship was
  not order-preserving.
- The disagreement became especially pronounced in the minibatch-16 pool.
- Decision for the first ADMM pilot: retain cross-entropy as the differentiable
  training objective, but select checkpoints by validation accuracy and use loss
  only as a tie-breaker/diagnostic.

## Stage III — fixed-length ADMM optimization

### Why this experiment

ADMM tests whether gradient information in continuous embedding space can find a
better fixed-length discrete instruction. Its central risk is not merely
optimization: a useful continuous vector must cross a nearest-token boundary and
remain useful after exact projection. The penalty sweep probes that tradeoff.

For instruction embeddings \(E\), discrete projected embeddings \(Z\), scaled
dual variables \(U\), and penalty \(\rho\), each outer iteration performs:

```text
E-step (3 gradient steps):
    minimize task_loss(E) + (rho / 2) * ||E - Z + U||^2

Z-step:
    for every one of the 25 instruction positions:
        project E + U to its exact nearest allowed vocabulary embedding

dual update:
    U <- U + E - Z

every 5 outer iterations:
    decode Z, require exact token round-trip, and evaluate on fixed validation
select the checkpoint with best discrete accuracy, breaking ties by lower loss
```

Special/control tokens are excluded from the projection vocabulary. Projection
uses exact Euclidean nearest-neighbor search over 151,613 allowed tokens, in
chunks only to control memory.

### Stage III-A: calibration and smoke sweep

Calibration on one training batch measured the scale of the task gradient and a
small synthetic displacement before choosing the full sweep.

| Calibration quantity | Value |
|---|---:|
| Batch size | 16 |
| Seed task loss | 6.9880 |
| Task-gradient norm | 119.3336 |
| Seed embedding norm | 7.4079 |
| Probe displacement norm | 1.1314 |
| Probe squared residual | 1.2800 |

| \(\rho\) | Probe penalty \((\rho/2)||r||^2\) |
|---:|---:|
| 0.0001 | 0.000064 |
| 0.001 | 0.000640 |
| 0.01 | 0.006400 |
| 0.1 | 0.064000 |
| 1 | 0.640000 |

A two-iteration smoke sweep at \(\rho \in \{1,10,100\}\) completed without
projection, parity, or checkpoint failures. Continuous losses changed, but all
three discrete prompts remained the seed during this short run.

### Stage III-B: full penalty sweep

All full runs used 30 outer iterations, 3 E-steps per iteration, learning rate
0.005, training batch size 16, validation batch size 32, and validation every 5
iterations. Each run made 314 forward calls and 90 backward calls.

| \(\rho\) | Best iteration | Discrete accuracy | Discrete loss | Continuous accuracy at best | Continuous loss at best | Changed tokens | Primal residual | Dual residual | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | **32.62%** | **8.1086** | 25.59% | 7.1142 | 2 | 6.7419 | 1.9144 | 61.1 s |
| 10 | 0 | 30.47% | 9.2584 | 30.47% | 9.2584 | 0 | 0.0000 | 0.0000 | 61.4 s |
| 100 | 0 | 30.47% | 9.2584 | 30.47% | 9.2584 | 0 | 0.0000 | 0.0000 | 61.2 s |

The selected \(\rho=1\) prompt was:

```text
Carefully determine the article's main topic from the label mapping, ignores incidental details
 and output only the corresponding capital letter.
```

It differs from the seed at two token positions:

| Position | Seed token | Selected token |
|---:|---|---|
| 13 | ` ignore` (ID 10034) | ` ignores` (ID 48278) |
| 16 | `,` (ID 11) | newline (ID 198) |

The \(\rho=1\) trajectory shows why final discrete evaluation is necessary:

| Iteration | Continuous loss | Discrete loss | Discrete accuracy | Changed tokens | Primal residual | Dual residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.2584 | 9.2584 | 30.47% | 0 | 0.0000 | 0.0000 |
| 5 | 1.8739 | 9.2584 | 30.47% | 0 | 3.6644 | 0.0000 |
| 10 | 2.3279 | 9.2584 | 30.47% | 0 | 3.5797 | 0.0000 |
| 15 | 2.3225 | 9.2584 | 30.47% | 0 | 6.0207 | 0.0000 |
| 20 | 7.1142 | 8.1086 | **32.62%** | 2 | 6.7419 | 1.9144 |
| 25 | 2.3588 | 9.2584 | 30.47% | 0 | 8.2193 | 0.0000 |
| 30 | 1.5961 | 7.9670 | 32.42% | 1 | 10.2300 | 2.3345 |

- The relaxed continuous objective often improved dramatically without changing
  a single projected token.
- \(\rho=10\) and \(\rho=100\) enforced the seed basin too strongly; their best
  checkpoints were iteration 0.
- \(\rho=1\) crossed token boundaries and improved accuracy by 2.15 percentage
  points, but the large primal residual means the continuous and discrete
  solutions were not close to consensus.
- The selected edit is grammatically worse, showing that nearest-embedding
  projection can exploit model behavior without producing a better human prompt.
- Every saved checkpoint passed exact token round-trip and `input_ids` versus
  `inputs_embeds` parity; all expected checkpoint files were present.

## Cross-stage interpretation

| Question | Current evidence |
|---|---|
| Does the implementation support a fair comparison? | Yes for the fixed evaluator and data split; all core correctness gates passed. |
| Did prompt optimization beat the seed? | Yes. GEPA reached 36.72%; ADMM reached 32.62%; seed was 30.47%. |
| Which method is currently strongest? | GEPA minibatch 8, by observed validation accuracy. |
| Is the GEPA conclusion statistically confirmed? | No. Only one fixed-seed run exists per minibatch size. |
| Does continuous ADMM loss predict discrete prompt quality? | Not reliably; large continuous improvements often vanished after projection. |
| Is loss interchangeable with accuracy? | No. All GEPA candidate pools had pairwise ranking disagreements. |
| Has the official test set been used? | No. It remains locked for a later final evaluation. |

## Reproducing the completed stages

Use the existing environment and expose one of the requested physical GPUs. The
configuration refers to logical `cuda:0`, so a process launched with
`CUDA_VISIBLE_DEVICES=4` uses physical GPU 4. Independent jobs can similarly be
assigned to GPUs 5–7.

```bash
source /dataMeR2/yutong/envs/myenv/bin/activate
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1

# Unit tests and tokenizer-only checks
python -m unittest discover -s tests -v
python -m gepa_pilot.sanity --tokenizer-only \
  --output artifacts/tokenizer_sanity_report.json

# Prepare deterministic development subsets; never reads the official test set
python -m gepa_pilot.prepare_data --output-dir artifacts/data

# Full GPU sanity gates and baselines
CUDA_VISIBLE_DEVICES=4 python -m gepa_pilot.sanity \
  --config configs/pilot.yaml \
  --examples configs/sanity_examples.json \
  --output artifacts/sanity_report.json
CUDA_VISIBLE_DEVICES=4 python -m gepa_pilot.run_baselines \
  --batch-size 32 --output artifacts/baseline_results.json
```

Run the three GEPA settings as independent jobs, for example on GPUs 4–6:

```bash
CUDA_VISIBLE_DEVICES=4 python -m gepa_pilot.run_gepa \
  --proposals 12 --reflection-minibatch-size 4 \
  --run-dir artifacts/gepa_observation
CUDA_VISIBLE_DEVICES=5 python -m gepa_pilot.run_gepa \
  --proposals 12 --reflection-minibatch-size 8 \
  --run-dir artifacts/gepa_confirmation_mb8
CUDA_VISIBLE_DEVICES=6 python -m gepa_pilot.run_gepa \
  --proposals 12 --reflection-minibatch-size 16 \
  --run-dir artifacts/gepa_confirmation_mb16

python -m gepa_pilot.objective_audit \
  --result artifacts/gepa_observation/gepa_result.json
python -m gepa_pilot.objective_audit \
  --result artifacts/gepa_confirmation_mb8/gepa_result.json
python -m gepa_pilot.objective_audit \
  --result artifacts/gepa_confirmation_mb16/gepa_result.json
python -m gepa_pilot.compare_gepa_runs
```

Calibrate ADMM, run the smoke or full sweep on separate GPUs, and summarize:

```bash
CUDA_VISIBLE_DEVICES=4 python -m gepa_pilot.run_admm \
  --run-dir artifacts/admm_calibration --rho 0.01 --calibrate-only

CUDA_VISIBLE_DEVICES=4 python -m gepa_pilot.run_admm \
  --run-dir artifacts/admm_rho1 --rho 1
CUDA_VISIBLE_DEVICES=5 python -m gepa_pilot.run_admm \
  --run-dir artifacts/admm_rho10 --rho 10
CUDA_VISIBLE_DEVICES=6 python -m gepa_pilot.run_admm \
  --run-dir artifacts/admm_rho100 --rho 100

python -m gepa_pilot.admm_report
```

Do not pass `allow_test=True` during development or optimization. That escape
hatch exists only for a separately authorized final test evaluation.

## Artifact index

| Artifact | Contents |
|---|---|
| [`artifacts/sanity_report.json`](artifacts/sanity_report.json) | Full evaluator and gradient sanity gates |
| [`artifacts/baseline_results.json`](artifacts/baseline_results.json) | Seed/manual metrics, confusion matrices, and records |
| [`artifacts/gepa_comparison/comparison.md`](artifacts/gepa_comparison/comparison.md) | Human-readable comparison of the three GEPA runs |
| [`artifacts/gepa_comparison/comparison.csv`](artifacts/gepa_comparison/comparison.csv) | Machine-readable GEPA comparison table |
| [`artifacts/gepa_confirmation_mb8/gepa_summary.md`](artifacts/gepa_confirmation_mb8/gepa_summary.md) | Best GEPA run summary |
| [`artifacts/gepa_confirmation_mb8/objective_audit.md`](artifacts/gepa_confirmation_mb8/objective_audit.md) | Candidate objective-alignment audit |
| [`artifacts/gepa_confirmation_mb8/candidate_tree.html`](artifacts/gepa_confirmation_mb8/candidate_tree.html) | Interactive GEPA candidate lineage |
| [`artifacts/admm_comparison/admm_summary.md`](artifacts/admm_comparison/admm_summary.md) | Human-readable ADMM sweep summary |
| [`artifacts/admm_comparison/admm_summary.csv`](artifacts/admm_comparison/admm_summary.csv) | Machine-readable ADMM comparison table |
| [`artifacts/admm_comparison/admm_curves.png`](artifacts/admm_comparison/admm_curves.png) | Continuous/discrete loss, accuracy, residual, and token-change curves |
| `artifacts/admm_rho*/checkpoint_*.pt` | Exact ADMM checkpoints at validation iterations |

## Limitations and next experiments

- Results use one dataset subset, one frozen model, one data seed, and one GEPA
  run per minibatch setting. No confidence intervals are available yet.
- The observed validation set is now part of method development; it must not be
  described as an untouched test set.
- Both methods retain severe Sports and Business errors. Macro accuracy and the
  full confusion matrix should remain visible in follow-up work.
- GEPA and ADMM do not consume equivalent compute, so this pilot compares
  outcomes under documented procedures rather than a matched-compute frontier.
- BF16 label scores depend on batch shape. Reproductions must keep batch size,
  padding, and order fixed.
- ADMM's main bottleneck is projection/feasibility, not the ability to reduce the
  relaxed loss.

The next defensible sequence is:

1. Repeat the winning GEPA configuration and the strongest ADMM configuration
   across multiple seeds and report uncertainty.
2. Add a lower-\(\rho\) ADMM sweep and/or adaptive residual balancing, while
   preserving exact discrete validation at every checkpoint.
3. Investigate class-balanced reflection batches or macro-accuracy feedback to
   address the Sports/Business collapse.
4. Freeze all choices, select one final prompt per method from validation only,
   and then run exactly one official-test evaluation.
