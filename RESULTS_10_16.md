# Results log

New results after [`CLASSIFICATION_RESULTS.md`](CLASSIFICATION_RESULTS.md), whose last
update covers Phases 1–6 and the fluency analysis (2026-10-06). Entries are newest last. Each
entry says what was done, what it shows, and where the code and artifacts are.

**No new optimization seeds were added in this period.** The new numbers come from:
- a larger evaluation set (re-scoring existing runs);
- re-analysis of existing runs (seed counts, ADMM iterates, timing).

Phase 7, the first new runs, is still in progress.

Naming: "GEPA + … ADMM proposer" means ADMM replaces GEPA's reflection step; "standalone
ADMM" is the pilot only. "Fair S′" means children are accepted on an independent held-out
batch (Phase 6); "S used twice" is the earlier version that accepted on the batch that also
chose the candidate.

---

## 2026-10-15

### Larger report sets (2,000 examples per task)
- **Why:** the 256-example report half has a standard error of about 3 pp per run at
  ~35% accuracy, comparable to the whole seed-to-seed spread.
- **What:** a class-stratified report set per task, drawn from the **unused** part of each
  official training split. It never overlaps the train or validation subsets, and the official
  test splits stay locked.
  - AG News: 2,000 examples (500 per class).
  - Emotion: 2,004 examples (334 per class).
- **Code:** [`report_set.py`](src/gepa_pilot/report_set.py) (`prepare`). **Data:**
  `artifacts/data/report.json`, `artifacts/data_emotion/report.json`, each with a manifest.

### Re-scoring of every existing run on the 2,000-example sets (Stage 0)
- **What:** the prompt every finished run selected (on the 256-example selection half; selection
  unchanged) was scored on the large report set. That's 111 AG News runs (78 distinct prompts)
  and the emotion runs, plus the seed instruction.
- **Code:** `report_set.py rescore`. **Results:**
  [`artifacts/report2k/`](artifacts/report2k/).
- **Main numbers** (10 seeds):

  | Arm | AG News, 2k examples | Emotion, 2k examples |
  |---|---:|---:|
  | Seed instruction | 28.90 | 37.03 |
  | Reflective GEPA (12 proposals) | 32.95 ± 1.98 | 36.81 ± 0.41 |
  | Matched reflective (30 proposals, held-out) | 34.09 ± 1.21 | 36.79 ± 0.66 |
  | GEPA + top-p ADMM proposer | 35.59 ± 2.63 | 37.03 ± 0.07 |
  | GEPA + beam ADMM proposer (S used twice) | 36.98 ± 2.70 | 37.23 ± 0.46 |
  | **GEPA + beam ADMM proposer (fair S′)** | **35.02 ± 2.66** | **37.92 ± 0.99** |

  | Fair beam ADMM proposer minus… | AG News | Emotion |
  |---|---|---|
  | matched reflective | +0.93 (p = 0.33) | **+1.13 (p = 0.009)** |
  | reflective GEPA | +2.06 (p = 0.066) | **+1.11 (p = 0.007)** |

- **Shows:**
  - **Emotion:** the fair beam ADMM proposer beats both reflective baselines, now with
    p < 0.01 (it was p = 0.03 on 256 examples). The larger set halved that arm's spread.
    Reflective GEPA ends slightly *below* the seed instruction, so on this task reflection
    doesn't improve the prompt, and ADMM improves it modestly (about +0.9 pp over the seed).
  - **AG News:** still a statistical tie between fair ADMM and matched reflection. The larger
    set barely reduced the spread (3.7 → 2.7 pp for fair beam ADMM), so **most AG News
    variance is genuine run-to-run variation, not measurement noise**. Every method improves
    on the seed by 4–6 pp; the 256-example half had overstated the seed's own accuracy
    (30.9 vs. 28.9).

### New code, checked before use
1. **Multi-length ADMM proposer** (`--admm-length-offsets 0 -2 2`):
   - each proposal also refines a 2-token-shorter variant (lowest-saliency tokens removed)
     and a 2-token-longer variant (the LM's most likely token inserted at the
     highest-saliency positions);
   - all candidates, including the unrefined variants, are pooled, chosen on S, and
     judged against the parent on S′.

   Code: `ADMMProposer._length_variant`, `_refine_from` in
   [`admm_proposer.py`](src/gepa_pilot/admm_proposer.py).
2. **Fair reflect-then-polish:** the held-out judge now wraps the beam polish and compares
   the final child with **GEPA's parent** on the polish's reserved batch S′, the same rule as
   the reflective baselines (`--proposer reflect_polish --reflection-acceptance heldout`).
3. **Shared-GPU queue** ([`scripts/gpu_queue.py`](scripts/gpu_queue.py)):
   - launches on the allowed GPU with the most free memory;
   - caps our jobs per GPU and retries out-of-memory failures;
   - marks a run **failed** if GEPA's log shows proposal errors it swallowed (see below).

**Checks:**
- **Single-length regression:** in deterministic mode, the refactored proposer gives ADMM
  proposal logs and GEPA event streams **identical** to the previous code.
- **Fair-polish wiring:** the judge's batch equals the proposer's S′, never overlaps S or the
  minibatch, and compares against GEPA's parent.
- **Multi-length smoke test:** after one bug fix (below), both proposals ran.
  - Proposal 1: the shortened variant won and was accepted on S′ (accuracy 25.0 → 31.3%).
  - Proposal 2: the lengthened variant won on S but was correctly **rejected** on S′
    (25.0 → 21.9%).

**A bug, and a silent-failure risk.** The first multi-length smoke test crashed on every
proposal (an in-place edit of an inference-mode tensor). **GEPA catches proposal exceptions,
logs them to its own `run_log.txt` and continues**, so a run can finish "successfully" having
proposed nothing. After fixing the bug and hardening the queue:
- all 189 earlier run logs were scanned, and none had swallowed proposal errors;
- the only "did not propose" entries were normal skips, where the parent was already perfect
  on its minibatch.

### Phase 7 launched, then restarted
- **Batch:** fair reflect-then-polish and the multi-length beam ADMM proposer, 10 seeds ×
  2 tasks (40 runs), through the queue on GPUs 4–7 in `tmux`
  ([`scripts/run_phase7.sh`](scripts/run_phase7.sh)). At the end it re-scores the new runs on
  the 2,000-example sets automatically.
- **Incident:** at 23:48–23:49, two `git stash -u` operations in the repo moved untracked
  files, including in-progress run outputs and logs. Effects:
  - The AG News re-scoring had **finished** at 23:46; its results were restored from the
    stash.
  - The 8 Phase 7 runs then in flight had their outputs split between the stash and the
    working tree, so they were discarded and the batch restarted at 23:53.
  - In-progress run directories are now git-ignored, so `stash -u` and `clean -fd` can't
    touch them.

---

## 2026-10-16

### Where the time goes: ADMM, not GEPA
| Run | Optimization time | Inside ADMM proposals | GEPA's own evaluations and bookkeeping |
|---|---:|---:|---:|
| Phase 6 beam ADMM proposer, AG News | 272 s | 251 s (93%) | 20 s |
| Phase 6 beam ADMM proposer, emotion | 263 s | 244 s (93%) | 19 s |
| Phase 2 top-p ADMM proposer, AG News | 261 s | 240 s (92%) | 20 s |
| Phase 7 multi-length (in progress, shared GPUs) | — | ~290 s per proposal | — |
| Phase 7 fair reflect-then-polish (in progress) | — | ~120 s per proposal | — |

- **Within ADMM, the projection step dominates.** On idle GPUs a Euclidean proposal took
  about 6 s, against about 15–20 s for top-p or beam. Those projections decode the slot one
  token at a time (20 rounds × about 26 positions), so they're limited by per-step overhead.
  That makes the projection the main target for speed-ups, e.g. by batching beams and
  candidate scoring.
- Phase 7 runs are about 2–3× slower than earlier phases because GPUs 4–7 are shared with
  other users at 100% utilization.

### Small vs. large evaluation set, side by side
Generated by [`eval_size_comparison.py`](src/gepa_pilot/eval_size_comparison.py)
(`artifacts/eval_size/`). These are **the same runs and the same selected prompts**. Each run
chose its prompt on the 256-example selection half; only the evaluation of that prompt differs:
the 256-example report half against the ~2,000-example report set. Any difference between the
two columns is therefore pure **evaluation noise**, not optimization randomness.

![Evaluation-set size](artifacts/figures/fig8_eval_size.png)

*Figure 8. For each arm, its 10 runs scored on 256 examples (hollow circles, upper row) and the
same runs scored on ~2,000 examples (filled, lower row), with mean ± 95% CI. Dotted and dashed
lines: the seed instruction on each set.*

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


**What the side-by-side comparison shows:**
- **The level shifts with the evaluation set.** The 256-example half happened to be easier
  on AG News (seed 30.86 vs. 28.90) and harder on emotion (35.94 vs. 37.03), so every arm
  moves with it. Comparisons *between* arms are what matter, and the seed instruction is the
  reference point.
- **The larger set narrows results most where measurement noise dominated:**
  - **Emotion ADMM arms:** the spread falls by 50–73% (fair beam ADMM: sd 1.98 → 0.99; 95% CI
    ±1.42 → ±0.71). Their runs pick different prompts with small true differences, which 256
    examples couldn't resolve.
  - **Little change where runs mostly selected the same prompt:** emotion reflective GEPA,
    where 7 of 10 runs end on the seed instruction.
  - **Little change where runs genuinely differ:** AG News, −10% to −29%. The AG News top-p
    arm even widens (+18%), because measurement noise had been partly hiding real differences
    between runs.
- **Conclusions that hold, and one that doesn't:**
  - **Emotion, fair beam ADMM vs. matched reflection:** the effect is *smaller* on the large
    set (+1.64 → +1.13 pp) but *more certain* (p = 0.030 → 0.009). The 256-example half
    overstated its size, not its existence.
  - **Emotion, top-p ADMM vs. reflective GEPA:** significant on 256 examples (p = 0.016) but
    **not** on 2,004 (p = 0.134), a false positive from the small set.
  - **AG News, fair beam ADMM vs. matched reflection:** 0.00 → +0.93 pp, still not significant
    (p = 0.33). The large set doesn't rescue this comparison, because the AG News variance is
    real.
  - **AG News, top-p ADMM vs. reflective GEPA:** holds and strengthens (p = 0.041 → 0.021).
- **For future runs:** score on the large report set by default (Phase 7 already is). For
  AG News, more seeds, not more evaluation examples, is what would reduce uncertainty.

### How results change with the number of seeds
Generated by [`seed_count_analysis.py`](src/gepa_pilot/seed_count_analysis.py)
(`artifacts/seed_count/`). For each 10-seed arm, the first n seeds (the order in which seeds
were actually run) are summarized at n = 3, 5 and 10, on both report measures.

#### AG News

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

#### Emotion

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


**What it shows:**
- **Three seeds were not enough, especially on AG News.** The 95% interval at n = 3 is ±7–9
  pp on the 256-example half. The first three seeds happened to favor ADMM:
  - the fair beam ADMM proposer went from 38.93 (n = 3) to 35.70 (n = 10);
  - its lead over matched reflection went from +4.3 pp to 0.

  Reflective arms moved much less. Early decisions taken at n = 3 (for example, choosing
  beam λ = 0.3) were made on optimistic numbers.
- **Emotion is the reverse.** The ADMM advantage grows and sharpens with n, from
  +0.8 (p = 0.36) at n = 3 to +1.6 (p = 0.03) at n = 10 on the 256-example half, and
  p = 0.009 on 2k examples. Emotion has low variance, so 10 seeds suffice there.
- **The larger report set helps where noise is measurement noise.** On emotion it roughly
  halves the spread; on AG News it reduces it only slightly, because there the variance comes
  from the optimization runs themselves. Resolving about a 1 pp difference on AG News would
  need about 30–40 seeds per arm.

### ADMM: the best candidate is rarely from the last round
Generated by [`iterate_diagnostics.py`](src/gepa_pilot/iterate_diagnostics.py)
(`artifacts/iterate_diagnostics/summary.json`). Each GEPA + ADMM proposal runs 20 ADMM
rounds, scores every new exact-round-trip token sequence on the held-out batch S, and returns
the lowest-loss one if it beats the parent.

![Round of the returned candidate](artifacts/figures/fig7a_admm_returned_round.png)

*Figure 7a. The ADMM round each returned candidate came from (fair beam ADMM proposer, both
tasks). Returned candidates come from every part of the run (median round 12–13); round 20
accounts for only 7 of 98 (AG News) and 5 of 97 (emotion).*

![What the round-20 iterate is](artifacts/figures/fig7b_admm_round20_status.png)

*Figure 7b. What a "take the last round" rule would return. Most often it's the parent itself
(no change: 72% on AG News, 53% on emotion). Otherwise it's a revisit of an earlier candidate,
or a new candidate that is usually worse on S than the one actually returned. The unlabeled
AG News segment is 8%.*

![One proposal's trajectory](artifacts/figures/fig7c_admm_trajectory.png)

*Figure 7c. One AG News proposal, starting from the seed instruction. The iterate visits several
1–5 token edits and repeatedly falls back to the parent:*
- *"ignore **linguistic** details" (S-loss 9.67);*
- *"…the corresponding capital **label**" (12.28);*
- *"…label mapping **and** ignore…**label** letter" (9.46, returned);*
- *"**Pleasefully** determine…" (11.32);*
- *"…label mapping **below and give details**…" (9.49);*
- *a 5-token edit with "(标题 tag - one" (11.94);*
- *"main **subject**" (11.39).*

*In round 20 it is back at the parent (S-loss 11.76).*

| Arm | Proposals | Round 20 = parent | Returned from round 20 | Median round returned | Returns to parent per proposal | Round-20 candidate worse on S than the returned one |
|---|---:|---:|---:|---:|---:|---|
| AG News / beam (S twice) | 120 | 94 (78%) | 2 / 93 | 14 | 2.2 | 98% (n = 84), median +1.00 |
| AG News / beam (fair S') | 120 | 86 (72%) | 7 / 98 | 13 | 2.4 | 92% (n = 91), median +0.95 |
| AG News / top-p | 120 | 92 (77%) | 4 / 98 | 14.5 | 2.2 | 96% (n = 97), median +0.98 |
| Emotion / beam (S twice) | 120 | 64 (53%) | 13 / 89 | 12 | 3.6 | 82% (n = 73), median +0.25 |
| Emotion / beam (fair S') | 120 | 64 (53%) | 5 / 97 | 12 | 3.9 | 93% (n = 70), median +0.34 |
| Emotion / top-p | 120 | 99 (82%) | 7 / 66 | 15 | 1.5 | 89% (n = 65), median +0.34 |

- **The discrete iterate oscillates rather than converging.** It returns to the parent about
  1.5–4 times per proposal, and **in round 20 it is the parent in 53–82% of proposals**. A
  "take the last round" rule would mostly propose nothing.
- **The returned candidate comes from round 20 in only 2–15% of proposals.** Returned rounds
  spread across the whole run (median 12–15).
- **When the round-20 iterate is identifiable, it is worse on S than the returned candidate
  in 82–98% of cases** (median +0.25 to +1.0 nats).
- **Why:** when ADMM's dual variable u has accumulated enough pull, the projection jumps to a
  new token. The residual then reverses, u unwinds, and the projection often jumps back.
  Top-p and beam always keep the parent's token available, which makes "back to the parent"
  easy. The heuristic mixed-integer ADMM literature (Takapoui et al.) likewise keeps the best
  feasible iterate, not the last.
- **Implication for the multi-length / multi-start variants:** pooling candidates from several
  refinements makes sense; taking each refinement's last round would not.

### What each proposer changes: text vs. predictions
Generated by [`proposal_diagnostics.py`](src/gepa_pilot/proposal_diagnostics.py) and
[`proposal_examples.py`](src/gepa_pilot/proposal_examples.py), from saved runs only
(`artifacts/proposal_diagnostics/`).

#### Definitions
A **proposal** is one parent → child step in GEPA: the parent is the prompt GEPA chose to
improve; the child is what the proposer (reflection or ADMM) made from it. For every proposal,
both prompts were scored by the post-hoc audit on the same **512 validation examples**, so for
each example we know the label each prompt predicted (A–D or A–F) and whether it was correct.
Each of the 512 examples falls into exactly one category:

| Category | Parent's prediction | Child's prediction | Counted as a flip? |
|---|---|---|---|
| unchanged, correct | correct | the same label (correct) | no |
| unchanged, wrong | wrong | the same label (wrong) | no |
| **fix** | wrong | the correct label | yes |
| **break** | correct | a wrong label | yes |
| **wrong → different wrong** | wrong (e.g. World) | another wrong label (e.g. Sports) | yes |

- **Flips (predictions changed)** = the number of examples whose predicted label differs
  between parent and child, i.e. fixes + breaks + wrong → different wrong. It ignores whether
  the change helped.
- **Flip rate** = flips / 512: how much the child changes the model's behavior.
- **Net** = fixes − breaks = the child's accuracy change in examples. Divided by 512, it equals
  the accuracy change (wrong → different wrong flips don't affect accuracy).
- **Fixes as % of fixes + breaks** measures direction. 50% means the changes cancel out; above
  50% means they mostly help.
- **Text edited** = token-level edit distance between parent and child, divided by the longer
  prompt's token count.

Only proposals whose child differs from its parent are counted. Rejected children are included,
since the audit scored them too.

#### Example A: a typical reflection proposal (AG News)

- Parent: “Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.”
- Child: “Carefully analyze the article to identify the central topic, prioritize the main subject over incidental details, and select the corresponding capital letter from the label mapping.”
- Parent accuracy 30.47% → child 30.47% (+0.00 pp).

| Category (512 validation examples) | Count | Counted as a flip? |
|---|---:|---|
| unchanged, correct | 137 | no |
| unchanged, wrong | 286 | no |
| fix (wrong → right) | 19 | yes |
| break (right → wrong) | 19 | yes |
| wrong → different wrong | 51 | yes |
| **flips (predictions changed)** | **89** (17.4%) | |
| net = fixes − breaks | +0 examples = +0.00 pp | |

| Category | Article (first 110 characters) | Gold | Parent predicted | Child predicted |
|---|---|---|---|---|
| fix (wrong → right) | DVD/HD-DVD Combo Developed December 08, 2004 - Memory-Tech Corp. and Toshiba have announced that they have dev… | D (Sci/Tech) | A (World) | D (Sci/Tech) |
| fix (wrong → right) | Search companies vye for the desktop All the major Web portals seem to be rushing delivery on one product this… | D (Sci/Tech) | A (World) | D (Sci/Tech) |
| break (right → wrong) | Bhopal survivors still short on medical treatment and compo The failure of the Indian Government and an Americ… | A (World) | A (World) | D (Sci/Tech) |
| break (right → wrong) | Sony Takes on iPod with New Hard Disk Walkman Sony Corp (6758.T: Quote, Profile, Research) unveiled its first … | D (Sci/Tech) | D (Sci/Tech) | B (Sports) |
| wrong → different wrong | MSN Search Beta Not Ranking MSN Search Blog MSN Search Beta Not Ranking MSN Search Blog\\Ok, I lied in the tit… | D (Sci/Tech) | A (World) | B (Sports) |
| wrong → different wrong | Microsoft, EU see pre-Christmas antitrust ruling LUXEMBOURG -- Microsoft Corp., which was ruled an illegal mon… | C (Business) | A (World) | D (Sci/Tech) |

#### Example B: a typical beam ADMM proposal (AG News)

- Parent: “Carefully fill in the sentence completion with from the label mapping. Suppose other details, and output only the corresponding capital letter.”
- Child: “Carefully fill in the sentence completion with from the label mapping. Suppose other details, and Output ONLY the corresponding capital letter.”
- Parent accuracy 35.35% → child 37.50% (+2.15 pp).

| Category (512 validation examples) | Count | Counted as a flip? |
|---|---:|---|
| unchanged, correct | 168 | no |
| unchanged, wrong | 285 | no |
| fix (wrong → right) | 24 | yes |
| break (right → wrong) | 13 | yes |
| wrong → different wrong | 22 | yes |
| **flips (predictions changed)** | **59** (11.5%) | |
| net = fixes − breaks | +11 examples = +2.15 pp | |

| Category | Article (first 110 characters) | Gold | Parent predicted | Child predicted |
|---|---|---|---|---|
| fix (wrong → right) | Sistani Urges Supporters to Wait at Najaf Gates Iraq #39;s top Shi #39;ite cleric Grand Ayatollah Ali al-Sista… | A (World) | D (Sci/Tech) | A (World) |
| fix (wrong → right) | The long goodbye It hit Julie Foudy not when she saw the 90,125 in the Rose Bowl for the final of the 1999 Wom… | B (Sports) | D (Sci/Tech) | B (Sports) |
| break (right → wrong) | MSN Search Beta Not Ranking MSN Search Blog MSN Search Beta Not Ranking MSN Search Blog\\Ok, I lied in the tit… | D (Sci/Tech) | D (Sci/Tech) | B (Sports) |
| break (right → wrong) | Sony Takes on iPod with New Hard Disk Walkman Sony Corp (6758.T: Quote, Profile, Research) unveiled its first … | D (Sci/Tech) | D (Sci/Tech) | B (Sports) |
| wrong → different wrong | Merck Adopts Severance Safeguards for Executives Drugmaker Merck  amp; Co., under financial siege since pullin… | C (Business) | A (World) | D (Sci/Tech) |
| wrong → different wrong | IBM, Others Push for Greater U.S. Innovation Nation needs government help to maintain competitive edge, group … | D (Sci/Tech) | B (Sports) | A (World) |

#### Results across all proposals

![Fixes vs breaks](artifacts/figures/fig9b_fixes_vs_breaks.png)

*Figure 9b. Mean number of validation examples a child fixes, breaks, or moves from one wrong
label to another, relative to its parent, per proposal. "net" = fixes − breaks.*

![Text vs behavior](artifacts/figures/fig9_text_vs_behavior.png)

*Figure 9. AG News, one point per proposal: text edited (x) vs. flip rate (y); ★ = median.
Reflection rewrites about 40% of the tokens; ADMM proposers edit about 4% (1–3 tokens), yet both
change a similar share of predictions.*

| Task / proposer | Proposals counted | Text edited (median % of tokens) | Flip rate (median) | Fixes (mean per proposal) | Breaks (mean per proposal) | Wrong → different wrong (mean per proposal) | Fixes as % of fixes + breaks (pooled) | Accuracy change (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AG News: reflection | 405 | 40% | 17.4% | 26.3 | 24.6 | 45.9 | 52% | +0.00 pp |
| AG News: Euclidean ADMM | 97 | 4% | 12.5% | 25.5 | 18.2 | 37.8 | 58% | +1.17 pp |
| AG News: top-p ADMM | 98 | 4% | 14.8% | 26.7 | 19.3 | 39.2 | 58% | +1.76 pp |
| AG News: beam ADMM | 191 | 4% | 11.7% | 25.7 | 17.0 | 35.1 | 60% | +1.95 pp |
| AG News: reflect-then-polish | 36 | 38% | 26.6% | 37.8 | 33.7 | 63.6 | 53% | +0.78 pp |
| Emotion: reflection | 412 | 48% | 18.9% | 30.3 | 33.3 | 30.5 | 48% | -0.39 pp |
| Emotion: top-p ADMM | 66 | 4% | 14.2% | 26.6 | 32.3 | 26.7 | 45% | -0.52 pp |
| Emotion: beam ADMM | 186 | 4% | 13.4% | 23.5 | 25.8 | 24.6 | 48% | -0.26 pp |

**How to read the numbers in this table:**
- **Fixes, breaks and wrong → different wrong are averages.** Each single proposal has whole
  numbers (e.g. 19 fixes and 19 breaks in Example A); the table shows their **mean over all
  proposals** in the group, so it can be fractional (26.3 fixes = 26.3 examples on average).
- **Flip rate and accuracy change are medians,** which are less sensitive to a few extreme
  proposals.
- **"Fixes as % of fixes + breaks" is pooled:** all fixes in the group divided by all fixes and
  breaks.

**Why the number of proposals differs between rows.** Each row pools every run of that proposer
in the comparisons, so groups contain different numbers of runs. Proposals where the child is
identical to its parent are excluded, since there is nothing to compare. That happens when an
ADMM proposal changes no token or finds no candidate better than the parent on S, and the
proposer returns the parent unchanged:

| Task / proposer | Runs pooled | Proposals made | Child = parent (excluded) | Counted |
|---|---|---:|---:|---:|
| AG News: reflection | 10 × 12-proposal + 10 × 30-proposal (held-out judge) | 420 | 15 | 405 |
| AG News: Euclidean ADMM | 6 minibatch + 6 held-out (Phase 1), 12 proposals each | 144 | 47 | 97 |
| AG News: top-p ADMM | 10 runs × 12 | 120 | 22 | 98 |
| AG News: beam ADMM | 10 (S used twice) + 10 (fair S′), 12 each | 240 | 49 | 191 |
| AG News: reflect-then-polish | 3 runs × 12 | 36 | 0 | 36 |
| Emotion: reflection | 10 × 12 + 10 × 30 (held-out judge) | 420 | 8 | 412 |
| Emotion: top-p ADMM | 10 runs × 12 | 120 | 54 | 66 |
| Emotion: beam ADMM | 10 (S used twice) + 10 (fair S′), 12 each | 240 | 54 | 186 |

Reflection almost always writes something new; ADMM returns the parent unchanged in 18–45% of
proposals (most often top-p on emotion, where few edits beat the parent).

- **Both kinds of proposal change many predictions.** On AG News, reflection changes about 17%
  of the 512 predictions by rewriting about 40% of the instruction. ADMM changes 12–15% by
  editing 1–3 tokens (about 4%). Per edited token, ADMM's edits move about 10× more
  predictions.
- **The difference is direction, not size.** Reflection's fixes and breaks nearly cancel: on
  AG News an average of 26.3 fixes against 24.6 breaks, net +1.7 examples, median accuracy
  change +0.00 pp (Example A: 19 fixes, 19 breaks, net 0). The ADMM proposers fix about as
  many examples (25.5–26.7) but break fewer (17.0–19.3), net +7.3 to +8.7 (Example B: 24 fixes,
  13 breaks, net +11 = +2.15 pp).
- **Almost half of all flips are wrong → different wrong** (about 35–46 per proposal on AG
  News). These change behavior without changing accuracy, e.g. a Business article moving from
  "World" to "Sci/Tech".
- **Correction to an earlier explanation.** `CLASSIFICATION_RESULTS.md` §5.10 had said
  reflection's paraphrases "don't move the model's predictions". They do move them, but in
  no consistent direction. Reflection rewrites the whole instruction after reading 8 errors,
  which shifts many predictions both ways; ADMM's gradient steps aim at the loss.
- **Emotion:** every proposer's changes are slightly break-heavy (net −2.3 to −5.7 examples per
  proposal), consistent with the instruction having little leverage on this task. Gains there
  come from GEPA keeping the rare good child and selecting it at the end, not from typical
  proposals helping.

### How reliable each acceptance rule is
Same data. A child counts as **actually better** if its accuracy on the 512 validation examples
is higher than its parent's, i.e. net > 0 in the terms above. No acceptance rule ever sees
validation data (they all use training examples), so this is an independent check. A useful rule
accepts mostly better children and rejects mostly worse ones.

![Acceptance reliability](artifacts/figures/fig10_acceptance_reliability.png)

*Figure 10. Share of accepted (dark) and rejected (light) children that were actually better.
The larger the gap between the two bars, the better the rule separates them.*

| Task / proposer / acceptance rule | Accepted | …actually better | Rejected | …actually better |
|---|---:|---:|---:|---:|
| AG News: reflection / minibatch (GEPA default) | 25 | 72% | 90 | 48% |
| AG News: reflection / held-out judge | 125 | 72% | 165 | 28% |
| AG News: Euclidean ADMM / minibatch (GEPA default) | 9 | 67% | 40 | 70% |
| AG News: Euclidean ADMM / held-out on S (S used twice) | 41 | 71% | 7 | 43% |
| AG News: top-p ADMM / held-out on S (S used twice) | 77 | 81% | 21 | 57% |
| AG News: beam ADMM / held-out on S (S used twice) | 68 | 87% | 25 | 60% |
| AG News: beam ADMM / held-out on independent S' | 79 | 82% | 19 | 53% |
| AG News: reflect-then-polish / minibatch (GEPA default) | 10 | 60% | 26 | 58% |
| Emotion: reflection / minibatch (GEPA default) | 20 | 30% | 100 | 32% |
| Emotion: reflection / held-out judge | 61 | 16% | 231 | 28% |
| Emotion: top-p ADMM / held-out on S (S used twice) | 44 | 27% | 22 | 27% |
| Emotion: beam ADMM / held-out on S (S used twice) | 62 | 37% | 27 | 15% |
| Emotion: beam ADMM / held-out on independent S' | 45 | 33% | 52 | 38% |

#### 2×2 tables: acceptance decision vs. whether the child was actually better (AG News)

"Better" = higher accuracy than the parent on the 512 validation examples.

| Proposer / acceptance rule | Accepted & better | Accepted & not better | Rejected & better | Rejected & not better |
|---|---:|---:|---:|---:|
| reflection / minibatch (GEPA default) | 18 | 7 | 43 | 47 |
| reflection / held-out judge | 90 | 35 | 47 | 118 |
| Euclidean ADMM / minibatch (GEPA default) | 6 | 3 | 28 | 12 |
| Euclidean ADMM / held-out on S (S used twice) | 29 | 12 | 3 | 4 |
| beam ADMM / held-out on S (S used twice) | 59 | 9 | 15 | 10 |
| beam ADMM / held-out on independent S' | 65 | 14 | 10 | 9 |

#### Rejected by GEPA's minibatch rule, yet better on validation (AG News, largest gains)

**Reflection** (12-proposal runs):

| Minibatch score: parent → child (out of 8) | Validation accuracy change (512 ex.) | Child prompt |
|---|---:|---|
| 0 → 0 (tie: rejected) | +4.88 pp | Carefully analyze the article to identify the central topic, focusing on the primary subject and context, and select the corresponding capital letter from the label mapping. |
| 3 → 3 (tie: rejected) | +4.30 pp | Carefully identify the article's central topic by focusing on the primary subject and context, ignoring incidental details, and select the corresponding capital letter from the label mapping. |
| 3 → 3 (tie: rejected) | +4.30 pp | Carefully identify the article's central theme from the label mapping, focusing on the primary subject and ignoring peripheral details, and output the corresponding capital letter. |

**Euclidean ADMM proposer** (Phase 1):

| Minibatch score: parent → child (out of 8) | Validation accuracy change (512 ex.) | Child prompt |
|---|---:|---|
| 2 → 2 (tie: rejected) | +10.55 pp | Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding资本 letter. |
| 2 → 2 (tie: rejected) | +5.47 pp | Carefully determine the article's main topic from the。 ⏎  mapping, ignore incidental details, analsex output only the corresponding capital letter. |
| 3 → 3 (tie: rejected) | +5.08 pp | Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding.DropDownItems alphabetical. |


**What these show:**
- **GEPA's minibatch rule can't see small edits.** For the Euclidean ADMM proposer it rejected
  28 children that were actually better, against 12 that weren't: rejected children were *more*
  often better (70%) than accepted ones (67%). The largest missed gains tied with their parent
  on the 8 minibatch examples (e.g. 2 → 2 correct) but were **+10.55 pp** and +5.47 pp better
  on validation. They are garbage-token prompts, but real improvements on the model.
- **The minibatch rule also discards good reflection rewrites.** It rejected 43 reflection
  children that were actually better (48% of its rejections), several by +4 to +5 pp, all on
  ties such as 3 → 3. The held-out judge separates much better: 90 accepted and better
  against 35 accepted and not, and 47 rejected and better against 118 rejected and not.
- **For ADMM on AG News, most children are better whatever the rule.** ADMM only proposes
  children that already beat their parent on S. Held-out acceptance still separates well
  (accepted 81–87% better, rejected 53–60%). On this measure the independent S′ (65 / 14 /
  10 / 9) looks about as good as reusing S (59 / 9 / 15 / 10); the double use mainly showed up
  in final accuracy (Phase 6).
- **On emotion, no rule separates reliably** (accepted 16–37% better, rejected 15–38%), and the
  reason is measurement, not the rules. Emotion children change accuracy by only a few
  examples. For a child that flips about 13% of examples, the net change has noise of roughly
  ±1.6 pp, more than most emotion children move. Treat the emotion rows as uninformative.

**Caveat for both tables:** "actually better on 512 validation examples" is itself noisy; small
differences (under about 2 pp) aren't resolved. The AG News contrasts above are large enough to
be robust; the emotion ones are not.

### Status of Phase 7 (in progress)
Fair reflect-then-polish and the multi-length beam ADMM proposer: 40 runs, restarted at
2026-10-15 23:53, expected to finish in the early morning. Results will be added here; check
`artifacts/phase7/STATUS`.
