# Results log (2026-10-15 – 2026-10-16)

This log follows [`CLASSIFICATION_RESULTS.md`](CLASSIFICATION_RESULTS.md), which covers Phases 1–6.
It is organized by question. Each section gives the evidence (plots and short tables), then the
takeaway.

- **Hypothesis labels:** anything not directly measured is marked **Hypothesis**.
- **Detailed tables:** linked from each section; every number comes from the analysis scripts
  named there.

**Setup shared by all runs:**
- **Model and tasks:** Qwen3-1.7B on AG News (4 classes) and emotion (6 classes).
- **GEPA:** selection/report split of validation (256 + 256), minibatch of 8, 12 proposals per
  run unless noted.
- **Evaluation:** the prompt each run selects is scored on a 2,000-example report set (see Q4).
- **Statistics:** 10 seeds per arm, Welch's t-tests. Many comparisons are made at n = 10, so a
  single p ≈ 0.03–0.05 is suggestive, not conclusive.

---

## Arms

| Label | What it is | Source |
|---|---|---|
| **A** | Reflective GEPA: reflection proposes, GEPA's minibatch rule accepts | Phase 2 / 4 |
| **A′** | Reflection, accepted by the S′ judge (AG News only) | Phase 3 |
| **B** | Matched reflective: reflection, S′ judge, **30** proposals (≈ ADMM's compute) | Phase 3 / 4 |
| **C** | **Fixed-length beam ADMM proposer**, independent S′ acceptance — the reference arm | Phase 6 |
| **D** | Multi-length ADMM, insertion = the LM's most likely token | Phase 7 |
| **E** | Reflect-then-polish: reflection drafts, beam ADMM polishes, S′ judge | Phase 7 |
| **F** | Multi-length ADMM, insertion chosen by the task-loss gradient | Phase 10 |
| **G** | Gated variable-length ADMM | Phase 8 |
| **H** | C with **36** GEPA proposals (3× compute spent on more proposals) | Phase 9 |
| **I** | Multi-start: **3** ADMM runs per proposal from the same parent, pooled (3× compute) | Phase 9 |

**Terms used throughout:**
- **S:** 32 training examples that choose the best candidate inside one ADMM proposal.
- **S′:** 32 *other* training examples that decide acceptance (the "fair" rule).
- **Minibatch rule:** GEPA's default. Accept the child if it scores strictly higher than the
  parent on 8 examples.
- **Validation net:** fixes − breaks of a child against its parent, counted on 512 validation
  examples. It's a diagnostic only; nothing is selected on it.

All ADMM arms use beam projection (fluency weight λ = 0.3), 20 ADMM rounds, and selection on S.
Code for arms C–I: [`admm_proposer.py`](src/gepa_pilot/admm_proposer.py),
[`gated_admm.py`](src/gepa_pilot/gated_admm.py). Analysis:
[`phase_7_10_analysis.py`](src/gepa_pilot/phase_7_10_analysis.py) → tables in
[`artifacts/phase_7_10/tables.md`](artifacts/phase_7_10/tables.md).

---

## Summary

- **Q1, parallel ADMM runs per proposal (I):** no clear benefit.
  - AG News +1.1 pp over C (p = 0.30), the same as spending that compute on 36 proposals (H,
    +1.3); emotion +0.1.
  - Its children are no better; it mostly turns "no change" proposals into real ones.
- **Q2, variable length:** it depends on the insertion rule and on the task.
  - Gradient-chosen insertion (F) is the best AG News arm: +3.2 pp over C (p = 0.013), +2.1 over
    equal-compute multi-start (p = 0.056).
  - LM insertion (D) doesn't help. Gated (G) is +1.6 at C's compute (p = 0.25).
  - On emotion, every insertion method falls back toward reflection.
- **Q3, reflect-then-polish works once judged fairly:** on AG News, +2.4 pp over reflection
  with the same judge (p = 0.018). The earlier "no effect" came from the acceptance rule: GEPA's
  minibatch rule rejected polished children essentially at random (20 of 26 rejections were
  exact ties). It doesn't help on emotion.
- **Q4, a 4×-larger evaluation set:**
  - halves the run-to-run spread on emotion, where noise was measurement noise;
  - barely changes AG News, where runs genuinely differ;
  - removed one false positive and shifted absolute levels.
- **Q5, seeds:** at 10 seeds, the smallest difference detectable with 80% power is about
  3 pp on AG News and about 1 pp on emotion.
  - The only open AG News question worth more seeds: is F's +2 pp over equal compute real?
    That needs about 20 seeds per arm.
- **Q6, ADMM's best candidate is rarely from the last round,** so the proposer keeps the best
  candidate it has seen.
- **Q7, flips:**
  - ADMM's small edits break fewer examples than reflection's rewrites.
  - Final accuracy follows the best accepted children, not the average child, so selection and
    acceptance matter as much as proposal quality.

---

## Q1. Do parallel ADMM runs per proposal help?

**How the arms differ** (all share GEPA's loop; only the proposer call differs):

```
C  (fixed-length beam ADMM):
    candidates = ADMM(parent, batch_order = 0)                 # 20 rounds; every new
                                                               # round-trip-exact iterate is a candidate
    child = argmin_loss_on_S(candidates)  if it beats the parent on S, else parent
    accept child  iff  loss_S′(child) < loss_S′(parent)  and  acc_S′(child) ≥ acc_S′(parent)

I  (multi-start, --admm-starts 3):
    candidates = ADMM(parent, order 0) ∪ ADMM(parent, order 1) ∪ ADMM(parent, order 2)
    # starts differ only in their training-batch order; same parent, same S and S′
    child = argmin_loss_on_S(candidates) ...                   # then exactly as C

H  (more GEPA proposals instead):
    exactly C, but GEPA runs 36 proposals instead of 12        # same total ADMM compute as I
```

![Q1](artifacts/figures/fig15_q1_multistart.png)

*Figure 15. Fixed-length ADMM with 1 start (C), 3 starts (I), or 36 proposals (H), plus matched
reflection (B) for reference. 2k report sets.*

**Per proposal** (all proposals of each arm; validation net vs. parent):

| Task | Arm | Proposals that changed the prompt | Mean net, changed children | Accepted (mean net) |
|---|---|---:|---:|---:|
| AG News | C | 82% | +7.9 | 79 (+9.3) |
| AG News | I | **94%** | +6.8 | 81 (+9.4) |
| AG News | H | 67% | +5.5 | 156 (+8.3) |
| Emotion | C | 81% | −2.0 | 45 (−0.8) |
| Emotion | I | **99%** | −1.7 | 54 (−1.1) |
| Emotion | H | 79% | −3.4 | 140 (−2.3) |

- **Final accuracy:**
  - AG News: I +1.13 (p = 0.30) and H +1.30 (p = 0.22) over C;
  - emotion: I +0.12, H −0.20; none significant.
- **What 3 starts change:** nearly every proposal now finds a candidate that beats the parent
  on S (94–99% vs. 81–82%), but the children aren't better. Accepted children are as good as
  C's (+9.4 vs. +9.3 on AG News).
- **H gets its gain differently:** about twice as many accepted children (156 vs. 79), each
  slightly weaker.
- **Hypothesis:** the gains for I and H, where present, come from more accepted steps stacking
  along a lineage, not from better individual edits.
- **Hypothesis (overfitting to S):** picking the best of more candidates on the same 32
  examples favors lucky candidates.
  - Weak support: in C-type proposals, a larger candidate pool goes with a slightly worse chosen
    child on AG News (Spearman ρ = −0.14, p = 0.049; +9.6 vs. +7.4).
  - Not seen on emotion (ρ = +0.04).
  - This is correlational only.
- **Takeaway:** parallel ADMM runs aren't worth 3× compute on their own. They're the fair
  control for the multi-length arms (Q2), which also pool 3 refinements.

---

## Q2. Comparing variable-length approaches

### What each approach does
All three replace C's single fixed-length refinement. Selection on S and acceptance on S′ are
unchanged.

```
D  multi-length, LM insertion  (--admm-length-offsets 0 -2 2)
    saliency_i = ‖∂ task_loss / ∂ e_i‖                         # per slot token, on one training batch
    short = parent with its 2 lowest-saliency tokens deleted
    long  = parent with 2 tokens inserted, each before a high-saliency position:
                token = argmax_w  P_LM(w | prefix, tokens so far)        # the LM's single top token
    candidates = {short, long}                                 # the raw variants are candidates too
               ∪ ADMM(parent) ∪ ADMM(short) ∪ ADMM(long)       # 3 refinements, 20 rounds each
    child = argmin_loss_on_S(candidates)                       # then S′ acceptance, as C

F  multi-length, gradient insertion  (… --admm-insertion gradient)
    as D, except each inserted token (one at a time, at the best of the 6 most salient gaps):
        for each gap g:
            place the LM's top token at g; take the gradient ∇ of the task loss at that slot
            rank the LM's 64 most likely tokens w by the linearized change (E(w) − e_g) · ∇
            keep the 8 best
        insert the (gap, token) with the lowest exact training loss among all ≤ 6 × 9 trials

G  gated variable length  (--admm-gated)
    slots = parent tokens, plus 4 spare slots before the 4 most salient positions
    each slot i: embedding e_i, keep-gate g_i = sigmoid(a_i)
        parent slots start open (g ≈ 0.98); spare slots start almost closed (g ≈ 0.12)
    scoring: attention logit to slot i += log g_i; positions advance by g_i
        → g = 0 is exactly deletion (checked to 1e-4 logits in float32)
    each ADMM round:
        E-step  (Adam on e and a):  CE(e, g) + ρ/2 Σ k_i ‖e_i − E(z_i) + u_i‖² + ρ_g/2 Σ (g_i − k_i + v_i)²
        Z-step  (beam search, left to right): each slot is
                  blank:   cost γ·t_i²
                  token w: cost  [‖s_i − E(w)‖² − min_w′ ‖s_i − E(w′)‖²]  +  γ(1 − t_i)²  +  λ·NLL(w | prefix)
                  (s_i = e_i + u_i,  t_i = g_i + v_i,  γ = 10;  k_i = 1 if slot i holds a token)
        duals:  u_i += e_i − E(z_i) for filled slots;   v_i += g_i − k_i
    every new token sequence is a candidate  →  argmin on S  →  S′ acceptance, as C
```

### Results
![Q2](artifacts/figures/fig16_q2_variable_length.png)

*Figure 16. C, multi-start I (same compute as D and F), and the three variable-length arms.
2k report sets.*

![Compute vs accuracy](artifacts/figures/fig13_compute_vs_accuracy.png)

*Figure 13. Mean 2k accuracy ± 95% CI against ADMM forward passes per run. Forward passes are a
compute proxy, not FLOPs; G's beam recomputes the whole prefix, so its passes are longer.*

- **AG News:**
  - **F is the only arm significantly above C:** +3.24 pp (p = 0.013; higher than C on 8 of 10
    seeds).
  - **At equal compute,** F beats I by +2.11 (p = 0.056) and H by +1.94 (p = 0.071). D is −0.56
    vs. I, so **the insertion rule makes the difference**.
  - **G:** +1.60 (p = 0.25), at about C's compute (7.9k vs. 7.0k forward passes); the others use
    about 21–24k.
- **Emotion:**
  - D (−0.88, p = 0.10) and F (−0.83, p = 0.045) fall toward the reflection level;
  - G ties C (−0.06).

### Which length variant wins inside D and F
| Arm | Task | Variant | Times chosen | Accepted on S′ | Mean validation net | Better than parent |
|---|---|---:|---:|---:|---:|---:|
| D | AG News | 2 shorter | 71 | 47 | **+13.8** | 68% |
| D | AG News | same length | 27 | 12 | −5.1 | 30% |
| D | AG News | 2 longer | 16 | 7 | −4.5 | 31% |
| F | AG News | 2 shorter | 11 | 7 | +11.8 | 73% |
| F | AG News | 2 longer | **101** | 65 | **+9.2** | 74% |
| D | Emotion | 2 longer | 22 | 9 | −7.8 | 23% |
| F | Emotion | 2 longer | 88 | 47 | **−14.0** | 31% |

(Full table, including the remaining rows: [`tables.md`](artifacts/phase_7_10/tables.md).)

- **D on AG News, deletion does the work:**
  - the recurring edit is "label mapping, ~~ignore incidental details,~~ **and provide
    details**…";
  - two seeds ended on the identical final prompt.
- **F on AG News, insertion takes over:**
  - all 10 seeds end on different prompts, e.g. "…output only the **numeric** corresponding
    **target** capital letter";
  - the LM-top-token rule mostly produced repetitions ("label capital **label** letter").
- **On emotion, the longer variant hurts under both rules,** yet S′ accepts about half of these
  children.
  - **Hypothesis:** on emotion, a single child's effect (a few examples) is below what 32
    acceptance examples can resolve (see Q3), so harmful insertions get through.

### Proposal bench: one child per method from the same parents
![Proposal bench](artifacts/figures/fig14_proposal_bench.png)

*Figure 14. Six parent prompts per task. Every method gets the same GEPA minibatch, S, S′ and
training batches; each child is checked on the 512 validation examples. Right of each row: mean
net | seconds per proposal | change in NLL per token (fluency; lower is better). Code:
[`length_bench.py`](src/gepa_pilot/length_bench.py).*

- **Fluency:** gated keeps it (ΔNLL +0.11 on AG News, −0.11 on emotion). Multi-length worsens
  it (+0.24 to +0.78); gradient insertion is the least fluent.
- **Cost:** gated is about 2× cheaper per proposal than D/F (114 s vs. 203–263 s).
- **Limit:** six parents can't rank the methods; single proposals swing by ±20 examples.

**Takeaway:**
- On AG News, gradient-chosen insertion is the most promising variable-length method; whether
  its edge over equal compute is real needs about 20 seeds (Q5).
- Gated is the cheapest and most fluent, with a smaller, uncertain gain.
- On emotion, variable length doesn't help with the current acceptance rule.

---

## Q3. Does reflect-then-polish work, and why didn't it earlier?

```
E  reflect-then-polish (fair)
    draft = reflection_LM(parent, minibatch mistakes)          # GEPA's normal reflective rewrite
    polished = fixed-length beam ADMM(draft) → best on S, if it beats the draft on S, else draft
    accept polished  iff  it beats GEPA's parent on S′         # same judge as A′ and B

old reflect-then-polish (Phase 2, 3 seeds)
    same draft and polish, but GEPA's minibatch rule decided acceptance (8 examples)
```

![Q3](artifacts/figures/fig17_q3_polish.png)

*Figure 17. Reflection with the minibatch rule (A), reflection with the S′ judge (A′ at 12
proposals, B at 30), reflect-then-polish (E), and fixed-length ADMM (C). 2k report sets.*

- **On AG News, the polish works:**
  - E vs. A′ (same judge, same 12 proposals): **+2.44 pp, p = 0.018**;
  - E vs. matched reflection (B, 30 proposals): +2.09 pp, p = 0.030.
- **Not on emotion:** E is 37.00, the same as reflection, and below C (−0.92, p = 0.021).

### Why it didn't work earlier: the acceptance rule
| Task | Proposer | Rule | Accepted | …better on validation (mean net) | Rejected | …better on validation (mean net) | Rejections that were ties |
|---|---|---|---:|---:|---:|---:|---:|
| AG News | reflect-then-polish (old) | minibatch, 8 examples | 10 | 60% (+4.1) | 26 | **58% (+4.1)** | **20 / 26** |
| AG News | reflect-then-polish (E) | S′, 32 examples | 63 | 68% (+10.0) | 54 | 43% (−5.6) | — |
| AG News | reflection (A) | minibatch, 8 examples | 25 | 72% (+10.6) | 90 | 48% (+2.2) | 71 / 95 |
| AG News | reflection (A′) | S′, 32 examples | 55 | 73% (+9.3) | 65 | 25% (−5.4) | — |
| Emotion | reflect-then-polish (E) | S′, 32 examples | 43 | 21% (−5.7) | 77 | 29% (−5.9) | — |

- **The minibatch rule rejected polished children at random:**
  - rejected children were as good as accepted ones (58% vs. 60% better, both +4.1 on
    average);
  - it accepted only 10 of 36;
  - 20 of the 26 rejections were exact ties on the 8 examples.
- **Why ties:** the polish changes a median of **1 token** of the draft. On 8 examples, a
  1-token change usually leaves every prediction unchanged, so parent and child tie and the
  strict-improvement rule rejects.
- **The S′ judge separates:** 68% of accepted vs. 43% of rejected children are better, with mean
  net +10.0 vs. −5.6.
- **The old comparison had only 3 seeds** (33.35 ± 0.35), so it couldn't have detected the
  effect anyway.
- **The polish is often a no-op:** it leaves the draft unchanged in 47% of AG News proposals
  (no token change, or no candidate beats the draft on S).
- **Hypothesis for why E beats A′:** E's accepted children are slightly more numerous (63 vs.
  55) and slightly better (+10.0 vs. +9.3). Small, directed 1-token gains stacked over 12
  proposals could add up to the 2.4 pp difference. Not tested directly.

### Acceptance rules in general
![Acceptance reliability](artifacts/figures/fig10_acceptance_reliability.png)

*Figure 10. Share of accepted (dark) and rejected (light) children that were actually better on
validation, per proposer and acceptance rule. A bigger gap means better separation.*

- **The minibatch rule can't see small edits:**
  - Euclidean ADMM's rejected children were *more* often better (70%) than its accepted ones
    (67%);
  - it also rejected 43 reflection children that were better, mostly on ties.
- **The S′ judge separates well on AG News** (reflection: 73% vs. 25% better).
- **On emotion, no rule separates** (accepted and rejected children are 16–38% better either
  way).
  - **Hypothesis:** most emotion children move accuracy by fewer examples than the measurement
    noise. For a child that flips about 13% of predictions, the net has noise of about ±1.6 pp
    on 512 examples, and more on 32.
  - This would also explain why harmful insertions pass S′ on emotion (Q2).
- **Takeaway:** acceptance is a bottleneck. It was the cause of the earlier polish failure, and
  it's the likely limit on emotion.
- **Next steps to consider:** a larger S′ on emotion, or accepting on a confidence bound rather
  than a point estimate.

---

## Q4. Does a larger evaluation set reduce the variance, and what else changes?

- **The 2k report sets:** class-stratified, drawn from the unused part of each official training
  split, no overlap with train or validation; the test splits stay locked.
  - AG News: 2,000 examples. Emotion: 2,004.
  - Code: [`report_set.py`](src/gepa_pilot/report_set.py).
- **The comparison:** the same runs and the same selected prompts are scored on the 256-example
  report half and on the 2k set, so any difference is evaluation noise. Code:
  [`eval_size_comparison.py`](src/gepa_pilot/eval_size_comparison.py).

![Evaluation-set size](artifacts/figures/fig8_eval_size.png)

*Figure 8. Each arm's 10 runs on 256 examples (hollow) and ~2,000 (filled), with mean ± 95% CI.
Dotted and dashed lines: the seed instruction on each set.*

![Comparisons on both sets](artifacts/figures/fig8b_eval_size_differences.png)

*Figure 8b. Key differences ± 95% CI (Welch) on each set; bold = p < 0.05.*

- **Variance drops only where measurement noise dominated:**
  - emotion ADMM arms: run-to-run sd −50% to −73% (C: 1.98 → 0.99);
  - AG News arms: −10% to −29%, and top-p even +18%.
  - **Hypothesis:** AG News runs genuinely end on prompts of different quality, so more seeds
    help there and more evaluation examples don't.
- **Other effects:**
  - **Absolute levels shift:** the 256 half was easier on AG News (seed instruction 30.9 vs.
    28.9) and harder on emotion (35.9 vs. 37.0). Compare arms, not levels.
  - **Effects can shrink:** emotion C vs. B went from +1.64 to +1.13 pp, while the p-value
    improved (0.030 → 0.009).
  - **One false positive removed:** emotion top-p vs. reflective GEPA went from p = 0.016 to
    p = 0.134.
- **Takeaway:** all results here use the 2k sets.

---

## Q5. Number of seeds: a diagnostic for whether to run more

![Seed count](artifacts/figures/fig11_seed_count.png)

*Figure 11. Left: arm mean ± 95% CI using the first n seeds (n = 3, 5, 10). Right: C minus
matched reflection (B) as seeds accumulate. Code:
[`seed_count_analysis.py`](src/gepa_pilot/seed_count_analysis.py).*

- **3 seeds were misleading on AG News:**
  - 95% CI of ±3–9 pp;
  - C's lead over B fell from +4.3 (n = 3) to +0.9 pp (n = 10).
- **On emotion, the effect sharpened with more seeds:** +0.08 (p = 0.85) at n = 3, +1.13
  (p = 0.009) at n = 10.

**Seeds per arm needed for 80% power** (two-sided α = 0.05, using each pair's sd at n = 10):

| Task | Comparison | Observed difference | sd | Seeds for the observed difference | For 1 pp | For 2 pp |
|---|---|---:|---:|---:|---:|---:|
| AG News | F vs. I (equal compute) | +2.11 | 2.30 | **19** | 83 | 21 |
| AG News | F vs. H (equal compute) | +1.94 | 2.24 | **22** | 80 | 20 |
| AG News | E vs. A′ (polish effect) | +2.44 | 2.05 | 12 | 67 | 17 |
| AG News | G vs. C | +1.60 | 2.98 | 55 | 140 | 35 |
| AG News | I vs. C | +1.13 | 2.35 | 68 | 87 | 22 |
| Emotion | G vs. C | −0.06 | 0.96 | — | 15 | 4 |
| Emotion | I vs. C | +0.12 | 0.90 | — | 13 | 4 |

- **How to use it:** if a comparison's observed difference needs about 20 seeds or fewer, it's
  worth running more. If it needs 50+, the difference is too small to matter at this scale.
- **Worth more seeds:**
  - F vs. I and F vs. H on AG News: 20 seeds per arm would settle whether gradient insertion
    beats equal compute;
  - the polish effect is already at about the needed power.
- **Not worth more seeds:**
  - G vs. C and I vs. C on AG News (55–68 needed);
  - all emotion ties (the differences are about 0).

---

## Q6. ADMM: the best candidate is rarely from the last round

Each ADMM proposal runs 20 rounds, scores every new candidate on S, and returns the best. Code:
[`iterate_diagnostics.py`](src/gepa_pilot/iterate_diagnostics.py). Per-arm numbers:
`artifacts/iterate_diagnostics/summary.json`.

![Round of the returned candidate](artifacts/figures/fig7a_admm_returned_round.png)

*Figure 7a. The round each returned candidate came from (fair beam ADMM proposer). Median round
12–13; round 20 supplies only 7 of 98 (AG News) and 5 of 97 (emotion).*

![What the round-20 iterate is](artifacts/figures/fig7b_admm_round20_status.png)

*Figure 7b. What "take the last round" would return: mostly the parent itself (72% AG News, 53%
emotion), otherwise a revisit or a worse candidate. The unlabeled AG News segment is 8%.*

![One proposal's trajectory](artifacts/figures/fig7c_admm_trajectory.png)

*Figure 7c. One AG News proposal from the seed instruction. It visits several 1–5 token edits
(e.g. "ignore **linguistic** details", "**Pleasefully** determine…"), returns its best at S-loss
9.46, and ends round 20 back at the parent (11.76).*

- **Oscillation, not convergence:**
  - the iterate returns to the parent 1.5–4 times per proposal;
  - in round 20 it is the parent in 53–82% of proposals.
- **The returned candidate is from round 20 in only 2–15% of proposals.**
- **The round-20 candidate is worse on S** than the returned one in 82–98% of cases (+0.25 to
  +1.0 nats).
- **Hypothesis (mechanism):**
  - the dual u builds up until the projection jumps to a new token;
  - the residual then reverses, u unwinds, and the projection jumps back;
  - top-p and beam always keep the parent's token available, which makes returning easy.
- **Implication:** keep the best candidate seen, and pool candidates across refinements
  (as D, F, I do); never take the last round.

---

## Q7. Flips: what they tell us for the next step

**Definitions:**
- **Scoring:** for one parent → child step, both prompts are scored on the same 512 validation
  examples.
- **Categories:** each example is a **fix** (wrong → right), a **break** (right → wrong),
  **wrong → different wrong**, or unchanged.
- **Flips** = all predictions that changed. **Net** = fixes − breaks = the accuracy change in
  examples.
- **Code:** [`proposal_diagnostics.py`](src/gepa_pilot/proposal_diagnostics.py),
  [`proposal_examples.py`](src/gepa_pilot/proposal_examples.py). Worked examples:
  [`examples.md`](artifacts/proposal_diagnostics/examples.md).

![Fixes vs breaks](artifacts/figures/fig9b_fixes_vs_breaks.png)

*Figure 9b. Mean fixes, breaks and wrong → different wrong per changed prompt (one experiment per
proposer, 120 proposals each). Right: net per change, net per proposer call, and how many of the
120 proposals changed the prompt.*

![Average child to final prompt](artifacts/figures/fig9d_average_to_final.png)

*Figure 9d. Net gain at each stage: all children and accepted children (vs. their parent), the
best child per run, and the final prompt (vs. the seed instruction).*

**What the flips show:**
- **Both proposers move many predictions:** reflection rewrites about 40% of the text and
  changes about 17% of predictions; ADMM edits 1–3 tokens and changes 12–15%.
- **The difference is direction:** on AG News both fix about 24–27 examples per change, but
  reflection breaks more (21.8 vs. 16.3–19.3), so net +4.1 vs. +7.3 to +7.9.
- **The average child is not the outcome:** GEPA discards rejected children and builds on the
  best accepted ones.
  - On emotion every proposer's average child is negative (−2 to −6), yet C's final prompt ends
    +7.2 examples above the seed, against −0.3 for reflection.

**What this suggests for the next step:**
- **Combine reflection and ADMM.** Reflection's large rewrites move behavior a lot but in no
  consistent direction; ADMM's small edits are directed.
  - This motivated reflect-then-polish, which works on AG News (Q3).
  - **Hypothesis:** small directed edits on top of big undirected rewrites is a good division of
    labor. Gradient insertion (F) is a similar "directed small edit".
- **Invest in acceptance and selection,** not only proposal quality. Outcomes follow the best
  accepted children, so a rule that reliably keeps good children and drops bad ones should pay
  off more than an average-child improvement.
  - Supported on AG News by Q3; on emotion this is the likely bottleneck (Q3, hypothesis).
- **Wrong → different-wrong flips** (about half of all flips) don't change accuracy. They do
  show which classes a prompt confuses.
  - **Hypothesis:** per-class targets (e.g. the dual-weighted loss from Phase 5) could use them.
    Untested.

---

## Appendix: run notes and data quality

- **Out-of-memory failures on the shared GPUs** hit 4 Phase 7 runs.
  - GEPA logs a proposal's GPU error and continues, so a run can "finish" with missing
    proposals.
  - The queue ([`gpu_queue.py`](scripts/gpu_queue.py)) and
    [`failure_watcher.py`](scripts/failure_watcher.py) detect GPU errors in either log and rerun
    with backoff. All reruns succeeded.
- **Gated ADMM crash:** a spare slot whose nucleus held only disallowed tokens had no candidates.
  It affected 3 emotion runs. Fixed (fallback to the most likely allowed token) and rerun with
  the same seeds.
- **Multi-start bug caught before Phase 9:** extra starts initially saw the same batches
  (reshuffling only happens after a full pass). Extra starts now draw their own order.
- **2k re-scoring:** each phase's scoring used to rebuild the run → score table, wiping earlier
  phases. No scores were lost (all are cached by prompt); the table now merges.
- **Silent-failure check:** all earlier run logs were scanned for errors GEPA had swallowed;
  none were found.
- **Incident 2026-10-15 23:48:** a `git stash -u` moved in-progress Phase 7 outputs. Affected
  runs were discarded and restarted; run directories are now git-ignored.
