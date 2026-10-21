# Results log (2026-10-17 – 2026-10-20)

This log follows [`RESULTS_10_16.md`](RESULTS_10_16.md), which covers AG News (letter answers) and
emotion through Phase 10. This entry adds **three new task settings** and the code adaptations they
needed:
- **Yahoo Answers topics:** a third single-label task.
- **AG News answered with label names:** multi-token answers.
- **GoEmotions:** multi-label.

**Conventions are the same as in `RESULTS_10_16.md`:**
- **Arm letters:** A–I, defined below.
- **Statistics:** 10 seeds per arm; mean ± 95% CI over seeds; Welch's t-tests against C, the
  fixed-length beam ADMM proposer.
- **Hypotheses:** anything not directly measured is marked **Hypothesis**.
- **Numbers:** all come from [`cross_task_analysis.py`](src/gepa_pilot/cross_task_analysis.py),
  with full tables in [`artifacts/cross_task/tables.md`](artifacts/cross_task/tables.md).

---

## Arms (as in `RESULTS_10_16.md`)

| Label | What it is |
|---|---|
| **A** | Reflective GEPA: reflection proposes, GEPA's minibatch rule accepts (12 proposals) |
| **B** | Reflection, S′ judge, **30** proposals (≈ ADMM's compute) |
| **C** | **Fixed-length beam ADMM proposer**, S′ acceptance (reference) |
| **D** | Multi-length ADMM, LM-top-token insertion (AG News letters and emotion only) |
| **E** | Reflection + ADMM polish, S′ judge |
| **F** | Multi-length ADMM, gradient-chosen insertion |
| **G** | Gated variable-length ADMM |
| **H** | C with 36 proposals (3× compute) |
| **I** | Multi-start: 3 ADMM runs per proposal (3× compute) |

## Summary

- **Gradient-chosen insertion (F) is the most consistently helpful ADMM variant:**
  - **Yahoo:** +3.2 pp over C (p < 0.001), and +2.0–2.5 over the equal-compute controls
    (p ≤ 0.01);
  - **AG News with letters:** +3.2 pp;
  - **GoEmotions:** the only arm that improves per-emotion recognition (micro F1 +1.6 over C,
    +1.7 over matched reflection; both p ≤ 0.01). Against the equal-compute controls the micro-F1
    gain shrinks to +1.2–1.3 (p = 0.03–0.06), suggestive rather than settled.
  - It doesn't help on emotion, and nothing helps on AG News with label names.
- **Yahoo** (10 topics, seed instruction 19.6%) confirms variable length beyond compute:
  - both F (+3.2) and gated G (+3.0) clearly beat C;
  - the 3× compute controls gain only about +1;
  - reflection and polish don't differ from C.
- **AG News answered with label names starts at 80% and nothing improves it** (all arms within
  +0.6 pp). With letters the seed instruction scores 29%.
  - **Hypothesis:** most of the earlier AG News gains came from learning the letter mapping, not
    from classifying better.
- **GoEmotions (multi-label):** after fixing the setup (optimize the reported metric, at the
  natural label rate), every arm still learns to **hedge**, listing extra emotions, because
  example F1 rewards it.
  - Reflection's and polish's example-F1 gains (+4.7 to +5.4 over C) are almost entirely hedging:
    micro and macro F1 are flat.
  - F gets the same example-F1 gain with less hedging and is the only arm that improves micro and
    macro F1.

---

## What was added

### A third single-label task: Yahoo Answers topics
I screened three single-input, single-token candidates by scoring the seed instruction and three
hand-written instructions on validation:

| Task | Seed instruction | Best hand-written instruction | Chance | Notes |
|---|---:|---:|---:|---|
| SST-5 (5 sentiment levels) | 39.1% | 39.8% | 20% | rewording barely matters |
| **Yahoo (10 topics, question titles)** | **19.5%** | **25.4%** | 10% | rewording alone gains about 6 pp; the seed sends 58% of answers to one class |
| DBpedia-14 (entity types) | 45.0% | 43.6% | 7% | same collapse onto one class, but rewording doesn't help; long inputs |

- **The pick:** Yahoo, for the most room to improve and its short, fast inputs.
- **Data:** 64 train and 128 validation examples per class; the 2k report set has 200 per class,
  drawn from unused training data. Configs: [`configs/yahoo.yaml`](configs/yahoo.yaml) (plus
  `sst5.yaml`, `dbpedia.yaml` for the screened candidates).

### Code adaptations (all keep the original single-label, letter-answer runs bit-identical)
- **Several text inputs per example:**
  - `task.text_field` can be a list (e.g. premise and hypothesis), written into the input block as
    "Heading: value" lines;
  - the instruction slot is unchanged, so ADMM needs nothing new;
  - tested end to end on RTE ([`configs/rte.yaml`](configs/rte.yaml)), but not run, because RTE's
    seed instruction already scores 85%.
- **Multi-token answers** (`prompt.answer_style: name`):
  - the model answers with the label name, and each class is scored by the likelihood of its full
    answer;
  - the prompt runs once; its KV cache is repeated per class and only the label tokens are added;
    logits are computed only at label positions;
  - matches brute force within 0.0002; gradients reach embeddings and gates; gate = 0 is still
    exact deletion.
- **Multi-label** (`task.multi_label`): described in the GoEmotions section below.
- **Verification that old runs are unaffected:**
  - letter-style prompts and reflection templates are byte-identical;
  - evaluation, ADMM discrete metrics and soft-slot scores are identical;
  - two ADMM steps and two gated steps on AG News are bit-identical before and after.

---

## Results across all five settings

![Cross-task comparison](artifacts/figures/fig18_cross_task.png)

*Figure 18. Each arm minus C (fixed-length ADMM), mean ± 95% CI over 10 seeds, per task setting;
filled = p < 0.05. Each panel title gives C's and the seed instruction's score. "Not run" = arm not
included for that setting. GoEmotions is shown twice: example F1 (the optimized metric) and micro F1
(per-emotion recognition, pooled).*

| Arm | AG News, letters | Emotion | Yahoo | AG News, names | GoEmotions, example F1 | GoEmotions, micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seed instruction | 28.90 | 37.03 | 19.55 | 80.00 | 37.0 | 46.0 |
| **C: fixed-length ADMM** | **35.02** | **37.92** | **23.70** | **80.11** | **38.7** | **46.5** |
| A: reflection | −2.06 | **−1.11** | −0.74 | −0.11 | **+4.68** | −0.41 |
| B: reflection, 30 proposals | −0.93 | **−1.13** | +0.75 | −0.07 | **+5.39** | −0.12 |
| E: polish | +1.16 | **−0.92** | +0.89 | +0.52 | **+4.74** | −0.54 |
| D: LM insertion | +0.58 | −0.88 | — | — | — | — |
| **F: gradient insertion** | **+3.24** | **−0.83** | **+3.23** | −0.06 | **+4.52** | **+1.60** |
| G: gated | +1.60 | −0.06 | **+2.97** | +0.17 | **+2.67** | +0.25 |
| I: multi-start | +1.13 | +0.12 | +0.74 | +0.38 | **+1.64** | +0.31 |
| H: 36 proposals | +1.30 | −0.20 | **+1.23** | +0.27 | +1.98 | +0.42 |

The C row gives absolute scores; the other rows give the difference vs. C (bold = p < 0.05).
Accuracy in %, F1 in points.

- **Where variable length helps:**
  - Yahoo: F and G;
  - AG News with letters: F;
  - GoEmotions: F for recognition.
  - **Hypothesis:** topic-like decisions, where adding or removing a phrase can redirect what the
    model attends to.
- **Where it doesn't:**
  - Emotion: small edits change little, and acceptance can't separate children (see
    `RESULTS_10_16.md` Q3);
  - AG News with label names: no room left.
- **Compute controls (I, H)** gain at most about +1–2 anywhere. So F's and G's Yahoo gains aren't
  just extra compute; on AG News with letters and on GoEmotions F's lead over them is borderline.
- **Reflection (A, B)** never beats C on accuracy. On GoEmotions it wins example F1 only by hedging
  (below).
- **Polish (E)** helps only on AG News with letters (`RESULTS_10_16.md` Q3). On GoEmotions it
  hedges like reflection.

### Yahoo Answers topics
- **F vs. equal compute:** F beats multi-start (I) by +2.49 (p = 0.008) and 36 proposals (H) by
  +2.00 (p = 0.010).
- **G vs. equal compute:** G beats I by +2.23 (p = 0.019) and H by +1.74 (p = 0.030), at about C's
  compute.
- **F vs. G:** tied (+0.26, p = 0.75).
- **Polish:** doesn't help (E vs. matched reflection +0.14, p = 0.82), as on emotion and unlike AG
  News with letters.

### AG News answered with label names: no room to improve
| Arm | Runs whose final prompt is the seed instruction | Accepted children per run |
|---|---:|---:|
| A: reflection | **10/10** | 1.3 |
| B: reflection, 30 proposals | 9/10 | 4.7 |
| C: fixed-length ADMM | 3/10 | 6.8 |
| F: gradient insertion | 2/10 | 4.4 |
| H: 36 proposals | 2/10 | 19.2 |

- **The seed instruction already scores 80%;** every arm ends within 80.0–80.6% on the report set.
- **Reflection almost never leaves the seed.** ADMM accepts children, but they don't generalize.
- **Hypothesis:** with letter answers, a 1.7B model spends most of its capacity on the letter
  mapping; prompt optimization then mostly "teaches" the mapping. With label names that problem
  is gone and little remains to fix. The arm comparisons on letter-style AG News stay valid (all
  arms faced the same format), but their absolute gains overstate what prompt optimization does
  for the task itself.

---

## GoEmotions (multi-label)

### Design
- **Task:** 8 frequent emotions (admiration, amusement, anger, annoyance, approval, gratitude, joy,
  love); several can apply to one comment.
- **Data:** half the sampled comments carry two or more of the 8 emotions; on the report set,
  comments have 1.52 true emotions on average. Sizes: 96 train / 192 validation / 250 report
  comments.

```
one example = one comment, gold = its set of emotions (e.g. {admiration, joy})
score(comment):
    for each of the 8 emotions k:
        ask "Comment: … / Candidate emotion: k" → A (expressed) or B (not expressed)
        log_odds_k = score(A) − score(B)
    predicted set = {k : log_odds_k > 0}
GEPA score, selection on S, acceptance on S′  =  F1(predicted set, gold set)      # the reported metric
ADMM loss = 1 − soft F1,   soft F1 = 2 Σ p_k y_k / (Σ p_k + Σ y_k),   p_k = sigmoid(log_odds_k)
reflection feedback: "Predicted …, gold …, missed …, wrongly added …"
reported: example F1 (mean per comment), micro F1 (pooled over all pairs), macro F1 (mean per emotion)
```

- **Why this design:**
  - The first version (Phase 13) optimized balanced yes/no pairs (50% "yes"), while the report set
    is 19% "yes". That pushed prompts toward "yes", and its results are not used.
  - Optimizing pair accuracy at the natural rate would push toward "no".
  - Making the comment the unit and optimizing its F1 removes the mismatch.
- **Checks:**
  - soft F1 equals F1 at 0/1 probabilities;
  - the evaluator's F1 and loss match an independent recomputation;
  - gradients reach embeddings and gates;
  - accumulating gradients over chunks of 4 comments gives the same gradient (difference 2e-6) at
    a third of the memory (13.4 vs. 43.6 GiB).

### Results
| Arm | Example F1 | Micro F1 | Macro F1 | Emotions predicted per comment (true: 1.52) | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Seed instruction | 37.0 | 46.0 | 48.0 | 2.22 | 38.7 | 56.6 |
| A: reflection | 43.4 | 46.1 | 47.6 | 3.24 | 33.9 | 72.0 |
| B: reflection, 30 proposals | **44.1** | 46.4 | 48.0 | 3.29 | 34.0 | 73.3 |
| E: reflection + ADMM polish | 43.4 | 45.9 | 47.4 | 3.28 | 33.7 | 72.4 |
| C: fixed-length ADMM | 38.7 | 46.5 | 48.5 | 2.32 | 38.5 | 58.7 |
| I: multi-start (3× compute) | 40.3 | 46.8 | 48.9 | 2.51 | 37.6 | 62.1 |
| H: 36 proposals (3× compute) | 40.7 | 46.9 | 48.9 | 2.53 | 37.7 | 62.6 |
| G: gated | 41.4 | 46.7 | 48.6 | 2.75 | 36.8 | 65.6 |
| **F: gradient insertion** | 43.2 | **48.1** | **49.9** | 2.75 | 37.7 | 67.4 |

![GoEmotions hedging](artifacts/figures/fig19_goemotions_hedging.png)

*Figure 19. One dot per run, all eight arms: emotions predicted per comment (x) against example F1
(left) and micro F1 (right). Black star: seed instruction. Dashed line: true emotions per comment.*

- **Example F1 rewards hedging:** a comment with one true emotion scores 0.5 if the prompt lists
  three emotions including it, and 0 for one wrong guess. So every arm learns to list more
  emotions, and on the left, example F1 rises with the number listed.
- **Reflection's gain is hedging:** A and B list about 3.3 emotions per comment; recall rises to
  72–73% while precision falls to 34%, and micro and macro F1 stay at the seed's level. **Polish (E)
  behaves the same** (3.28 per comment, micro F1 45.9): its 1-token ADMM edits don't undo the
  reflection draft's hedging.
- **F improves recognition:**
  - same example F1 as matched reflection (−0.86, p = 0.35), with less hedging (2.75 per comment)
    and precision close to the seed's;
  - **micro F1:** +1.60 over C (p = 0.009) and +1.73 over B (p = 0.010);
  - **macro F1:** +1.44 over C (p = 0.031) and +1.96 over B (p = 0.010);
  - no other arm moves micro or macro F1 significantly.
  - **Hypothesis:** gradient-chosen inserted words sharpen each yes/no decision, while
    reflection's rewrites mainly widen what counts as "expressed".
- **F vs. equal compute (I, H):**

  | F minus … | Example F1 | Micro F1 | Macro F1 |
  |---|---:|---:|---:|
  | I: multi-start | **+2.88** (p = 0.004) | **+1.29** (p = 0.028) | +1.03 (p = 0.107) |
  | H: 36 proposals | +2.54 (p = 0.065) | +1.18 (p = 0.064) | +0.99 (p = 0.156) |

  - The extra compute alone (I, H) lifts micro F1 by only +0.3–0.4 over C, so most of F's +1.6
    survives equal compute (+1.2–1.3).
  - At 10 seeds this is significant against I but only borderline against H, and not significant
    for macro F1. **Suggestive, not settled.**
- **Limitation, ADMM's training pool:** GoEmotions has 96 training comments; after GEPA's minibatch
  (8), S (32) and S′ (32), **ADMM trains on only 24 comments**, against 184–568 examples on the other
  tasks. All ADMM arms share this, so the comparison within GoEmotions is fair, but it probably
  handicaps ADMM here. **Hypothesis:** a larger training split (e.g. 288 comments) would favor the
  ADMM arms.

---

## Next steps
- **Settle F vs. equal compute on GoEmotions:** about 20 seeds per arm for F, I and H, and/or a larger
  training split so ADMM has more than 24 comments to train on.
- **A metric that doesn't reward hedging:** e.g. example F1 with a penalty for predicted set size,
  or each comment's contribution to micro F1. This would test whether reflection can improve
  recognition when hedging isn't rewarded.
- **Letter vs. label-name answers for the other tasks:** whether the label-name ceiling also holds
  for Yahoo and emotion.

---

## Appendix: run notes
- **Batches:**
  - Phase 11: Yahoo, 80 runs.
  - Phases 12–13: AG News with label names and GoEmotions, pair-based (superseded); 160 runs.
  - Phase 14 / 14b: GoEmotions, comment-level F1, core arms A, B, C, G, F; 50 runs.
  - Phase 15: GoEmotions, E, I, H; 30 runs.
  - All 10 seeds per arm.
- **Phase 13 GoEmotions (pair-based, balanced 50/50) is superseded;** its results stay in
  `artifacts/report2k/goemotions.json` for reference only.
- **Out of memory in Phase 14:** multi-label ADMM held 16 comments × 8 emotions of activations at
  once (about 44 GB). 23 runs failed, alongside another user's 42–47 GB per GPU. Fixed by gradient
  accumulation; all 30 ADMM runs were rerun with the fix (Phase 14b), and the old folders are kept
  in `artifacts/phase14_goemotions_f1/superseded_unchunked/`.
- **Audit of the running batches (all clean):**
  - no errors swallowed by GEPA in any GoEmotions run;
  - every run made exactly its proposal budget;
  - ADMM losses are 1 − soft F1, and S/S′ scores are F1 values;
  - reflection feedback matches the gold emotions;
  - GEPA's accept/reject always followed the S′ verdict of the same proposal;
  - multi-start's three starts begin from different batches and each contributes candidates.
- **A small acceptance leak:** ADMM sometimes re-proposes the identical child later in a run, where
  a fresh S′ batch judges it again, so a borderline child gets several chances.
  - It accounts for 6% of ADMM proposals on AG News, 0% on Yahoo and 10% on GoEmotions.
  - Children accepted after an earlier rejection: 1, 0 and 4 respectively, so no conclusion changes.
  - Possible fix for future runs: reject a child already rejected from the same parent.
- **Development safety:**
  - new code was developed in a separate worktree, so running batches never imported untested
    code;
  - every new arm/task combination was smoke-tested before its batch, and Phase 14–15 batches were
    gated on smoke jobs.
