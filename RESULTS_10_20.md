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

## Method specifications: D, E, F, G

Formulation and implementation, written as Python-style pseudocode with a comment on each step.
The real code is in [`admm_proposer.py`](src/gepa_pilot/admm_proposer.py),
[`gated_admm.py`](src/gepa_pilot/gated_admm.py) and [`objective.py`](src/gepa_pilot/objective.py).
Every method is a **proposer** inside the same GEPA loop: GEPA picks a parent (Pareto rule) and an
8-example minibatch, the proposer returns one child, and the child is accepted or rejected.

### Shared core: fixed-length ADMM refinement, projection and acceptance (this alone is arm C)

```python
# Notation
#   x        parent instruction, as tokens x_1..x_L        E(w)   embedding of token w
#   prefix   the fixed prompt text before the instruction slot
#   ℓ_B(e)   task loss of slot embeddings e on a batch B (objective.py):
#              single-label: cross-entropy over the label scores
#              multi-label:  1 − soft F1 per text, soft F1 = 2 Σ_k p_k y_k / (Σ_k p_k + Σ_k y_k), p_k = sigmoid(log-odds_k)
#   S, S′    two disjoint 32-example training batches: S picks the candidate, S′ judges acceptance
#   ℓ_S(x)   loss of a discrete prompt x on S (real tokens, no gradients); ℓ_S2(x) the same on S′
#   select_on_S(pool, parent): the candidate with the LOWEST ℓ_S in the pool, if that loss is below
#            the parent's ℓ_S; otherwise the parent itself (= "nothing better found")
#   T        the remaining training examples (ADMM's own training data, drawn 16 at a time)

def admm_refine(x, rounds=20, rho=0.1, lr=0.005, e_steps=3):
    e, z, u = E(x), x, 0                           # continuous copy e, discrete tokens z, scaled dual u
    candidates = {}
    for t in range(rounds):
        B = next_batch(T, 16)                      # a different 16-example batch every round
        for _ in range(e_steps):                   # E-step: move e to lower the task loss,
            e -= Adam(grad_e(ℓ_B(e) + rho/2 * ||e - E(z) + u||**2), lr)   # while staying near the tokens z
        z = beam_project(target=e + u)             # Z-step: closest fluent token sequence (below)
        u += e - E(z)                              # dual step: accumulate the remaining gap e − E(z)
        if z != x and round_trips(z):              # a new prompt that re-tokenizes exactly
            candidates[z] = ℓ_S(z)                 # score it on S with the real tokens (no gradients)
    return select_on_S(candidates, parent=x)       # lowest ℓ_S among all rounds (not the last round,
                                                   # RESULTS_10_16 Q4), and only if it beats the parent on S

def select_on_S(pool, parent):
    best = min(pool, key=ℓ_S)                      # the candidate with the lowest loss on S
    return best if ℓ_S(best) < ℓ_S(parent) else parent   # no candidate beats the parent: return the parent

def beam_project(target, lam=0.3, top_p=0.99, width=4):
    # Left to right, keeping the `width` cheapest partial sequences. At position i the allowed tokens are
    # the LM's top-p nucleus given prefix + tokens chosen so far, plus the parent's own token x_i.
    #   cost(w_1..w_i) = Σ_j ||target_j − E(w_j)||²   +   lam · Σ_j NLL_LM(w_j | prefix, w_<j)
    #                    └ close to the ADMM target ┘      └ fluent text ┘
    return cheapest full sequence

def accept(child, parent):                         # the fair S′ judge used by every ADMM arm and by E
    return ℓ_S2(child) < ℓ_S2(parent) and metric_S2(child) >= metric_S2(parent)
    # metric = accuracy, or mean comment F1 for GoEmotions; S′ is disjoint from S, the minibatch and T

def propose_C(parent):
    child = admm_refine(parent)                    # one fixed-length refinement
    return child, accept(child, parent)
```

### D and F: multi-length ADMM (D = LM insertion, F = gradient-chosen insertion)

```python
def propose_multilength(parent, insert_two):       # D: insert_two = insert_two_lm;  F: insert_two_gradient
    B = first_batch(T, 16)
    s = [norm(grad_{e_i} ℓ_B(E(parent))) for i in range(L)]   # saliency: how strongly the loss reacts to token i
    short = delete(parent, two_smallest(s, exclude_last=True)) # 2-token-shorter prompt
    long_ = insert_two(parent, s)                              # 2-token-longer prompt
    pool  = {short, long_}                         # the raw variants are candidates themselves
    for start in (parent, short, long_):           # three refinements, each keeping its own length
        pool |= candidates_of(admm_refine(start))
    child = select_on_S(pool, parent)              # lowest ℓ_S over all three lengths, if it beats the parent
    return child, accept(child, parent)

def insert_two_lm(x, s):                           # D
    for g in two_largest(s):                       # before the 2 most loss-sensitive tokens
        w = argmax_w P_LM(w | prefix, x[:g])       # the LM's single most likely next token: no task signal
        x = x[:g] + [w] + x[g:]
    return x

def insert_two_gradient(x, s):                     # F: two insertions, one at a time
    for _ in range(2):
        trials = []
        for g in six_largest(s):                   # candidate gaps
            w0   = argmax_w P_LM(w | prefix, x[:g])                   # placeholder token at the gap
            grad = grad_{e_g} ℓ_B(E(x[:g] + [w0] + x[g:]))            # loss gradient at the inserted slot
            W    = top_k(P_LM(. | prefix, x[:g]), 64)                 # only fluent candidates
            W8   = smallest_8(W, key=lambda w: (E(w) - E(w0)) @ grad) # first-order loss change (GCG-style)
            trials += [(g, w) for w in {w0} | W8]
        g, w = min(trials, key=lambda gw: ℓ_32(x[:g] + [gw[1]] + x[g:]))  # exact loss on 32 training examples
        x = x[:g] + [w] + x[g:]
    return x
```

| | D: LM insertion | F: gradient insertion |
|---|---|---|
| Cost vs. C | about 2.9× (three refinements) | about 3.4× (plus the insertion search) |
| Flags | `--admm-length-offsets 0 -2 2` | `--admm-length-offsets 0 -2 2 --admm-insertion gradient` |
| Run on | AG News (letters), emotion | all five settings |

**D examples** (AG News with letters, accepted; change in correct answers out of 512 validation
examples):
- **Shorter variant, +31** (D's most common winning edit):
  - `delete` picks the two least salient tokens, which are just the commas: "… from the label
    ~~mapping,~~ **mapping** ignore incidental ~~details,~~ **details** and output …"
  - `admm_refine` then rewrites the freed span: "… from the label ~~mapping, ignore incidental
    details,~~ **mapping and provide details** and output only the corresponding capital letter."
- **Longer variant, −10:** `insert_two_lm` duplicates a nearby word ("… the corresponding
  **label** capital **label** letter"), and ADMM only partly repairs it ("… **label** capital
  **letter** letter").
- **Result:** on AG News the longer variant won only 16 of D's 114 proposals.

**F examples** (accepted, from the longer variant):
- **Yahoo, +24 correct answers of 1,280:**
  - `insert_two_gradient` inserts "Carefully **remove** determine the question's main topic … the
    corresponding ~~capital~~ **capital-letter** letter";
  - `admm_refine` then edits it to "… from the label ~~mapping, ignore incidental~~ **mapping. This
    entails** details, …".
- **GoEmotions, +8.4 F1 points:**
  - inserted: "Carefully decide whether the **person** comment expresses the **emotion** candidate
    emotion …";
  - accepted child: "… the **person** comment expresses the ~~candidate~~ **emotion of** emotion …".
- **Result:** the inserted words carry task signal, so the longer variant won 101 of F's 116 proposals
  on AG News. But they are often ungrammatical, which is F's fluency cost.

### G: gated variable-length ADMM

```python
# Relaxation of deletion. Each slot i has an embedding e_i and a keep-gate g_i = sigmoid(a_i):
#   attention logit from any query to slot i   +=  log g_i
#   position of slot i                          =   p0 + Σ_{j<i} g_j
# g_i = 1 is the normal token; g_i → 0 is exactly the prompt without token i (checked to 1e-4).

def propose_G(parent, spare=4, rho=0.1, rho_g=1.0, lr_gate=0.1):
    s = saliency(parent)                           # as in D/F
    slots = parent tokens with `spare` spare slots inserted before the most salient positions
    # each spare slot's embedding starts at the LM's most likely token there; its gate starts almost off
    a = [4.0 if parent_slot else -2.0 for slot in slots]   # g ≈ 0.98 for parent tokens, ≈ 0.12 for spares
    k = [1 if parent_slot else 0 for slot in slots]        # current decision per slot: token (1) or blank (0)
    for t in range(20):
        B = next_batch(T, 16)
        for _ in range(3):                         # E-step on embeddings (lr 0.005) and gate logits (lr 0.1)
            L = (ℓ_B(e, g)                                              # task loss with gated attention
                 + rho / 2 * Σ_i k_i * ||e_i - E(z_i) + u_i||**2        # token consensus, filled slots only
                 + rho_g / 2 * Σ_i (g_i - k_i + v_i)**2)                # gate consensus with token/blank
            e, a = Adam(grad(L))
        z, k = gated_beam(target=e + u, gate_target=g + v)    # Z-step: blank or token, per slot
        u = [k_i * (u_i + e_i - E(z_i)) for i]     # token dual; reset to 0 for blank slots
        v = v + g - k                              # gate dual
        if new and round_trips(z without blanks):  # as in admm_refine
            candidates[z without blanks] = ℓ_S(z without blanks)
    child = select_on_S(candidates, parent)        # lowest ℓ_S among all rounds, if it beats the parent
    return child, accept(child, parent)

def gated_beam(target, gate_target, gamma=10.0, lam=0.3):
    # Left to right, beam width 4. For slot i with gate target t_i:
    #   blank:    cost  gamma · t_i²                                  cheap when the gate wants "off"
    #   token w:  cost  ||target_i − E(w)||² − min_w' ||target_i − E(w')||²   distance relative to the closest candidate
    #                 + gamma · (1 − t_i)²                            cheap when the gate wants "on"
    #                 + lam · NLL_LM(w | prefix, chosen tokens)       fluency
    # A slot therefore flips between blank and token roughly when t_i crosses 0.5. The final token is never blank.
    return per-slot choices
```

- **Cost:** about 1.1× C in forward passes.
- **Flags:** `--admm-gated` (spare slots `--admm-spare-slots 4`, gate weight `--admm-gate-weight 10`).
- **Examples** (accepted):
  - **Yahoo, +99 correct answers of 1,280:** one spare slot opened (gate → on) and one token swapped
    by the Z-step: "Carefully ~~determine~~ **classify** the **following** question's intent by
    topic from the label mapping, …"
  - **GoEmotions, +8.1 F1 points:** one spare slot opened: "Carefully decide whether the comment
    expresses the ~~candidate~~ **candidate's** emotion …"
- **Result:** edits are small and mostly fluent, typically one opened spare plus a token swap.
  Original tokens are rarely deleted, because their gates tend to rise toward 1 during the E-steps.

### E: reflect-then-polish

```python
def propose_E(parent, minibatch):
    draft = reflection_LM(parent, mistakes(parent, minibatch))  # GEPA's reflective rewrite, as in arm A
                                                                # (the same Qwen3-1.7B writes it)
    child = admm_refine(draft)               # C's fixed-length beam ADMM, started from the draft; it returns
                                             # the draft unchanged if no candidate beats the draft on S
    return child, accept(child, parent)      # judged against GEPA's parent on S′, not against the draft
```

- **Cost:** about 1.3× C (one reflection call plus one ADMM refinement).
- **Flags:** `--proposer reflect_polish --reflection-acceptance heldout --admm-acceptance-batch-size 32`.
- **Example** (Yahoo, accepted, +34 correct answers of 1,280 vs. GEPA's parent):
  - `reflection_LM` writes "Carefully analyze the question to ~~extract~~ **identify** the primary
    topic, **focusing on the core subject matter and** ignoring any incidental details or keywords,
    and select the corresponding capital letter from the label mapping."
  - `admm_refine` changes one phrase: "… and select the corresponding ~~capital letter~~ **topic
    label** from the label mapping."

More examples (the largest and median accepted child of each method and task):
[`artifacts/cross_task/examples.md`](artifacts/cross_task/examples.md). The quoted effects are
diagnostics on the validation audit, which GEPA never sees:
- AG News: the change in correct answers out of 512;
- Yahoo: the change in correct answers out of 1,280;
- GoEmotions: the change in mean comment F1, in points.

---

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
