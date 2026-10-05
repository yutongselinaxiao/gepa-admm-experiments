# Parity check: pip-installed GEPA vs. the `gepa-admm` fork

**Result: the two runs are identical.** Rerunning the best pilot setting
(reflection minibatch 8) on the editable fork reproduces the original pilot run
exactly. That covers every proposed instruction, every reflection-model output,
every accept/reject decision, every per-example label logit, and the saved GEPA
state file.

Work on the ADMM proposer changes the fork. This run is the reference those
changes are checked against: any change that leaves reflective-only GEPA alone
should still pass this check.

## What was compared

|  | Reference (pilot) | Candidate (fork) |
|---|---|---|
| Run directory | [`artifacts/gepa_confirmation_mb8`](../gepa_confirmation_mb8) | `artifacts/parity_fork_mb8` (this folder) |
| GEPA code | `gepa==0.1.4` wheel from PyPI, in `site-packages` | Editable install of [`yutongselinaxiao/gepa-admm`](https://github.com/yutongselinaxiao/gepa-admm), branch `admm-proposer` at tag `v0.1.4` (`8b0ce6cd`), no local changes |
| Started (UTC) | 2026-10-02 10:07 | 2026-10-05 09:29 |
| GPU | NVIDIA H100 80GB HBM3 | NVIDIA H100 80GB HBM3 (physical GPU 5) |
| Python / PyTorch / Transformers | 3.10.12 / 2.11.0+cu128 / 5.5.4 | same |
| Command | `run_gepa --proposals 12 --reflection-minibatch-size 8` | same |
| Wall time | 132.49 s | 132.35 s |

Before running anything, the wheel was compared file by file with the tag's
`src/gepa`:
- Every file present in both is byte-identical.
- The only differences are non-code files the wheel doesn't ship: adapter
  READMEs, example prompt templates, `gskill/assets`, and similar.
- At this tag, the fork's `pyproject.toml` still says `version = "0.1.3"`.
  Upstream never bumped it, so `pip show gepa` reports 0.1.3 even though the
  code is v0.1.4.

From this run on, `run_manifest.json` records `gepa_source`, `gepa_git_commit`
and `gepa_git_dirty`. That shows which GEPA code produced a run, rather than
relying on a hard-coded version string.

## Metrics

All checks are computed by
[`src/gepa_pilot/compare_parity.py`](../../src/gepa_pilot/compare_parity.py)
and stored in [`parity_report.json`](parity_report.json). The comparison is
strict equality, with no tolerance, at every level.

| Tier | What is compared | Source | Pass condition | Result |
|---|---|---|---|---|
| 1 | Seed instruction's validation score on each of the 512 examples | first `valset_evaluated` event | identical 0/1 vector | identical |
| 2 | The 12 proposed instruction texts, in order | `proposal_end` events | identical strings in identical order | identical; no divergence |
| 3a | Accept/reject decision for each iteration | `candidate_accepted` / `candidate_rejected` events | identical sequence | identical (4 accepted, 8 rejected) |
| 3b | Validation score vectors of all accepted candidates | all `valset_evaluated` events | identical | identical (5 evaluations × 512) |
| 3c | GEPA-selected candidate | `gepa_summary.json` | same index, text, accuracy, loss | identical: candidate 3, 36.72%, loss 6.486268 |
| 4a | Full callback event stream, including reflection prompts and raw reflection outputs | `callback_events.jsonl` | identical | 81/81 events identical |
| 4b | Reflection-model call records (prompt/completion token counts, raw text) | `gepa_result.json → reflection` | identical | 12/12 identical |
| 4c | Post-hoc audit of all 12 unique candidates on the full validation set (accuracy, mean loss, per-class accuracy, confusion matrix, per-example prediction, per-example loss) | `gepa_result.json → audit_by_instruction` | identical | identical |
| 4d | Per-example outputs with the four label logits (A/B/C/D), loss and prompt length | `generated_best_outputs_valset/` | byte-identical files; max abs difference = 0 | 574/574 byte-identical; max \|Δ\| = **0.0** |
| 4e | Serialized optimizer state | `gepa_state.bin` | byte-identical | identical |

Accuracy is the share of examples whose highest-scoring label among A, B, C and
D is the gold label. Loss is cross-entropy over those four label logits at the
answer position. Both definitions are the same as in the pilot README.

**Negative control.** To confirm the check can fail, the reference was compared
against the minibatch-16 pilot run (`gepa_confirmation_mb16`).
- Only Tier 1 passed. Both runs use the same evaluator, so the seed is scored
  identically.
- Tier 2 diverged at the first proposal, and every later tier failed.
- Overall result: `parity: false`.

## Examples from the two runs

In every example below, the reference and fork values are identical. That's the
point of the comparison, so each example is shown once, marked "both runs".

### 1. One reflection step

At iteration 1, GEPA gives the local Qwen reflection model the seed instruction
and the 8 minibatch records built by the adapter. Here is one of those records,
as it appears in the reflection prompt (both runs; the prompt is 6,671 characters,
or 1,760 tokens):

```text
## article
Sun previews next version of Java Sun Microsystems has posted a prerelease,
quot;snapshot quot; version of Java 2 Standard Edition 6.0, codenamed Mustang, ...
## assistant_prediction
A
## gold_class
D
## label_scores
A 63.25 | B 45.25 | C 43.5 | D 48.5
## label_loss
14.75
## feedback
Predicted A (World) but the gold class is D (Sci/Tech). Revise the general decision
rule to distinguish the article's central topic from incidental entities and keywords.
```

The reflection model's raw output (28 tokens, both runs):

> Carefully analyze the article to identify the primary topic, ignoring any
> incidental details, and select the corresponding capital letter from the label
> mapping.

The reflection model samples with temperature 0.7, top-p 0.8 and top-k 20,
re-seeded with `20262002 + call_index` before every call. Even so, all 12
sampled outputs match token for token. On this hardware and software stack,
sampling repeats exactly.

### 2. All 12 proposals and how GEPA treated them

"Minibatch score" is the number correct out of 8 on the proposal's own
reflection minibatch. "Validation" is the post-hoc score on all 512 examples.
GEPA does not use that score to accept or reject.

| # | Parent | Instruction (proposed text) | Tokens | Minibatch score | Decision | Validation acc. | Val. loss |
|---:|---:|---|---:|---:|---|---:|---:|
| 0 | — | *Seed:* Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter. | 25 | — | seed | 30.47% | 9.2584 |
| 1 | 0 | Carefully analyze the article to identify the primary topic, ignoring any incidental details, and select the corresponding capital letter from the label mapping. | 27 | 4 vs 4 | rejected (tie) | 30.47% | 8.4433 |
| 2 | 0 | Carefully identify the article's central topic by focusing on the primary subject and context, ignoring incidental details, and select the corresponding capital letter from the label mapping. | 32 | 4 | **accepted → cand. 1** | 34.77% | 6.1397 |
| 3 | 1 | …central *theme* … prioritizing the main event or issue, and ignoring incidental details. Select the corresponding capital letter from the label mapping. | 40 | 0 vs 0 | rejected (tie) | 34.96% | 6.4502 |
| 4 | 0 | Carefully analyze the article to identify the central topic, focusing on the main subject and context, and select … ignoring any incidental details or keywords. | 37 | 2 vs 2 | rejected (tie) | 32.03% | 6.9312 |
| 5 | 1 | Carefully identify the article's central theme by focusing on the primary subject and context, prioritizing the main event or issue, and ignoring incidental details, then select the corresponding capital letter from the label mapping. | 41 | 5 | **accepted → cand. 2** | 31.84% | 7.7014 |
| 6 | 2 | *(identical to its parent, candidate 2)* | 41 | 3 vs 3 | rejected (tie) | 31.84% | 7.7014 |
| 7 | 0 | Carefully identify the article's central topic based on the label mapping, focusing on the primary subject matter and ignoring incidental details, and output only the corresponding capital letter. | 33 | 2 vs 2 | rejected (tie) | 33.59% | 6.1855 |
| 8 | 2 | Carefully analyze the article to determine the primary subject and context, focusing on the main event or issue, and ignore incidental details. Select the corresponding capital letter from the label mapping. | 36 | 4 | **accepted → cand. 3 (GEPA-selected)** | **36.72%** | 6.4863 |
| 9 | 1 | Carefully identify the article's central topic by focusing on the primary subject and context, prioritizing the main event or issue, and ignoring incidental details, then select … | 41 | 4 | **accepted → cand. 4** | 30.08% | 7.5603 |
| 10 | 1 | …central topic by focusing on the primary subject and context, prioritizing the main event or issue, and select … | 36 | 2 vs 2 | rejected (tie) | 34.77% | 5.8798 |
| 11 | 1 | …central topic by focusing on the primary subject and context, distinguishing it from incidental details and keywords, and select … | 36 | 1 vs 1 | rejected (tie) | 32.23% | 5.0672 |
| 12 | 3 | Carefully analyze the article to identify the primary subject and context, focusing on the main event or issue, and ignore any incidental details. Select … | 37 | 3 vs 3 | rejected (tie) | 36.33% | 6.5799 |

A few points about meaning, all identical in both runs:
- **All 8 rejections were ties.** No proposal scored *worse* on its minibatch;
  GEPA's strict-improvement rule rejected each one for scoring equal to its
  parent.
- **Proposal 6 copies its parent word for word.** The reflection model
  returned candidate 2 unchanged.
- **Minibatch acceptance and full-validation quality disagree.**
  - Proposal 9 was accepted but scores 30.08% on the full validation set,
    worse than the seed.
  - Proposals 3 and 10 were rejected even though each scores 34.8–35.0%.
  
  This is the noisy-acceptance problem documented in the pilot README.
- **Proposals drift along a narrow path.** Almost every proposal keeps the
  "Carefully … incidental details … capital letter" frame. The changes are
  "topic → central topic/theme → primary subject and context", adding "main event
  or issue", and replacing "output only" with "select … from the label mapping".

### 3. Token level: seed vs. the GEPA-selected instruction

The Qwen3 tokenizer splits the seed into 25 tokens and the selected instruction
(candidate 3) into 36. `Ġ` marks a leading space.

```text
seed (25):  Care|fully| determine| the| article|'s| main| topic| from| the| label| mapping|,|
            ignore| incidental| details|,| and| output| only| the| corresponding| capital| letter|.
best (36):  Care|fully| analyze| the| article| to| determine| the| primary| subject| and| context|,|
            focusing| on| the| main| event| or| issue|,| and| ignore| incidental| details|.| Select|
            the| corresponding| capital| letter| from| the| label| mapping|.
```

A token-level alignment of the two, using `difflib.SequenceMatcher` on token
strings:

| Operation | Seed tokens | Selected-instruction tokens |
|---|---|---|
| keep | `Care fully` | `Care fully` |
| insert | — | `Ġanalyze Ġthe Ġarticle Ġto` |
| keep | `Ġdetermine Ġthe` | `Ġdetermine Ġthe` |
| replace | `Ġarticle 's` | `Ġprimary Ġsubject Ġand Ġcontext , Ġfocusing Ġon Ġthe` |
| keep | `Ġmain` | `Ġmain` |
| replace | `Ġtopic` | `Ġevent Ġor Ġissue , Ġand Ġignore Ġincidental Ġdetails . ĠSelect Ġthe Ġcorresponding Ġcapital Ġletter` |
| keep | `Ġfrom Ġthe Ġlabel Ġmapping` | `Ġfrom Ġthe Ġlabel Ġmapping` |
| delete | `, Ġignore Ġincidental Ġdetails , Ġand Ġoutput Ġonly Ġthe Ġcorresponding Ġcapital Ġletter` | — |
| keep | `.` | `.` |

Token IDs that matter for the ADMM work:
- `Care` = 31999, `fully` = 3641, ` ignore` = 10034, ` incidental` = 83216.
- The labels `A`/`B`/`C`/`D` are single tokens, IDs 32–35.

GEPA's edit is a sentence rewrite: it changes length by 11 tokens and moves a
whole clause. The pilot's ADMM ρ=1 edit changed exactly 2 tokens in place
(` ignore` → ` ignores`, and `,` → newline). This contrast is the reason for
running ADMM refinement *after* a GEPA proposal rather than in place of it.

### 4. Per-example label logits

Each row shows the four label logits at the answer position. Both runs produced
the same values to every printed digit, and the saved files are byte-identical.

| Validation example (gold) | Instruction | A | B | C | D | Prediction | Loss |
|---|---|---:|---:|---:|---:|---|---:|
| `23955`: "Web Data Transfer to Handsets Made Easier (AP) … feed phone numbers" (**D** Sci/Tech) | seed | **59.5** | 48.75 | 43.0 | 49.5 | A ✗ | 10.0001 |
|  | cand. 3 (selected) | 48.0 | 48.25 | 43.5 | **54.75** | D ✓ | 0.0027 |
| `26712`: "Tech investors see sunnier days ahead … revenue warning from … Broadcom" (**C** Business) | seed | 49.5 | 48.75 | 48.25 | **54.75** | D ✗ | 6.5092 |
|  | cand. 3 (selected) | 37.0 | 42.0 | **54.5** | 48.25 | C ✓ | 0.0019 |
| `28779`: "Beware India, Pakistan are on fire … thrashed Kenya by seven wickets" (**B** Sports) | seed | **58.75** | 54.25 | 43.75 | 49.75 | A ✗ | 4.5112 |
|  | cand. 1 | 49.25 | **53.0** | 41.5 | 46.75 | B ✓ | 0.0251 |
| `112714`: "Bush to Focus on Tort Reform at Forum (AP) … federal deficit" (**A** World) | seed | 53.75 | 50.0 | 44.5 | **54.75** | D ✗ | 1.3196 |
|  | cand. 2 | **51.75** | 46.25 | 41.75 | 51.25 | A ✓ | 0.4766 |

The logits are in BF16, which is why they move in steps of 0.25. Exact equality
here means the fork reproduced the forward passes bit for bit, not just the same
argmax.

### 5. Aggregate change: seed → selected instruction

Computed from the per-example predictions in the post-hoc audit (identical in
both runs):
- 147 of 512 predictions change.
- **59 errors fixed, 27 new errors**, a net +32 examples, or +6.25 points
  (30.47% → 36.72%).

| Gold | Seed predicted | Selected predicted | Count | Effect |
|---|---|---|---:|---|
| D Sci/Tech | A | D | 57 | fixed |
| C Business | A | D | 32 | still wrong |
| B Sports | A | D | 26 | still wrong |
| A World | A | D | 24 | newly wrong |
| B Sports | B | D | 3 | newly wrong |
| C Business | D | C | 1 | fixed |
| B Sports | A | B | 1 | fixed |
| D Sci/Tech | A | C or B | 2 | still wrong |
| C Business | D | A | 1 | still wrong |

Almost all of the gain comes from moving the model's default answer from World
(A) toward Sci/Tech (D). Sports and Business are still largely misclassified, as
the pilot README notes. These numbers are not new results: they reproduce the
pilot exactly and are listed only so the outputs can be checked.

## What this check does not establish

- **It is one configuration.** Minibatch sizes 4 and 16 were not rerun on the
  fork. The code is byte-identical, so they are expected to match, but that
  hasn't been observed.
- **The hardware stack was fixed.** Exact repeatability was observed on H100 with
  BF16, PyTorch 2.11.0+cu128 and Transformers 5.5.4. A different GPU type or
  library version may change the sampled reflection text, and with it
  everything downstream. Then the comparison would have to be statistical
  across seeds, not exact.
- **It says nothing about ADMM.** The ADMM proposer will add new code paths to
  the fork. This check only ensures that, with ADMM switched off, the existing
  GEPA path is unchanged.
- **Wall time is not compared.** It differed by 0.14 s, which is noise.

## Reproduce

```bash
source /dataMeR2/yutong/envs/myenv/bin/activate
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1
python -c "import gepa; print(gepa.__file__)"   # should be under third_party/gepa/src

CUDA_VISIBLE_DEVICES=5 python -m gepa_pilot.run_gepa \
  --proposals 12 --reflection-minibatch-size 8 \
  --run-dir artifacts/parity_fork_mb8_rerun     # must be a new directory
python -m gepa_pilot.compare_parity \
  --reference artifacts/gepa_confirmation_mb8 \
  --candidate artifacts/parity_fork_mb8_rerun   # exits 0 on parity, 1 otherwise
```
