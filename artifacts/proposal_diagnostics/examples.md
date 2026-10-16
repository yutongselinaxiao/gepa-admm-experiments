**Example A: a typical reflection proposal (AG News)**

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

**Example B: a typical beam ADMM proposal (AG News)**

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


#### Average child vs. what GEPA keeps (like-for-like runs, 512 validation examples)

| Task / proposer | Mean net, all children | Children with net ≥ +5 | Accepted: count, mean net | Best child per run (mean) | Final prompt vs. seed (mean) |
|---|---:|---:|---:|---:|---:|
| AG News / reflection | +4.1 | 48% | 25, +10.6 | +23.5 | +18.3 |
| AG News / top-p ADMM | +7.3 | 60% | 77, +8.9 | +24.5 | +33.2 |
| AG News / beam ADMM (fair S') | +7.9 | 67% | 79, +9.3 | +25.4 | +29.0 |
| Emotion / reflection | -2.5 | 10% | 20, -1.9 | +6.5 | -0.3 |
| Emotion / top-p ADMM | -5.7 | 11% | 44, -5.6 | +4.8 | +0.7 |
| Emotion / beam ADMM (fair S') | -2.0 | 23% | 45, -0.8 | +12.3 | +7.2 |

#### Like-for-like proposal table (10 runs x 12 proposals per row)

| Task / proposer | Changed / made | Text edited (median) | Flip rate (median) | Fixes | Breaks | Wrong → different wrong | Net per change | Net per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AG News / reflection | 115 / 120 | 41% | 17.8% | 25.8 | 21.8 | 44.4 | +4.1 | +3.9 |
| AG News / top-p ADMM | 98 / 120 | 4% | 14.8% | 26.7 | 19.3 | 39.2 | +7.3 | +6.0 |
| AG News / beam ADMM (fair S') | 98 / 120 | 4% | 12.7% | 24.1 | 16.3 | 34.1 | +7.9 | +6.4 |
| Emotion / reflection | 120 / 120 | 61% | 20.1% | 32.0 | 34.5 | 35.0 | -2.5 | -2.5 |
| Emotion / top-p ADMM | 66 / 120 | 4% | 14.2% | 26.6 | 32.3 | 26.7 | -5.7 | -3.1 |
| Emotion / beam ADMM (fair S') | 97 / 120 | 4% | 13.8% | 24.6 | 26.6 | 24.7 | -2.0 | -1.6 |

#### All runs pooled, per proposer

| Task / proposer | Proposals counted | Text edited (median) | Flip rate (median) | Fixes | Breaks | Wrong → different wrong | Accuracy change (median) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AG News / reflection | 405 | 40% | 17.4% | 26.3 | 24.6 | 45.9 | +0.00 pp |
| AG News / Euclidean ADMM | 97 | 4% | 12.5% | 25.5 | 18.2 | 37.8 | +1.17 pp |
| AG News / top-p ADMM | 98 | 4% | 14.8% | 26.7 | 19.3 | 39.2 | +1.76 pp |
| AG News / beam ADMM | 191 | 4% | 11.7% | 25.7 | 17.0 | 35.1 | +1.95 pp |
| AG News / reflect-then-polish | 36 | 38% | 26.6% | 37.8 | 33.7 | 63.6 | +0.78 pp |
| Emotion / reflection | 412 | 48% | 18.9% | 30.3 | 33.3 | 30.5 | -0.39 pp |
| Emotion / top-p ADMM | 66 | 4% | 14.2% | 26.6 | 32.3 | 26.7 | -0.52 pp |
| Emotion / beam ADMM | 186 | 4% | 13.4% | 23.5 | 25.8 | 24.6 | -0.26 pp |

#### Acceptance reliability (share of accepted / rejected children actually better)

| Task / proposer / acceptance rule | Accepted | …actually better | Rejected | …actually better |
|---|---:|---:|---:|---:|
| agnews / reflection / minibatch (GEPA default) | 25 | 72% | 90 | 48% |
| agnews / reflection / held-out judge | 125 | 72% | 165 | 28% |
| agnews / Euclidean ADMM / minibatch (GEPA default) | 9 | 67% | 40 | 70% |
| agnews / Euclidean ADMM / held-out on S (S used twice) | 41 | 71% | 7 | 43% |
| agnews / top-p ADMM / held-out on S (S used twice) | 77 | 81% | 21 | 57% |
| agnews / beam ADMM / held-out on S (S used twice) | 68 | 87% | 25 | 60% |
| agnews / beam ADMM / held-out on independent S' | 79 | 82% | 19 | 53% |
| agnews / reflect-then-polish / minibatch (GEPA default) | 10 | 60% | 26 | 58% |
| emotion / reflection / minibatch (GEPA default) | 20 | 30% | 100 | 32% |
| emotion / reflection / held-out judge | 61 | 16% | 231 | 28% |
| emotion / top-p ADMM / held-out on S (S used twice) | 44 | 27% | 22 | 27% |
| emotion / beam ADMM / held-out on S (S used twice) | 62 | 37% | 27 | 15% |
| emotion / beam ADMM / held-out on independent S' | 45 | 33% | 52 | 38% |
