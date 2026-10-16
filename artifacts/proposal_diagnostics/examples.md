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

