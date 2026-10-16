## Example accepted proposals (AG News)

Bold = inserted words, strikethrough = deleted words. Net = validation net of the child vs. its parent (correct-count change on the 512 validation examples; +5 ≈ +1 pp).

### C fixed-length beam ADMM (79 accepted)

- **largest net, net +41** (agnews_beam_indep_s4): ~~Carefully determine the~~ **Please specify your preferred** article's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- **median net, net +10** (agnews_beam_indep_s4): Carefully determine the article's main topic from the label mapping, ignore ~~incidental~~ **other** details, and output only the corresponding capital letter.

### I multi-start, winner from an extra start (48 accepted)

- **largest net, net +44** (agnews_multistart_s6): Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the category ~~capital~~ **name** letter.
- **median net, net +10** (agnews_multistart_s9): Carefully determine the article's main topic from the label ~~mapping, ignore incidental~~ **mapping and provide the** details, and output only the corresponding capital ~~letter.~~ **letters.**

### D multi-length, LM insertion (2-longer variant) (7 accepted)

- **largest net, net +32** (agnews_multilength_s6): ~~Carefully~~ **###fully** determine the ~~article's~~ **most likely's** main topic from the label mapping, ignore incidental details, and output only the corresponding label name and **the** letter.
- **median net, net -10** (agnews_multilength_s6): Carefully determine the article main topic from the label mapping for it. Output only the corresponding **label** capital **letter** letter.

### D multi-length, LM insertion (2-shorter variant) (47 accepted)

- **largest net, net +36** (agnews_multilength_s3): Carefully ~~determine~~ **choose** the article's main topic from the label ~~mapping, ignore incidental details,~~ **mapping and classify it** and output only the corresponding capital letter.
- **median net, net +31** (agnews_multilength_s3): Carefully determine the article's main topic from the label ~~mapping, ignore incidental details,~~ **mapping and provide details** and output only the corresponding capital letter.

### F multi-length, gradient insertion (2-longer variant) (65 accepted)

- **largest net, net +53** (agnews_mlgrad_s1): Carefully determine the article's main topic from the ~~label mapping,~~ **text, and** ignore incidental details, and output only the **numeric** corresponding **target** capital letter.
- **median net, net +12** (agnews_mlgrad_s1): Carefully class determine the article's main topic from the label mapping and extract details to output only the **JSON** numeric corresponding **of** capital letter.

### G gated, length changed (65 accepted)

- **largest net, net +35** (agnews_gated_s5): Carefully determine the ~~article's~~ **most appropriate and** main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- **median net, net +13** (agnews_gated_s7): Carefully **read** determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding **label** capital letter.

### G gated, same length (17 accepted)

- **largest net, net +21** (agnews_gated_s9): Carefully determine the article's main topic from the label ~~mapping, ignore incidental~~ **mapping and input text** details, and output ~~only~~ **the** the corresponding capital letter.
- **median net, net +13** (agnews_gated_s1): Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding ~~capital~~ **label** letter.

## Final prompts (AG News, each arm's median-scoring seed on the 2k set)

| Arm | 2k accuracy | Final prompt |
|---|---:|---|
| A: reflection, minibatch rule (12) | 34.05 | Carefully identify the article's central topic based on the label mapping, focusing on the primary event or subject, and output the corresponding capital letter. |
| A′: reflection, S′ judge (12) | 34.35 | Carefully analyze the article to identify the central topic, focusing on the primary subject and context, and select the corresponding capital letter from the label mapping. |
| B: reflection, S′ judge (30) | 34.50 | Carefully identify the article's central topic from the label mapping, focusing on the primary subject and ignoring incidental details, and output only the corresponding capital letter. |
| E: reflection + ADMM polish, S′ judge | 35.70 | Carefully identify the article's central topic based on the label mapping, focusing on the primary subject matter and ignoring incidental details, and output only the corresponding category letter. |
| C: fixed-length beam ADMM | 35.55 | Carefully determine the article's main topic from the label mapping (ignore formatting details, and output only the corresponding capital letter. |
| I: multi-start, 3 ADMM runs/proposal | 35.60 | Carefully determine the article's main topic from the text below, ignore linguistic details, and output only the corresponding topic letter. |
| H: fixed-length ADMM, 36 proposals | 36.70 | Carefully choose the article's main topic from the label mapping, ignore linguistic details, and output only the corresponding capital letter. |
| D: multi-length, LM insertion | 35.40 | Carefully determine the article's main topic from the label mapping and provide details and output only the corresponding capital letter. |
| F: multi-length, gradient insertion | 37.55 | Carefully extract determine the topics article's main objective character topic from the label mapping, ignoring incidental details, and output only the corresponding capital letter. |
| G: gated variable length | 35.55 | Please determine the article's main topic from the label mapping, ignore the details, and output only the corresponding label capital letter. |

