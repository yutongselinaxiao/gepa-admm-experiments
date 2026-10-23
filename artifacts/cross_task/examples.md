# Example proposals of E, F and G (accepted children; bold = inserted, strikethrough = deleted)

## Yahoo

### E: reflect-then-polish (44 accepted children where the polish changed the draft)

**largest effect: +84.0 answers vs. GEPA's parent** (draft alone: n/a; yahoo_E_polish_fair_s0)

- GEPA parent: Carefully determine the question's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- reflection draft: Carefully ~~determine~~ **analyze** the question's **core topic, focusing on the** main ~~topic~~ **subject matter, and ignore any incidental details or keywords. Identify the most relevant label** from the ~~label mapping, ignore incidental details,~~ **mapping** and output only the corresponding capital letter.
- after ADMM polish (vs. draft): Carefully analyze ~~the~~ **each** question's core topic, focusing on the main subject matter, and ignore any incidental details or keywords. Identify the most relevant label from the mapping and output only the corresponding capital letter.

**median effect: +34.0 answers vs. GEPA's parent** (draft alone: n/a; yahoo_E_polish_fair_s4)

- GEPA parent: Carefully analyze the question to extract the primary topic, ignoring any incidental details or keywords, and select the corresponding capital letter from the label mapping.
- reflection draft: Carefully analyze the question to ~~extract~~ **identify** the primary topic, **focusing on the core subject matter and** ignoring any incidental details or keywords, and select the corresponding capital letter from the label mapping.
- after ADMM polish (vs. draft): Carefully analyze the question to identify the primary topic, focusing on the core subject matter and ignoring any incidental details or keywords, and select the corresponding ~~capital letter~~ **topic label** from the label mapping.

### F: gradient insertion (66 accepted children from the 2-longer variant)

**largest effect: +96.0 answers vs. parent** (refined child from ADMM round 18; yahoo_F_mlgrad_s4)

- parent: Carefully determine the question's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-longer variant (2 gradient-chosen insertions, before ADMM): Carefully **create** determine the **sentiment** question's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- accepted child (vs. parent): Carefully ~~determine~~ **classify each** the **sentiment** question's main topic from the label mapping, ~~ignore~~ **ignoring** incidental ~~details, and~~ **details. For** output only the ~~corresponding~~ **category** capital letter.

**median effect: +24.0 answers vs. parent** (refined child from ADMM round 16; yahoo_F_mlgrad_s6)

- parent: Carefully determine the question's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-longer variant (2 gradient-chosen insertions, before ADMM): Carefully **remove** determine the question's main topic from the label mapping, ignore incidental details, and output only the corresponding ~~capital~~ **capital-letter** letter.
- accepted child (vs. parent): Carefully **remove** determine the question's main topic from the label ~~mapping, ignore incidental~~ **mapping. This entails** details, and output only the corresponding ~~capital~~ **capital-letter** letter.

### G: gated (87 accepted children)

**largest effect: +99.0 answers vs. parent** (spare slots opened: 1, original tokens deleted: 0, length change +1 tokens; yahoo_G_gated_s7)

- parent: Carefully determine the question's intent by topic from the label mapping, ignoring incidental details, and output only the corresponding label letter.
- accepted child: Carefully ~~determine~~ **classify** the **following** question's intent by topic from the label mapping, ignoring incidental details, and output only the corresponding label letter.

**median effect: +28.0 answers vs. parent** (spare slots opened: 1, original tokens deleted: 0, length change +1 tokens; yahoo_G_gated_s8)

- parent: Carefully determine the question's main topic, the label mapping, ignore incidental details, and output only the label corresponding to the letter.
- accepted child: Carefully determine the **correct** question's main topic, the label mapping, ignore incidental details, and output only the label corresponding to the letter.

## GoEmotions

### E: reflect-then-polish (11 accepted children where the polish changed the draft)

**largest effect: +7.3 F1 points vs. GEPA's parent** (draft alone: n/a; goemotions_E_polish_fair_s0)

- GEPA parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- reflection draft: Carefully ~~decide whether~~ **evaluate** the comment ~~expresses~~ **for** the candidate emotion ~~from the label mapping, ignore~~ **based on context and tone, prioritizing expressions of emotion over** incidental details, and output only the corresponding capital letter.
- after ADMM polish (vs. draft): Carefully evaluate the comment for the candidate emotion based on context and ~~tone,~~ **evidence,** prioritizing expressions of emotion over incidental details, and output only the corresponding capital letter.

**median effect: +3.1 F1 points vs. GEPA's parent** (draft alone: +7.4; goemotions_E_polish_fair_s6)

- GEPA parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- reflection draft: Carefully ~~decide whether~~ **assess** the comment ~~expresses~~ **for** the candidate emotion ~~from the label mapping, ignore incidental~~ **based on context and tone, ignoring irrelevant** details, and output only the corresponding capital letter.
- after ADMM polish (vs. draft): Carefully assess the comment for the candidate emotion based on context and tone, ignoring irrelevant details, and output only the corresponding ~~capital letter.~~ **emotion label.**

### F: gradient insertion (16 accepted children from the 2-longer variant)

**largest effect: +8.4 F1 points vs. parent** (refined child from ADMM round 1; goemotions_F_mlgrad_s1)

- parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-longer variant (2 gradient-chosen insertions, before ADMM): Carefully decide whether the **person** comment expresses the **emotion** candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- accepted child (vs. parent): Carefully decide whether the **person** comment expresses the ~~candidate~~ **emotion of** emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.

**median effect: +3.8 F1 points vs. parent** (refined child from ADMM round 19; goemotions_F_mlgrad_s9)

- parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-longer variant (2 gradient-chosen insertions, before ADMM): Carefully decide whether the comment **posted** expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital **abbreviation** letter.
- accepted child (vs. parent): Carefully decide whether the comment ~~expresses the~~ **posted contains a** candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital **abbreviation** letter.

### G: gated (32 accepted children)

**largest effect: +8.1 F1 points vs. parent** (spare slots opened: 1, original tokens deleted: 0, length change +1 tokens; goemotions_G_gated_s4)

- parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- accepted child: Carefully decide whether the comment expresses the ~~candidate~~ **candidate's** emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.

**median effect: +1.8 F1 points vs. parent** (spare slots opened: 0, original tokens deleted: 0, length change +0 tokens; goemotions_G_gated_s1)

- parent: Carefully decide whether the comment expresses the candidate emotion from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- accepted child: Carefully decide whether the comment expresses ~~the~~ **a** candidate emotion from the label mapping, ignore incidental details, and output only the corresponding ~~capital~~ **emotion** letter.

