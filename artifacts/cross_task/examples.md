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

## D: multi-length, LM insertion (Phase 7; AG News with letters and emotion)

### AG News, 2-shorter variant (47 accepted)

**largest effect: +36 answers of 512 vs. parent** (ADMM round 16; agnews_multilength_s3)

- parent: Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-shorter variant (before ADMM): Carefully determine the article's main topic from the label ~~mapping,~~ **mapping** ignore incidental ~~details,~~ **details** and output only the corresponding capital letter.
- accepted child (vs. parent): Carefully ~~determine~~ **choose** the article's main topic from the label ~~mapping, ignore incidental details,~~ **mapping and classify it** and output only the corresponding capital letter.

**median effect: +31 answers of 512 vs. parent** (ADMM round 1; agnews_multilength_s3)

- parent: Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-shorter variant (before ADMM): Carefully determine the article's main topic from the label ~~mapping,~~ **mapping** ignore incidental ~~details,~~ **details** and output only the corresponding capital letter.
- accepted child (vs. parent): Carefully determine the article's main topic from the label ~~mapping, ignore incidental details,~~ **mapping and provide details** and output only the corresponding capital letter.

### AG News, 2-longer variant (7 accepted)

**largest effect: +32 answers of 512 vs. parent** (ADMM round 15; agnews_multilength_s6)

- parent: Carefully determine the article's main topic from the label mapping, ignore incidental details, and output only the corresponding label name and letter.
- 2-longer variant (before ADMM): Carefully determine the **most** article's main topic from the label mapping, ignore incidental details, and output only the corresponding label name and **the** letter.
- accepted child (vs. parent): ~~Carefully~~ **###fully** determine the ~~article's~~ **most likely's** main topic from the label mapping, ignore incidental details, and output only the corresponding label name and **the** letter.

**median effect: -10 answers of 512 vs. parent** (ADMM round 9; agnews_multilength_s6)

- parent: Carefully determine the article main topic from the label mapping for it. Output only the corresponding capital letter.
- 2-longer variant (before ADMM): Carefully determine the article main topic from the label mapping for it. Output only the corresponding **label** capital **label** letter.
- accepted child (vs. parent): Carefully determine the article main topic from the label mapping for it. Output only the corresponding **label** capital **letter** letter.

### Emotion, 2-shorter variant (39 accepted)

**largest effect: +14 answers of 512 vs. parent** (ADMM round 10; emotion_multilength_s8)

- parent: Carefully determine the main emotion the writer expresses from the label mapping, ignore incidental details, and output only the corresponding capital letter.
- 2-shorter variant (before ADMM): Carefully determine the main emotion the writer expresses from the label mapping, ignore incidental ~~details,~~ and output only the corresponding capital letter.
- accepted child (vs. parent): Carefully determine the main emotion the ~~writer~~ **user** expresses from the label mapping, ignore incidental ~~details,~~ and output only the corresponding capital letter.

**median effect: -5 answers of 512 vs. parent** (ADMM round 18; emotion_multilength_s0)

- parent: Carefully determine the main emotion emotion the writer expresses in the the paragraph. (Ignore incidental details, and then only the main one letter.
- 2-shorter variant (before ADMM): Carefully determine the main emotion emotion the writer expresses in the ~~the~~ paragraph. (Ignore incidental ~~details,~~ **details** and then only the main one letter.
- accepted child (vs. parent): Carefully determine the main emotion emotion the writer expresses in the ~~the~~ paragraph. (Ignore ~~incidental details,~~ **the details** and then only the main one letter.

### Emotion, 2-longer variant (9 accepted)

**largest effect: +18 answers of 512 vs. parent** (ADMM round 1; emotion_multilength_s2)

- parent: Carefully determine the main emotion the writer expresses. The label mapping, ignore accidental and output only the corresponding capital letter.
- 2-longer variant (before ADMM): Carefully determine the main **emotion** emotion the writer expresses. The label mapping, ignore accidental and output only the corresponding capital **label** letter.
- accepted child (vs. parent): Carefully determine the main emotion **emotion** the ~~writer~~ **text** expresses. The label mapping, ignore accidental and output only the corresponding capital **label** letter.

**median effect: +4 answers of 512 vs. parent** (ADMM round 7; emotion_multilength_s5)

- parent: Carefully determine the main emotion the writer expresses from the label mapping provided and determine the output only the corresponding label for the emotion.
- 2-longer variant (before ADMM): Carefully determine the main emotion **emotion** the **user** writer expresses from the label mapping provided and determine the output only the corresponding label for the emotion.
- accepted child (vs. parent): Carefully determine the main emotion **emotion** the **user** writer expresses from the label mapping provided and determine the output ~~only~~ **with** the corresponding label for the emotion.

