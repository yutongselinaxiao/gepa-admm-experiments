# Objective audit

- Unique candidates: 13
- Best-accuracy candidate: proposal_9
- Lowest-loss candidate: proposal_12
- Same top candidate: False
- Pearson correlation (loss versus accuracy): -0.2082
- Spearman rank correlation (loss versus accuracy): -0.4133
- Pairwise ranking disagreements: 23 / 78
- Decision gate: `treat_label_loss_as_diagnostic_due_to_top_candidate_mismatch`

Lower loss should generally accompany higher accuracy, so the expected correlation is negative. Pairwise disagreements are retained with representative validation examples in `objective_audit.json`.
