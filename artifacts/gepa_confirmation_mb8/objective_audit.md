# Objective audit

- Unique candidates: 12
- Best-accuracy candidate: proposal_8
- Lowest-loss candidate: proposal_11
- Same top candidate: False
- Pearson correlation (loss versus accuracy): -0.6174
- Spearman rank correlation (loss versus accuracy): -0.5965
- Pairwise ranking disagreements: 20 / 66
- Decision gate: `treat_label_loss_as_diagnostic_due_to_top_candidate_mismatch`

Lower loss should generally accompany higher accuracy, so the expected correlation is negative. Pairwise disagreements are retained with representative validation examples in `objective_audit.json`.
