# Objective audit

- Unique candidates: 12
- Best-accuracy candidate: proposal_11
- Lowest-loss candidate: proposal_1
- Same top candidate: False
- Pearson correlation (loss versus accuracy): -0.8160
- Spearman rank correlation (loss versus accuracy): -0.7881
- Pairwise ranking disagreements: 12 / 66
- Decision gate: `treat_label_loss_as_diagnostic_due_to_top_candidate_mismatch`

Lower loss should generally accompany higher accuracy, so the expected correlation is negative. Pairwise disagreements are retained with representative validation examples in `objective_audit.json`.
