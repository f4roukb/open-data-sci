# Machine Learning — Evaluation & Diagnostics

- Aggregate metrics can hide a lot; slicing by relevant subgroups, prediction ranges, or time periods often reveals where a model underperforms in ways the headline number conceals
- Error analysis — directly examining mispredictions (false positives, false negatives, worst residuals, confused classes) — is often more informative than metrics alone; patterns in where a model fails point directly at what to fix
- For classification, the decision threshold affects the precision-recall tradeoff and should be chosen deliberately based on the relative cost of false positives versus false negatives, not left at a default
- Probability calibration matters when predicted scores are used as actual probability estimates rather than just rankings
- Residual analysis for regression surfaces systematic patterns — heteroscedasticity, non-linearity, outlier influence — that aggregate error metrics don't capture
- Cross-validation spread (mean ± std across folds) characterises how stable performance is, not just what the best-case number is

## Part of

- `machine_learning` — the machine learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
