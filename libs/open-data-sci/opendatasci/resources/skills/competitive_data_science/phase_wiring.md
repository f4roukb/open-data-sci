# Competitive Data Science — Phase Wiring

When signals from a later phase challenge the assumptions of an earlier one, looping back is part of the playbook rather than a sign of failure. The most common patterns:

- **Consistent CV-LB gap that diverges after introducing new features** → return to Phase 4's review; if unresolved, return to Phase 2 to redesign the CV scheme
- **Adversarial validation easily separates train from test** → return to Phase 1 to inspect the responsible features and to Phase 4 to engineer around them (or remove them)
- **A single feature dominates importance with implausibly high signal** → return to Phase 4 to trace feature provenance for leakage before trusting any downstream metric
- **Ensemble does not improve over the best single model** → return to Phase 5 to build genuinely different model families, or to Phase 4 to construct alternative feature subsets
- **Hyperparameter tuning yields large CV gains that do not appear on LB** → return to Phase 2 (CV scheme may not reflect test) or Phase 4 (features may be CV-specific)
- **Final-week consolidation reveals brittleness in the pipeline** → reserve compute for stability over additional features and return to Phase 0 to harden the harness
- **Mid-competition shared insight from the discussion forum changes the picture** (e.g. a documented leak, a previously unknown grouping structure) → revisit Phase 1 to incorporate the new understanding, then re-validate Phase 2 and Phase 4 in light of it
- **Distribution shift detected when comparing final-submission predictions to the training target distribution** → return to Phase 1 (re-examine differences) and Phase 2 (verify CV scheme reflects them)
- **CV variance per fold is large enough to obscure improvements** → return to Phase 2 to consider repeated CV, more folds, or a fold structure better aligned with the test distribution

The loops are bounded by time: late in the campaign, the cost of redesigning validation or pruning load-bearing features is high, and Phase 8's diversification across CV and LB is the pragmatic hedge against decisions that can no longer be unwound.

## Metadata

- parent domain: competitive_data_science
