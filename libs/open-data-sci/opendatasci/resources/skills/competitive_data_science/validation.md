# Competitive Data Science — Phase 2: Validation Strategy

**Planning**
- The validation scheme is the single most leverage-laden decision in a competition and is best designed before serious modelling begins; a strong local CV that correlates tightly with leaderboard score is more valuable than any single model improvement
- The plan starts from how the test set was constructed (time cutoff, geographic split, entity holdout, draw from a different distribution) and replicates that structure in CV
- Persistence of fold assignments across the campaign ensures every model's OOF predictions are directly comparable downstream

**Knowledge & Information**
- CV splits should mirror the test split: time-ordered for temporal holdouts, group-aware when rows share an identity (user, device, session, location), stratified for rare outcomes, nested combinations when several conditions apply
- The public leaderboard is a noisy signal on a small sample; private leaderboards routinely reshuffle relative to public — trusting a robust local CV over public-LB chasing is the default of strong competitors, though when the public sample is large, the CV-LB correlation has been demonstrably tight across submissions, or the test split is known to be drawn from the same distribution as train, LB carries genuine information worth weighing alongside CV
- The gap between local CV and public LB across submissions is itself a diagnostic: a consistent offset is acceptable, an inconsistent one signals that the CV scheme does not reflect the test distribution
- Repeated CV (multiple seeds, multiple shuffles) reduces noise in the validation estimate at the cost of compute and is worth running for final model selection rather than every iteration
- Out-of-fold (OOF) predictions are a free byproduct of CV that enables stacking, post-processing calibration, and error analysis — saving them by default removes friction later
- Adversarial validation results from Phase 1 directly inform CV design: when a feature separates train from test, validating on a fold drawn from the train distribution will not reflect how the model performs on the test distribution
- The public LB is computed on a small fraction of the test set (often 20–50%); its score variance is large enough that small public-LB movements between submissions often reflect noise rather than improvement

**Tricks**
- Constructing a fold structure that explicitly mimics observed train/test differences (e.g. using the most recent time slice as the validation fold when the test set is the future) is more reliable than relying on stratification alone
- A small "blend holdout" — a fold reserved for selecting ensemble weights and never used during base model training or hyperparameter search — preserves the integrity of ensembling decisions
- Persisting fold assignments to disk and re-using them across every model in the campaign keeps OOF predictions directly comparable and enables clean stacking later
- Sample-weighted CV, where validation weights reflect the test distribution (e.g. up-weighting recent observations under temporal shift), can produce CV scores that track LB more tightly than equal-weight CV

**Review** (close before moving on)
- Does the CV scheme replicate the test set construction? If not, redesign before any feature or model work proceeds
- Is the CV variance across folds small enough that meaningful improvements will be distinguishable from noise?
- Has at least one baseline submission anchored the CV-LB correspondence? If not, defer further work until it has
- Have OOF predictions and fold assignments been persisted so they can be reused throughout the campaign?

## Metadata

- parent domain: competitive_data_science
