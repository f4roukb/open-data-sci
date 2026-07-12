# Competitive Data Science — Phase 4: Feature Engineering

**Planning**
- A feature plan starts from explicit hypotheses about signal sources (relational structure, temporal context, interactions, domain knowledge) rather than mechanical generation of every possible transformation
- Features cheap to compute and individually testable (each evaluated against the same CV scheme) make iteration fast and attribution clear
- A per-iteration feature budget — add N features, evaluate, prune — prevents accumulation of dead weight and keeps the feature set interpretable
- The feature engineering plan is the longest phase in most competitions; structuring it as a sequence of small, evaluable batches makes progress visible and prevents the search from going stale

**Knowledge & Information**
- Competition datasets frequently reward entity-level aggregations: group-by statistics (mean, std, min, max, count, nunique, skew, median) computed across users, sessions, locations, or time windows encode relational structure that row-level features miss
- Target encoding (mean of the target per category level, smoothed against the global mean) is consistently effective for high-cardinality categoricals but must be computed within each CV fold to avoid leakage
- Lag features, rolling statistics, expanding-window aggregations, exponentially weighted means, and time-since-event features form the core vocabulary for temporal datasets; their window sizes are hyperparameters worth searching
- Interaction features (products, ratios, differences, polynomial terms) between top-importance columns often outperform any single transformation; the cheap heuristic is to interact the top-K features from the baseline's importance ranking
- Frequency encoding (count of occurrences of each category level) is a near-free, leakage-safe alternative to one-hot for high-cardinality columns
- Cyclic encodings (sine/cosine of hour, day-of-week, month) preserve continuity at the boundaries that integer encoding does not natively express; the benefit is most pronounced for linear and neural models, while tree-based models can recover the modular structure through multiple splits on the integer-encoded feature when enough data is available
- For text, character-level and word-level n-grams, length statistics, sentiment scores, and pre-trained sentence embeddings each capture different aspects of the signal and ensemble well
- For image, traditional descriptors (HOG, colour histograms, LBP) and pre-trained backbone embeddings complement each other when compute is constrained
- Categorical features with cardinality in the thousands to millions typically respond better to target encoding, hashing, or learned embeddings than to one-hot; the right choice depends on the model family and the data volume
- Feature provenance matters: any feature computed using information unavailable at prediction time silently inflates CV and LB scores — tracing the construction of every feature against the data timeline is the only defence

**Tricks**
- Permutation importance and SHAP values on a trained baseline give a more reliable feature ranking than impurity-based importance, which biases toward high-cardinality features; impurity-based importance remains useful as a near-zero-cost first pass, particularly when many features need a directional ranking quickly or when permutation/SHAP would be prohibitively expensive
- Target encoding with K-fold nested inside the outer CV is the standard leak-safe pattern; failing to nest is the single most common source of silent CV-LB gaps in competition pipelines
- Forward feature selection by greedy CV gain is expensive but surfaces the truly load-bearing subset; backward elimination starting from the full feature set is faster and often sufficient
- A "control" feature — pure random noise added to the feature set — calibrates how much importance is attributable to chance; any real feature ranking below it is a candidate for pruning
- Re-running adversarial validation after adding new features detects features that encode the train/test split itself
- Storing the feature engineering as a pure transformation function (fit on train, applied to any split) prevents training/inference skew and makes leak-safety easier to audit

**Review** (close before moving on)
- Does the CV improvement from new features hold on a freshly seeded split? If not, suspect overfitting to fold structure — return to Phase 2 to evaluate CV variance and possibly redesign
- Has adversarial validation been re-run after the new feature set? If new features separate train from test more easily than before, those features encode the split — return to Phase 1 to inspect them and consider removal
- Are feature importances dominated by a single feature with implausibly high signal? Suspect leakage — trace the feature's construction against the data timeline before trusting any downstream metric
- Is the CV-LB gap consistent with the pre-feature-work baseline? A sudden divergence is a leak signal or a CV scheme problem — if persistent, return to Phase 2
- Has redundancy been pruned? Highly correlated features add noise without value and slow training; a brief correlation review at the end of the phase pays off in later iteration speed

## Part of

- `competitive_data_science` — the phased competition playbook this phase belongs to; load it for the full campaign overview, phase sequencing, and the cross-phase loop-back wiring.
