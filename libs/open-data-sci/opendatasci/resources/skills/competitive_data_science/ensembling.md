# Competitive Data Science — Phase 7: Ensembling & Stacking

**Planning**
- The ensembling plan starts from the set of diverse base models built across Phase 5 rather than from squeezing a final percentage point out of any single model
- A simple weighted average is the natural first ensemble; stacking is the next step when base model errors are uncorrelated enough to support a meta-learner
- Ensemble selection and weighting are performed on a holdout that no base model has seen — typically a reserved blend fold or out-of-fold predictions — to avoid overfitting the blend

**Knowledge & Information**
- Diversity of predictions drives ensemble gains; two moderately strong models with uncorrelated errors outperform two strong models that make the same mistakes
- Ensembling models from the same family (multiple gradient-boosting implementations, multiple variants of the same neural architecture) produces highly correlated predictions, but their differences in growth policy (leaf-wise vs. level-wise), categorical handling, regularisation, or initialisation are large enough that small but consistent gains over the single best member are common — in tight competitions where every fraction of a metric point matters, within-family blends are worth keeping in the ensemble even when their marginal lift is modest
- The largest gains typically come from combining genuinely different families (gradient boosting + neural network + linear) or models trained on substantially different feature sets or data samples
- Stacking with a simple meta-learner (logistic regression, ridge, light boosted tree with few leaves) on out-of-fold predictions captures systematic differences between base models with low overfitting risk on the blend; more expressive meta-learners can still be appropriate when base models are many and their interactions are non-trivial, provided the meta-learner is itself validated on a fold disjoint from the one used to train it
- Weight optimisation via constrained optimisers (Nelder-Mead, simplex methods, gradient-based solvers under a simplex constraint) on held-out CV often improves over uniform averaging when base models have meaningfully different strengths; validating the optimised weights on a separate fold guards against overfitting the blend
- Geometric mean (averaging in log-space) and rank-averaging are alternatives to arithmetic mean that work better when predictions have heterogeneous scale or are used as rankings rather than probabilities

**Tricks**
- Adding a poorly tuned, low-capacity model from a different family (a small MLP alongside boosted trees) often improves the ensemble despite being individually weaker
- Correlation matrices of OOF predictions across base models reveal which models contribute genuine diversity and which are redundant; pruning redundant models stabilises the blend without hurting score
- Training base models on different feature subsets (full, A, B) and ensembling produces cheap diversity without new architectures
- Capping prediction range to the empirical target range, or clipping outliers, can yield small but reliable gains under metrics that penalise extreme errors
- Multi-level stacking (a second-stage stacker over first-stage stacker outputs) has won mature, ensemble-heavy competitions where many strong and architecturally varied base models exist, but the marginal return diminishes quickly with each added stage and the complexity overhead is high — single-level stacking with diverse base models captures most of the available signal in most settings, and the additional level is worth the cost mainly when the ensemble is already large and well-validated

**Review** (close before moving on)
- Does the ensemble CV exceed the best single-model CV by a margin that survives across seeds? If not, the ensemble is not contributing — return to Phase 5 to build genuinely different models, or to Phase 4 to construct alternative feature subsets
- Are the base model OOF predictions correlated above ~0.95? Diversity is insufficient — return to Phase 5 (different family) or Phase 4 (different feature subset)
- Is the public LB improvement consistent with the CV improvement? If not, the blend is overfitting OOF — return to Phase 2 to inspect CV structure
- Has the meta-learner been validated on a fold disjoint from the one used to fit base models? If not, the stacking estimate is optimistic

## Metadata

- parent domain: competitive_data_science
