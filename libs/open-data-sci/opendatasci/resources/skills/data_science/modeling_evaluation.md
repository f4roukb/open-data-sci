# Data Science — Modeling & Evaluation

- A naive baseline (mean/mode predictor, last-value carry-forward, a simple rule) makes the value of a model concrete — sometimes the baseline is surprisingly hard to beat, which is itself informative
- Model choice should be driven by the problem structure, not by a default preference for a particular algorithm: consider the full spectrum from linear models (interpretable, well-regularised) through tree-based ensembles (robust to feature scale, capture interactions) to neural networks (high capacity, need volume and tuning) — the right family depends on signal strength, data volume, interpretability needs, and the nature of the decision boundary or regression surface
- Metric choice should reflect the real objective and data distribution; accuracy misleads on imbalanced problems, RMSE penalises large errors disproportionately, percentage-based errors behave badly near zero — each metric embeds assumptions worth making explicit
- Split strategy encodes assumptions about how the model will be used: stratified splits for class balance, time-ordered splits when temporal structure exists, group-aware splits when rows share an entity; the wrong strategy produces optimistic numbers that don't hold
- Evaluating on a single held-out set can be noisy; cross-validation spread gives a better picture of how stable performance is across different data slices
- Slicing metrics by relevant subgroups or prediction ranges often reveals where a model underperforms in ways aggregate numbers conceal
- When combining multiple models, diversity of predictions is what drives ensemble gains — architecturally similar models (e.g., XGBoost, LightGBM, and CatBoost) are likely to produce highly correlated outputs and make the same mistakes, so blending them yields little improvement; meaningful gains come from ensembling models from different families (tree-based, linear, neural) or models trained on different feature sets or subsets of the data

## Part of

- `data_science` — the general data science skill domain this belongs to; load it for the full map of skills and when to reach for each.
