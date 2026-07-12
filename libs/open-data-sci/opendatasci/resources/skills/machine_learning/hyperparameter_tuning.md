# Machine Learning — Hyperparameter Tuning

- The search strategy should match the budget: exhaustive search is only feasible for small spaces; random search and Bayesian optimisation cover large spaces more efficiently
- The wall-clock cost of a search is the product of configurations evaluated and the cost of each evaluation; on large datasets both factors deserve explicit management — run the search on a carefully constructed representative subsample (stratified by class and any structural variable such as time period or entity) to cut per-evaluation time dramatically while preserving most of the directional signal; techniques like successive halving and Hyperband go further by pruning unpromising configurations early rather than running every trial to completion; always re-train or re-evaluate the winning configuration on the full dataset before committing to it
- The same cross-validation strategy used for model evaluation should be used during tuning to keep estimates consistent
- Reporting the distribution of CV scores across configurations, not just the best, gives a clearer picture of sensitivity to hyperparameter choices
- All stochastic components should have fixed random seeds to ensure results are stable across runs

## Part of

- `machine_learning` — the machine learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
