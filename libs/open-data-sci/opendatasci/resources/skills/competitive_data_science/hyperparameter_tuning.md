# Competitive Data Science — Phase 6: Hyperparameter Tuning

**Planning**
- A tuning budget — number of trials, wall-clock cap — set before search starts prevents the common failure mode of tuning expanding to fill all available time with diminishing returns past the first 50–100 trials per model
- The search strategy matches the budget: Bayesian optimisation for large continuous spaces, random search for moderate spaces, grid search for small or discrete spaces where exact coverage and reproducibility of the search matter more than efficiency
- Running the search on a representative stratified subsample and validating the winner on the full dataset is the standard speed-cost tradeoff

**Knowledge & Information**
- The same CV scheme used for evaluation should be used for tuning to keep estimates consistent and comparable
- Reporting the distribution of CV scores across configurations (not only the best) characterises sensitivity and reveals whether the winner is a stable optimum or a lucky tail draw
- For boosted trees, the high-leverage hyperparameters are typically learning rate, number of leaves / max depth, min child weight / min data in leaf, L1/L2 regularisation, feature/row subsampling fractions, and early-stopping rounds
- For neural networks, learning rate and learning-rate schedule, batch size, optimiser choice, weight decay, dropout, and augmentation strength dominate; architecture changes often matter less than these
- Successive halving and Hyperband prune unpromising configurations early and routinely cut search cost by 3–10× while typically preserving the winning configuration; the pruning can occasionally drop late-bloomers whose early loss is high but whose converged optimum is strong, so a full evaluation budget on a small confirmation slate is a reasonable hedge when search outcomes are surprising
- Winning hyperparameters from past competitions on the same data modality often transfer as strong defaults and reduce the search to narrow refinement

**Tricks**
- Persisting the full search history (every trial's parameters, score, and intermediate state) enables resuming after interruptions, inspecting parameter importance, and warm-starting future searches in the same competition
- A coarse-to-fine schedule — a wide search with few trials, then a narrow search around the best region — is more efficient than a single broad search
- Tuning multiple models in parallel using independent studies, then ensembling, often yields more total signal than exhaustively tuning a single model

## Metadata

- parent domain: competitive_data_science
