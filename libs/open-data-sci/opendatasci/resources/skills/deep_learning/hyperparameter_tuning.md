# Deep Learning — Hyperparameter Tuning

**High-leverage hyperparameters**
- Learning rate is almost always the single most impactful hyperparameter; a log-uniform search between 1e-5 and 1e-2 is a reasonable starting range for Adam-family optimisers, wider for SGD
- Batch size affects both optimisation dynamics (smaller batches add noise that can help generalisation) and computational efficiency (larger batches utilise hardware better); typical values are 32, 64, 128, 256 — the interaction between batch size and learning rate (linear scaling rule) deserves attention
- Weight decay, dropout rate, and the number of layers/units per layer are the next tier; search these after the learning rate is approximately right
- Learning rate schedule parameters (warmup steps, decay rate, minimum learning rate) are often set by convention rather than searched: warmup over 5–10% of total training steps, decay to 1e-6 or 1e-7

**Search strategy**
- Start with a small number of epochs (1–3) to do a coarse learning rate sweep; this surfaces obviously bad configurations without wasting compute
- Use `optuna` for Bayesian hyperparameter search over continuous and categorical spaces; it prunes unpromising trials early via median stopping or Hyperband and supports parallel trials
- Run the search on a representative subsample of the data to cut per-trial cost; validate the winning configuration on the full dataset before committing
- Fix random seeds across trials so differences in score reflect hyperparameter choices, not initialisation variance

## Metadata

- parent domain: deep_learning
