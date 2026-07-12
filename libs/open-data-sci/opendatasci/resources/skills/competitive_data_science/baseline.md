# Competitive Data Science — Phase 3: Baseline Model

**Planning**
- The point of the baseline is to compress the end-to-end pipeline into something runnable — data load, validation split, training, prediction, submission file — before optimising any single step
- A baseline submitted within the first one or two days establishes the floor and reveals pipeline bugs while they are still cheap to fix
- The baseline doubles as a measurement device: the value of every later improvement is expressed relative to this anchor

**Knowledge & Information**
- A naive baseline (target mean, mode, last-value carry-forward, simple rule keyed off the most predictive column) is the floor against which all subsequent complexity is measured — sometimes it is surprisingly hard to beat, which is itself a strong signal about the problem
- For tabular data, a gradient-boosting model with sensible defaults trained on raw features is the natural first real baseline; for text, TF-IDF with logistic regression or a small distilled transformer; for image, a pre-trained backbone with a linear head; for time series, a seasonal-naive or simple boosted-tree lag model
- Submitting the baseline locks in the CV-LB correspondence and provides the reference point for every later experiment
- The baseline's per-fold variance characterises noise floor — improvements smaller than this variance are unlikely to be real

**Tricks**
- Logging baseline metrics across all CV folds (mean, std, per-fold scores) characterises stability and informs how much variance later improvements need to overcome
- Storing OOF predictions and feature importances from the baseline produces an immediate map of which features matter and where the model is uncertain — both feed directly into Phase 4
- A baseline ablation (one feature removed at a time, scored against CV) is cheap and surfaces leakage candidates and dead features before serious feature work begins

## Part of

- `competitive_data_science` — the phased competition playbook this phase belongs to; load it for the full campaign overview, phase sequencing, and the cross-phase loop-back wiring.
