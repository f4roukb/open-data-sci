# Deep Learning — scikit-learn MLP

Use `MLPClassifier` / `MLPRegressor` when:
- The task is classification or regression on tabular features and a shallow network (1–3 hidden layers, hundreds of units) is plausible given data size
- You need sklearn pipeline compatibility (transformers, cross-validation, grid search) and the overhead of JAX is not justified
- You want a quick neural-network baseline without leaving the sklearn ecosystem

Monitor training loss via `loss_curve_` after fitting. Use early stopping (`early_stopping=True`) with a validation fraction to avoid overfitting. For anything beyond shallow feedforward networks — convolutional layers, recurrent layers, attention, custom loss functions, fine-grained training control — use the JAX stack instead.

## Part of

- `deep_learning` — the deep learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
