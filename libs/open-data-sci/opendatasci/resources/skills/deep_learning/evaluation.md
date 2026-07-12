# Deep Learning — Evaluation

The same evaluation principles from classical ML apply: appropriate split strategies, metric selection aligned with the real objective, and subgroup analysis to surface hidden failure modes.

**Neural-specific evaluation considerations**
- Ensure the model is in evaluation mode (dropout disabled, batch norm using running statistics) before computing any validation or test metrics — forgetting this is a common source of inconsistent results
- For classification, predicted probabilities from neural networks are often poorly calibrated; temperature scaling or Platt scaling on a held-out calibration set improves probability estimates when they will be used as actual probabilities rather than just rankings
- Averaging predictions across multiple random seeds (same architecture, different initialisations) reduces variance and gives a more stable performance estimate; the spread across seeds characterises how sensitive the result is to initialisation
- Test-time augmentation (averaging predictions over augmented copies of each test input) provides small but consistent gains for image tasks and sometimes for other modalities
- When ensembling neural models with classical models (gradient boosting, linear), the diversity of the neural model's errors relative to the classical model's errors is what drives ensemble gains — even a weaker neural model can improve an ensemble if its mistakes are uncorrelated

## Metadata

- parent domain: deep_learning
