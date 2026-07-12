# Deep Learning — Regularisation and Overfitting

The gap between training and validation performance is the primary diagnostic for overfitting, and the right regularisation strategy depends on the architecture, data volume, and the specific manifestation of the overfit.

**Core regularisation techniques**
- Dropout (`nnx.Dropout`) randomly zeros activations during training, forcing the network to learn redundant representations; typical rates range from 0.1 to 0.5, with higher rates for larger models or smaller datasets; remember to disable dropout at evaluation time (`deterministic=True`)
- Weight decay (via `optax.adamw` or explicit L2 penalty) penalises large weights and acts as a smoothness prior; values between 1e-4 and 1e-1 are typical, with larger values for models that overfit aggressively
- Early stopping — monitoring validation loss and stopping training when it stops improving — is the simplest and most reliable regulariser; patience (number of epochs without improvement before stopping) should be large enough to survive temporary plateaus
- Batch normalisation and layer normalisation have an implicit regularising effect through noise injection (batch statistics vary per mini-batch); this effect diminishes with larger batch sizes
- Label smoothing (replacing hard 0/1 targets with soft targets like 0.05/0.95) prevents the model from becoming overconfident and improves calibration, particularly for classification with noisy labels

**Data augmentation**
- For images: random crops, horizontal flips, colour jitter, cutout, and mixup are the standard augmentation vocabulary; implement as NumPy/JAX transformations applied per batch during data loading
- For tabular data: noise injection (Gaussian noise on continuous features), feature masking (randomly zeroing features), and mixup between training examples can regularise when data is scarce
- For sequences: random masking, token dropping, time warping, and window cropping depending on the modality

**Diagnosing capacity problems**
- When training loss is high and does not decrease: the model lacks capacity (increase width or depth), the learning rate is too low, or the data preprocessing has a bug — check data first
- When training loss is low but validation loss is high: overfitting — apply or increase regularisation, add data augmentation, or reduce model capacity
- When both losses plateau at a mediocre level: the feature representation may be insufficient, the architecture may be mismatched to the data structure, or the learning rate schedule may need adjustment
- Learning curves (validation performance as a function of training set size) distinguish data-limited regimes from model-limited regimes and guide whether to invest in more data or more architecture

## Part of

- `deep_learning` — the deep learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
