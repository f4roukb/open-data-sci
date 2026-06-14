# Deep Learning Skill

## Library Stack

**Key documentation references:**
- JAX fundamentals (jit, grad, vmap, scan): https://docs.jax.dev/en/latest/
- Flax NNX (modules, state, lifting): https://flax.readthedocs.io/en/latest/nnx_basics.html
- Optax (optimisers, schedules, losses): https://optax.readthedocs.io/en/latest/

## When to Use Deep Learning

Deep learning is the right choice when:
- The data has spatial, sequential, or relational structure (images, text, audio, time series, graphs) that classical models cannot capture without extensive manual feature engineering
- Data volume is large enough that representation learning outperforms hand-crafted features — as a rough heuristic, thousands of samples for simple MLPs, tens of thousands for CNNs and RNNs, hundreds of thousands or more for transformers trained from scratch
- Transfer learning from a pre-trained checkpoint is available for the domain, substantially reducing the data and compute needed to reach strong performance
- Tabular data warrants a neural approach (MLP, TabNet, FT-Transformer) after gradient-boosting baselines have been tried and plateaued, or when learned embeddings for high-cardinality categoricals carry signal that encoding schemes miss

Prefer scikit-learn or gradient boosting (LightGBM, CatBoost, XGBoost) over deep learning when data is tabular and moderate in size, when interpretability and iteration speed matter more than squeezing out the last percentage point, or when the data volume is too small to support the capacity of a neural model without severe overfitting.

## scikit-learn MLP

Use `MLPClassifier` / `MLPRegressor` when:
- The task is classification or regression on tabular features and a shallow network (1–3 hidden layers, hundreds of units) is plausible given data size
- You need sklearn pipeline compatibility (transformers, cross-validation, grid search) and the overhead of JAX is not justified
- You want a quick neural-network baseline without leaving the sklearn ecosystem

Monitor training loss via `loss_curve_` after fitting. Use early stopping (`early_stopping=True`) with a validation fraction to avoid overfitting. For anything beyond shallow feedforward networks — convolutional layers, recurrent layers, attention, custom loss functions, fine-grained training control — use the JAX stack instead.

## JAX Fundamentals

JAX programs are built from pure functions transformed by a small set of composable primitives. Understanding these primitives and the constraints they impose is essential before writing any model code.

**Functional purity and side effects**
- JAX transformations (`jit`, `grad`, `vmap`, `scan`) require pure functions: given the same inputs, the function must return the same outputs with no observable side effects; in-place mutation of arrays, Python-level state changes, and I/O inside transformed functions will silently produce incorrect results or raise errors
- All randomness flows through explicit PRNG keys (`jax.random.key`); splitting a key into subkeys before each stochastic operation (dropout, initialisation, data augmentation) ensures reproducibility and correct behaviour under `jit` and `vmap`
- State (model parameters, optimiser state, batch-norm statistics, RNG keys) is passed explicitly as function arguments and returned as outputs rather than mutated in place — this is the central design difference from imperative frameworks

**Core transformations**
- `jax.jit` compiles a function via XLA for fast execution; the first call traces and compiles, subsequent calls with the same input shapes and dtypes hit the cache — shape-changing inputs trigger recompilation, so avoid variable-length sequences without padding
- `jax.grad` computes gradients of a scalar-valued function with respect to its first argument (or specified `argnums`); for auxiliary outputs alongside the gradient, use `jax.value_and_grad` with `has_aux=True`
- `jax.vmap` vectorises a function over a batch dimension, replacing explicit loops with efficient batched operations; use it to write per-example logic and let JAX handle batching
- `jax.lax.scan` replaces Python for-loops over sequential operations (RNN steps, iterative algorithms) with an XLA-compiled loop that is both faster and memory-efficient through automatic gradient checkpointing

**Array semantics**
- JAX arrays are immutable; "updates" produce new arrays (e.g. `x.at[i].set(v)` returns a new array rather than modifying `x`)
- Default dtype promotion in JAX follows its own rules, not NumPy's; float32 is the standard training dtype, and explicit dtype management avoids silent precision loss or promotion to float64
- JAX's NumPy API (`jax.numpy`) mirrors NumPy closely but not identically — in particular, out-of-bounds indexing clamps rather than raising, and some operations behave differently under `jit` when control flow depends on array values

## Flax NNX: Defining Models

Flax NNX is the module API for defining neural network architectures on JAX. It provides a Pythonic, mutable-object interface that handles the functional-purity requirements of JAX under the hood.

**Module basics**
- Subclass `nnx.Module` to define layers and models; parameters are declared as `nnx.Param` (or created implicitly by built-in layers like `nnx.Linear`, `nnx.Conv`, `nnx.BatchNorm`) and become part of the module's state
- Modules are mutable Python objects during construction and outside JIT; inside `nnx.jit`-wrapped functions, Flax NNX manages the functional transformation automatically — you write imperative code and NNX lifts it to pure functions for JAX
- Use `nnx.Rngs` to manage PRNG keys for initialisation, dropout, and other stochastic layers; pass an `nnx.Rngs` object at module construction and Flax handles key splitting across layers

**Built-in layers**
- `nnx.Linear` — dense layer; the fundamental building block for MLPs and projection heads
- `nnx.Conv` — convolution; supports arbitrary dimensionality via `kernel_size` and standard options (strides, padding, dilation, feature groups)
- `nnx.BatchNorm` — batch normalisation; tracks running statistics via `nnx.BatchStat` and requires `use_running_average` to switch between train and eval modes
- `nnx.LayerNorm` — layer normalisation; preferred over batch norm for small batches, sequence models, and transformers
- `nnx.Dropout` — inverted dropout; requires `deterministic=False` during training and an active RNG stream, switches to identity with `deterministic=True` at eval
- `nnx.Embed` — embedding table; maps integer indices to dense vectors, the entry point for categorical features and token-based inputs
- `nnx.MultiHeadAttention` — scaled dot-product multi-head attention

**State management**
- `nnx.state(model, nnx.Param)` extracts all trainable parameters as a nested pytree; `nnx.state(model, nnx.BatchStat)` extracts batch statistics — this separation enables clean update logic where only parameters receive gradients
- For serialisation, `nnx.state(model)` captures the full state which can be saved and restored
- When composing models from submodules, parameter namespacing follows the attribute hierarchy automatically

**Train/eval mode**
- Modules that behave differently during training versus inference (dropout, batch norm) are controlled by flags passed to their `__call__` method (`deterministic`, `use_running_average`) or set globally — always ensure the correct mode is active before each forward pass

## Optax: Optimisation and Schedules

Optax provides the optimiser, learning rate schedule, and gradient transformation pipeline for training JAX models.

**Optimiser selection**
- `optax.adamw` — Adam with decoupled weight decay; the default starting point for most deep learning tasks; weight decay acts as L2 regularisation without interfering with adaptive moment estimates
- `optax.adam` — standard Adam; suitable when weight decay is handled separately or not needed
- `optax.sgd` — SGD with optional momentum and Nesterov acceleration; can outperform Adam on well-tuned image classification and other tasks where the loss landscape is smooth and the learning rate schedule is carefully designed
- `optax.lamb` — layer-wise adaptive moments; scales to very large batch sizes for distributed training
- `optax.lion` — evolved optimiser; uses sign-based updates and tends to generalise well with lower memory than Adam

**Learning rate schedules**
- `optax.warmup_cosine_decay_schedule` — linear warmup followed by cosine decay; the most common schedule for transformer training and a strong default for any architecture
- `optax.cosine_decay_schedule` — cosine annealing without warmup; suitable when training is long enough that warmup is unnecessary
- `optax.linear_schedule` — linear interpolation between two values; useful for warmup phases or simple decay
- `optax.exponential_decay` — step-based exponential decay; commonly used with SGD for image classification
- `optax.piecewise_constant_schedule` — manual step-function schedule; useful when domain knowledge dictates specific rate changes at known training milestones
- Schedules are passed as the `learning_rate` argument to the optimiser; they receive the step count and return the current rate

**Gradient transformations**
- `optax.clip_by_global_norm` — clips gradients to a maximum global norm; essential for training RNNs, transformers, and any architecture prone to gradient explosion; a global norm of 1.0 is a common starting point
- `optax.chain` — composes multiple gradient transformations sequentially (e.g. clip, then scale by learning rate, then apply Adam); the standard way to build custom optimiser pipelines
- `optax.apply_every` — accumulates gradients over multiple steps before applying; simulates larger effective batch sizes when memory is constrained
- `optax.ema` — exponential moving average of parameters; used for maintaining a smoothed copy of weights for evaluation (Polyak averaging)

**Loss functions**
- `optax.softmax_cross_entropy_with_integer_labels` — classification with integer targets; numerically stable and avoids manual one-hot encoding
- `optax.softmax_cross_entropy` — classification with one-hot or soft targets
- `optax.sigmoid_binary_cross_entropy` — binary or multi-label classification
- `optax.l2_loss`, `optax.huber_loss`, `optax.squared_error` — regression losses with different outlier sensitivity profiles

**Optimiser state management**
- `optax.inject_hyperparams` wraps an optimiser to make hyperparameters (learning rate, weight decay) accessible and modifiable in the optimiser state — useful for logging the current learning rate or implementing custom schedule logic
- Optimiser state is a pytree that mirrors the parameter structure; it is initialised with `opt.init(params)` and updated with `opt.update(grads, opt_state, params)`

## Training Loop Design

The training loop ties JAX, Flax, and Optax together. Getting the structure right from the start prevents a class of bugs that are difficult to diagnose later.

**Standard loop structure**
- A training step function takes the model, optimiser state, a batch of data, and (when needed) an RNG key; it computes the forward pass, loss, and gradients, applies the optimiser update, and returns the updated model, updated optimiser state, and metrics — this function is the natural unit of `jit` compilation
- Use `jax.value_and_grad` with `has_aux=True` to compute the loss and gradients in a single pass while returning auxiliary outputs (per-example losses, logits, intermediate activations for logging)
- Apply `nnx.jit` (or `jax.jit`) to the training step function; this compiles it once and reuses the compiled version for every batch — ensure all inputs have static shapes to avoid recompilation
- An epoch loops over batches from the data loader, calls the compiled training step, and accumulates metrics; an outer loop iterates over epochs

**Data loading**
- JAX does not include a data loading pipeline; data preparation and batching are handled outside JAX using NumPy, Pandas, or any standard Python tooling
- Convert data to JAX arrays (`jnp.array`) at the batch level, not the dataset level — loading entire large datasets into device memory is often unnecessary and wasteful
- For datasets that fit in memory, a simple pattern is: shuffle indices at the start of each epoch, slice into batches, and convert each batch to `jnp.array` as it is consumed
- Ensure consistent batch sizes (pad the last batch if necessary) to avoid triggering JIT recompilation on the final batch of each epoch

**Logging and monitoring**
- Track training loss, validation loss, and the primary evaluation metric per epoch at minimum; per-batch training loss reveals learning dynamics (oscillation, divergence, plateaus) that per-epoch averages can conceal
- When using a learning rate schedule, log the current learning rate alongside loss to diagnose whether decay is too aggressive or too slow
- A sudden spike in training loss often indicates a learning rate that is too high, a data loading bug (corrupted batch), or numerical instability — investigate immediately rather than hoping the model recovers

**Reproducibility**
- Fix all random seeds: Python's `random.seed`, NumPy's `np.random.seed`, and the initial JAX PRNG key (`jax.random.key(seed)`)
- Deterministic data shuffling (seeded permutation of indices) ensures the same batch order across runs
- JAX's XLA compilation is deterministic given the same inputs and platform, but results may differ across hardware (CPU vs GPU) due to floating-point non-associativity in parallel reductions

## Architecture Selection

Architecture choice should be driven by the data modality, the nature of the prediction task, and the available data volume — not by a default preference for a familiar architecture.

**Tabular data**
- A 2–4 layer MLP with ReLU (or GELU) activations, batch normalisation or layer normalisation, and dropout is the standard neural baseline for tabular data; hidden dimensions between 64 and 512 depending on feature count and data volume
- Learned embeddings (`nnx.Embed`) for categorical features, concatenated with normalised continuous features, often outperform one-hot or ordinal encoding for high-cardinality columns
- For tabular tasks, gradient-boosted trees are usually the stronger baseline; the neural model's value is often as an ensemble component providing prediction diversity rather than as a standalone winner

**Sequences and time series**
- For short-to-medium sequences, 1D convolutions (`nnx.Conv` with appropriate kernel sizes) with residual connections capture local patterns efficiently and are faster to train than recurrent architectures
- GRU and LSTM cells process sequences step-by-step and naturally handle variable-length inputs; implement the recurrence with `jax.lax.scan` for efficient compiled execution rather than Python for-loops
- For long sequences where global context matters, self-attention (transformer) architectures are more expressive but scale quadratically with sequence length; for very long sequences, consider windowed or linear attention variants

**Images**
- Convolutional architectures (ResNet-style blocks using `nnx.Conv` + batch norm + residual connections) are the standard entry point; depth and width scale with data volume and image resolution
- When building from scratch with limited data, prefer shallower architectures with aggressive data augmentation over deep networks that overfit
- For transfer learning scenarios, load pre-trained weights into a Flax model and fine-tune the head (or the full model with a lower learning rate for pre-trained layers)

**Attention and transformers**
- The transformer block (multi-head self-attention + feedforward + layer norm + residual connections) is the dominant architecture for sequence modelling tasks with sufficient data
- Pre-norm (layer norm before attention and feedforward) tends to train more stably than post-norm, especially for deeper models
- Positional encoding (sinusoidal, learned, or rotary) is essential — without it, the attention mechanism is permutation-invariant and cannot distinguish token order

## Regularisation and Overfitting

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

## Hyperparameter Tuning

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

## Evaluation

The same evaluation principles from classical ML apply: appropriate split strategies, metric selection aligned with the real objective, and subgroup analysis to surface hidden failure modes.

**Neural-specific evaluation considerations**
- Ensure the model is in evaluation mode (dropout disabled, batch norm using running statistics) before computing any validation or test metrics — forgetting this is a common source of inconsistent results
- For classification, predicted probabilities from neural networks are often poorly calibrated; temperature scaling or Platt scaling on a held-out calibration set improves probability estimates when they will be used as actual probabilities rather than just rankings
- Averaging predictions across multiple random seeds (same architecture, different initialisations) reduces variance and gives a more stable performance estimate; the spread across seeds characterises how sensitive the result is to initialisation
- Test-time augmentation (averaging predictions over augmented copies of each test input) provides small but consistent gains for image tasks and sometimes for other modalities
- When ensembling neural models with classical models (gradient boosting, linear), the diversity of the neural model's errors relative to the classical model's errors is what drives ensemble gains — even a weaker neural model can improve an ensemble if its mistakes are uncorrelated