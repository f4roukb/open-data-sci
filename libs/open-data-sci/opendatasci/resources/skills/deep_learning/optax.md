# Deep Learning — Optax: Optimisation and Schedules

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

## Metadata

- parent domain: deep_learning
