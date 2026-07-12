# Deep Learning — Flax NNX: Defining Models

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

## Part of

- `deep_learning` — the deep learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
