# Deep Learning — Training Loop Design

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

## Part of

- `deep_learning` — the deep learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
