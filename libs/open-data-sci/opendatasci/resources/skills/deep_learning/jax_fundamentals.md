# Deep Learning — JAX Fundamentals

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

## Metadata

- parent domain: deep_learning
