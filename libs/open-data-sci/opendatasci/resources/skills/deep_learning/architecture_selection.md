# Deep Learning — Architecture Selection

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

## Metadata

- parent domain: deep_learning
