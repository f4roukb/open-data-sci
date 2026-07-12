# Machine Learning — Feature Engineering & Selection

- Preprocessing transformations need to be fit only on training data and applied consistently to every split; fitting on the full dataset before splitting leaks distributional information into evaluation
- For tabular data, useful transformations include datetime decomposition, lag and rolling features, interaction terms, and outlier treatment — the right choices are domain-dependent
- High-cardinality categoricals benefit from encoding strategies that don't naively explode dimensionality; the right approach depends on the model family and available data volume
- Feature magnitude matters for some model families and is irrelevant for others — scaling decisions should match the model's sensitivity
- More features is not always better: irrelevant or redundant features add noise and variance, can hurt distance-based models, and make models harder to interpret and debug; feature selection (variance-based filtering, correlation-based pruning, importance-based selection) is often worth doing before fitting complex models
- Training/inference skew — features computed differently at training time versus inference time — is a common source of silent degradation

## Metadata

- parent domain: machine_learning
