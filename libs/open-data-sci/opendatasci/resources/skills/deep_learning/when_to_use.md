# Deep Learning — When to Use Deep Learning

Deep learning is the right choice when:
- The data has spatial, sequential, or relational structure (images, text, audio, time series, graphs) that classical models cannot capture without extensive manual feature engineering
- Data volume is large enough that representation learning outperforms hand-crafted features — as a rough heuristic, thousands of samples for simple MLPs, tens of thousands for CNNs and RNNs, hundreds of thousands or more for transformers trained from scratch
- Transfer learning from a pre-trained checkpoint is available for the domain, substantially reducing the data and compute needed to reach strong performance
- Tabular data warrants a neural approach (MLP, TabNet, FT-Transformer) after gradient-boosting baselines have been tried and plateaued, or when learned embeddings for high-cardinality categoricals carry signal that encoding schemes miss

Prefer scikit-learn or gradient boosting (LightGBM, CatBoost, XGBoost) over deep learning when data is tabular and moderate in size, when interpretability and iteration speed matter more than squeezing out the last percentage point, or when the data volume is too small to support the capacity of a neural model without severe overfitting.

## Metadata

- parent domain: deep_learning
