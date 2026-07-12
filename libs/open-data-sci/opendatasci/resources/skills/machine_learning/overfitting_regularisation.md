# Machine Learning — Overfitting & Regularisation

- The gap between training and validation performance is the primary diagnostic for overfitting; a model that performs well on training data but poorly on held-out data has memorised rather than generalised
- Overfitting risk increases with model capacity relative to data volume — more parameters, more trees, deeper networks all have higher capacity and require more data or more regularisation to generalise
- Regularisation techniques (penalising model complexity, limiting depth, adding noise during training, early stopping) are the primary tools for closing the train/val gap; the right form depends on the model family
- Learning curves (performance as a function of training set size) are a useful diagnostic: poor performance that improves with more data suggests a data problem; a persistent train/val gap that doesn't close suggests a regularisation or capacity problem
- Underfitting — where even training performance is poor — points in the opposite direction: the model may lack the capacity or the features to capture the signal

## Part of

- `machine_learning` — the machine learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
