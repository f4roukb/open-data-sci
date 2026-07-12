# Machine Learning — Splitting Strategy

- The split strategy encodes assumptions about the real-world prediction setting; getting it wrong produces metrics that don't generalise
- Temporal data requires time-ordered splits to avoid the model seeing the future during training — random shuffling of chronological data is a quiet but serious source of leakage
- When rows share an entity (user, customer, session, location), entity-aware splits prevent the model from memorising entities it will never see again
- Stratified splits preserve class distribution across folds and are particularly important when classes are rare
- The test set should be treated as a one-time evaluation; tuning against it invalidates it as an unbiased estimate

## Metadata

- parent domain: machine_learning
