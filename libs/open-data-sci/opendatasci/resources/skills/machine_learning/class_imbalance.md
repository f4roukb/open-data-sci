# Machine Learning — Class Imbalance

- Class imbalance affects both what the model learns and how performance is measured; addressing only one side gives a misleading picture
- Threshold adjustment and class weighting are low-cost interventions that often recover substantial minority-class performance without resampling
- Resampling techniques change the training distribution and should only ever be applied within the training fold — contaminating validation or test data with synthetic samples invalidates evaluation
- Extreme imbalance changes the problem framing: precision at a given recall threshold or anomaly detection approaches may be more appropriate than standard classification evaluation

## Metadata

- parent domain: machine_learning
