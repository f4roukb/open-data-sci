# Competitive Data Science — Phase 5: Model Development

**Planning**
- The model plan covers a portfolio of model families to evaluate (linear, tree-based, neural) with a budget per family, rather than a single architecture to perfect
- Building several distinct base models — even when individually weaker than the strongest — pays compounding returns at the ensembling phase
- Iteration speed dominates progress in this phase; running on a representative stratified subsample first and full data once the architecture is settled cuts wall-clock cost substantially without losing directional signal

**Knowledge & Information**
- Model selection follows problem structure and data characteristics rather than a habitual preference for any one family; the choice is a design decision deserving the same rigour as any other modelling choice
- For tabular data, gradient-boosted decision trees are the most consistent top performer across small-to-medium datasets; differences across implementations in handling of categoricals, missing values, and split criteria occasionally swing one toward better performance on a given dataset, and at very large scale or in domains with rich high-cardinality interactions (CTR prediction, recommendation, certain industrial datasets) neural approaches can match or surpass them
- Linear and logistic regression with engineered features can be primary contenders in low-data regimes, when interactions are well-captured by hand-crafted features, or under strict interpretability constraints; they also serve as complementary ensemble components, as a sanity check on whether non-linear models add value, and as the standard meta-learner in stacking
- Neural architectures on tabular data (MLP, TabNet, FT-Transformer, NODE) are competitive at scale and increasingly close the gap to boosted trees on small-to-medium datasets when learned categorical embeddings or cross-feature interactions carry signal; their primary value in many competitions is diversity contribution to the ensemble, but in data regimes where they match or beat boosted trees they belong as a primary candidate rather than only as an ensemble component
- For text, fine-tuned transformer checkpoints (BERT-family, DeBERTa, RoBERTa, ELECTRA, distilled variants) lift performance substantially over feature-based baselines at meaningful compute cost on most natural-language tasks; on very short, highly structured, or heavily label-noisy text, TF-IDF or hashed n-grams with a linear classifier can match or outperform transformers at a fraction of the cost
- For image, pre-trained backbones (EfficientNet, ConvNeXt, ViT, Swin) via transfer learning are the standard entry point; augmentation design, head architecture, and training schedule often move the score more than swapping backbones of similar capacity, while in tasks where the backbone's inductive bias aligns particularly well with the data (fine-grained classification, medical imaging, satellite imagery, dense prediction) the backbone choice itself can be the dominant factor
- For time series with many parallel series, gradient boosting on lag features competes with and often beats dedicated forecasting architectures (LSTM, Temporal Fusion Transformer); the dedicated architectures win when complex exogenous structure or long-range dependencies dominate
- Test-time augmentation (TTA) — averaging predictions over augmented copies of each test instance — produces small but consistent gains in image tasks and sometimes in text and tabular
- Models train on the train split and select hyperparameters on the validation fold; the test set is never seen by any model selection process

**Tricks**
- Training the same model with several random seeds and averaging predictions is the cheapest, most reliable way to reduce variance and improve score
- For boosted trees, early stopping against the validation fold within each CV split removes the n_estimators hyperparameter from the search and acts as the primary regulariser
- Pseudo-labelling — training on high-confidence test predictions, then re-training — can add meaningful gains when the test set is large relative to train, but risks amplifying mistakes if confidence calibration is poor
- Saving OOF and test predictions from every meaningful model run feeds Phase 7 directly; ensembling later without these arrays forces re-running expensive trainings
- Knowledge distillation (training a smaller or differently-architected student on the soft predictions of a strong teacher) produces useful diversity for ensembling when the student family is genuinely different from the teacher's
- For any neural training, starting with a low number of epochs (one or two), monitoring train and validation curves, and continuing only when results warrant it avoids wasted compute on unpromising architectures and surfaces data or pipeline issues early

## Part of

- `competitive_data_science` — the phased competition playbook this phase belongs to; load it for the full campaign overview, phase sequencing, and the cross-phase loop-back wiring.
