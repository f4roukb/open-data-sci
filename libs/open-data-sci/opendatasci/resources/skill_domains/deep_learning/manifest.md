# Deep Learning Skill Domain

A JAX/Flax NNX/Optax-based playbook for deep learning: when it's the right tool, the library stack and its primitives, architecture selection, training loop design, regularisation, tuning, and evaluation. Load the skill for the piece of the workflow you're actively working on.

## Library Stack

### Metadata
- skill_domain_name: deep_learning
- skill_name: library_stack

### Description
Pointers to the JAX, Flax NNX, and Optax documentation that underpin this domain's approach; load for quick reference to the canonical docs.

## When to Use Deep Learning

### Metadata
- skill_domain_name: deep_learning
- skill_name: when_to_use

### Description
When data structure, volume, and available transfer-learning checkpoints favour a neural approach versus scikit-learn or gradient boosting; load before committing to deep learning as the modelling approach.

## scikit-learn MLP

### Metadata
- skill_domain_name: deep_learning
- skill_name: sklearn_mlp

### Description
When a shallow `MLPClassifier`/`MLPRegressor` inside the sklearn ecosystem is the right level of complexity, and its monitoring and early-stopping options; load for a quick neural baseline that doesn't need the JAX stack.

## JAX Fundamentals

### Metadata
- skill_domain_name: deep_learning
- skill_name: jax_fundamentals

### Description
Functional purity and PRNG key handling, the core transformations (`jit`, `grad`, `vmap`, `scan`), and JAX's array and dtype semantics; load before writing any JAX model code.

## Flax NNX: Defining Models

### Metadata
- skill_domain_name: deep_learning
- skill_name: flax_nnx

### Description
Module basics, built-in layers, state management, and train/eval mode switching in Flax NNX; load while defining or modifying a model architecture.

## Optax: Optimisation and Schedules

### Metadata
- skill_domain_name: deep_learning
- skill_name: optax

### Description
Optimiser selection, learning rate schedules, gradient transformations, loss functions, and optimiser state management via Optax; load while setting up or adjusting the optimisation pipeline.

## Training Loop Design

### Metadata
- skill_domain_name: deep_learning
- skill_name: training_loop

### Description
Standard training step and epoch structure, data loading patterns outside JAX, logging and monitoring, and reproducibility practices; load while building or debugging the training loop.

## Architecture Selection

### Metadata
- skill_domain_name: deep_learning
- skill_name: architecture_selection

### Description
Choosing architectures by modality — tabular, sequences and time series, images, and attention/transformers; load when deciding what kind of model to build.

## Regularisation and Overfitting

### Metadata
- skill_domain_name: deep_learning
- skill_name: regularisation_overfitting

### Description
Core regularisation techniques, data augmentation by modality, and diagnosing capacity versus overfitting problems from the train/validation loss gap; load when a model is over- or under-fitting.

## Hyperparameter Tuning

### Metadata
- skill_domain_name: deep_learning
- skill_name: hyperparameter_tuning

### Description
The highest-leverage hyperparameters and a coarse-to-fine, Optuna-driven search strategy; load once a model trains end-to-end and is ready to be tuned.

## Evaluation

### Metadata
- skill_domain_name: deep_learning
- skill_name: evaluation

### Description
Neural-specific evaluation considerations — eval-mode correctness, calibration, seed averaging, test-time augmentation, and ensembling with classical models; load when evaluating a trained model.
