# SISA Machine Unlearning using MNIST

## Project Overview

This project implements the core ideas of the paper:

> *Machine Unlearning* — Bourtoule et al. (2021)

The implementation focuses on the **SISA framework**:

* Sharded
* Isolated
* Sliced
* Aggregated

The system demonstrates:

* SISA-based training
* Incremental slice checkpointing
* Localized machine unlearning
* Digit-level forgetting
* Ensemble prediction behavior
* Confidence-based evaluation
* Visualization of forgetting effects

Dataset used:

* MNIST

Framework:

* PyTorch

---

# Project Structure

```text
sisa_project/
│
├── src/
│   ├── aggregation.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── shard.py
│   ├── train.py
│   ├── unlearn_digit.py
│   └── __init__.py
│
├── experiments/
│   ├── main_digit_unlearning.py
│   ├── main_ensemble_unlearning.py
│   ├── main_train.py
│   └── main_visualize_unlearning.py
│
├── models/
├── unlearn_sharded_models/
├── data/
├── shard_mapping.pt
├── requirements.txt
└── README.md
```

---

# SISA Workflow

## Training Phase

1. Dataset is divided into multiple shards
2. Each shard is divided into slices
3. Each shard model is trained incrementally slice-by-slice
4. Checkpoints are saved after every slice

---

## Unlearning Phase

1. Select target digit to forget
2. Remove that digit from one shard
3. Retrain only the affected shard
4. Save retrained model separately

---

## Evaluation Phase

The project evaluates:

* Confidence before unlearning
* Confidence after unlearning
* Ensemble prediction behavior
* Visualization of forgetting effects

---

# Source Files Explanation

---

# `src/config.py`

## Purpose

Contains all centralized configuration constants.

---

## Variables

### `S`

Number of shards.

Example:

```python
S = 5
```

---

### `R`

Number of slices per shard.

Example:

```python
R = 10
```

---

### `BATCH_SIZE`

Training batch size.

---

### `EPOCHS_PER_SLICE`

Training epochs for every slice.

---

### `LEARNING_RATE`

Optimizer learning rate.

---

### `BASE_MODEL_DIR`

Stores original trained SISA models.

---

### `UNLEARN_SHARDED_MODEL_DIR`

Stores retrained/unlearned shard models.

---

# `src/dataset.py`

## Purpose

Loads MNIST dataset.

---

## Functions

### `get_mnist()`

Loads:

* MNIST training dataset
* tensor transformation

Returns:

```python
torchvision.datasets.MNIST
```

---

# `src/model.py`

## Purpose

Defines CNN architecture used for MNIST classification.

---

## Class

### `Net`

CNN architecture consisting of:

* Conv2D layers
* ReLU activations
* Max pooling
* Fully connected layers

Output:

```python
10 digit classes
```

---

# `src/shard.py`

## Purpose

Implements SISA dataset partitioning.

---

## Functions

---

### `create_shards(dataset_size, S)`

Divides dataset into:

```text
S disjoint shards
```

Returns:

```python
List of shard indices
```

---

### `create_slices(shards, R)`

Divides each shard into:

```text
R incremental slices
```

Returns:

```python
Nested slice structure
```

---

# `src/train.py`

## Purpose

Implements incremental SISA shard training.

---

## Functions

---

### `train_shard(shard_id, slices, dataset, device)`

Responsibilities:

* trains one shard
* trains incrementally slice-by-slice
* saves checkpoints after each slice

Checkpoint naming:

```text
slice_0.pt
slice_1.pt
...
slice_9.pt
```

Final checkpoint:

```text
slice_(R-1).pt
```

represents:

```text
complete shard model
```

---

# `src/unlearn_digit.py`

## Purpose

Implements digit-level machine unlearning.

---

## Functions

---

### `unlearn_digit_from_shard(...)`

Responsibilities:

* removes all instances of target digit
* retrains only one shard
* preserves original baseline models
* saves retrained checkpoints separately

This demonstrates:

```text
localized machine unlearning
```

---

# `src/evaluate.py`

## Purpose

Centralized evaluation utilities.

---

## Functions

---

### `predict(model, image, device)`

Predicts:

* class label
* prediction confidence

Returns:

```python
(prediction, confidence)
```

---

### `predict_full_distribution(model, image, device)`

Returns:

* predicted class
* confidence
* full probability distribution

Used for:

```text
visualization
```

---

### `evaluate_digit_confidence(model, dataset, digit, device)`

Computes:

```text
average confidence for one digit
```

across dataset samples.

Used for:

```text
quantitative forgetting evaluation
```

---

# `src/aggregation.py`

## Purpose

Implements SISA ensemble prediction.

---

## Functions

---

### `ensemble_predict(image, device, use_unlearned=False)`

Responsibilities:

* loads all shard models
* aggregates probabilities
* averages predictions across shards

Supports:

* baseline models
* unlearned models

Returns:

```python
(prediction, confidence, probability_distribution)
```

---

# Experiment Scripts

---

# `experiments/main_train.py`

## Purpose

Main training pipeline.

---

## Responsibilities

* load dataset
* create shards
* create slices
* create index mapping
* save shard mapping
* train all shard models

---

# `experiments/main_digit_unlearning.py`

## Purpose

Numerical forgetting experiment.

---

## Workflow

1. Evaluate confidence before unlearning
2. Unlearn target digit from one shard
3. Evaluate confidence after unlearning
4. Compute confidence drop

---

## Demonstrates

```text
quantitative forgetting
```

---

# `experiments/main_visualize_unlearning.py`

## Purpose

Visual forgetting demonstration.

---

## Workflow

1. Select one digit image
2. Predict before unlearning
3. Unlearn target digit
4. Predict after unlearning
5. Visualize probability distributions

---

## Demonstrates

```text
behavioral forgetting
```

---

# `experiments/main_ensemble_unlearning.py`

## Purpose

Demonstrates ensemble behavior after localized forgetting.

---

## Workflow

1. Aggregate predictions from all shards
2. Unlearn digit from one shard only
3. Re-evaluate ensemble prediction
4. Compare confidence changes

---

## Demonstrates

```text
localized forgetting with ensemble resilience
```

---

# Important Concepts

---

# Shard

A shard is:

```text
an independent model
```

Each shard trains on different subset of data.

---

# Slice

A slice is:

```text
incremental training stage/checkpoint
```

Slices are NOT separate models.

---

# Final Slice Checkpoint

```text
slice_(R-1).pt
```

represents:

```text
complete trained shard model
```

---

# Machine Unlearning

The project removes:

```text
digit-specific knowledge
```

from:

```text
one selected shard
```

without retraining entire system.

---

# Expected Experimental Results

---

# Before Unlearning

Example:

```text
Prediction: 5
Confidence: 0.99
```

---

# After Unlearning

Example:

```text
Prediction: 5
Confidence: 0.71
```

or

```text
Prediction: 3
Confidence: 0.41
```

---

# Interpretation

Confidence drop indicates:

```text
successful forgetting
```

within affected shard.

---

# How to Run

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train SISA models

```bash
python experiments/main_train.py
```

---

## 3. Run numerical forgetting experiment

```bash
python experiments/main_digit_unlearning.py
```

---

## 4. Run visualization experiment

```bash
python experiments/main_visualize_unlearning.py
```

---

## 5. Run ensemble experiment

```bash
python experiments/main_ensemble_unlearning.py
```

---

# Key Features

* SISA training
* Incremental checkpointing
* Localized retraining
* Digit-level forgetting
* Ensemble aggregation
* Confidence-based evaluation
* Visual probability comparison
* Baseline preservation

---

# References

Bourtoule, Lucas, et al.

> *Machine Unlearning.*
> IEEE Symposium on Security and Privacy, 2021.
