# Execution Critique

## Artifacts

- Fine for ML
- Explicit DL handling needed for .pth/.pt files. And the optimizer and model state_objects


## Config

- Need to decide together, list of possible actions so that the translation from ActionDecision is better
- Needs to be a lot more thorough on hyperparameters, too shallow currently
- Design decision whether same translator to handle ML and DL decisions

## Data

- Loader good for ML, need similar handling for DL with explicit DataLoader and Dataset objects
- Classification Unique threshold needs better handling (can lead to edge-cases)
- Same critique, profiling good for ML (imbalance_ratio, feature_variance_mean, class_entropy) missing
- DL data profiling needed

## Validation

- Very thin validation right now
- Need thorough validation, same column names not allowed, N/A columns don't get processed correctly, etc
- Dataset inference is hardcoded tabular, need multi-modal handling.

## Pipelines

- Verify if all models handled
- Elaborate control on hyperparameters for each model needed
- Tuning needs GridSearch vs RandomizedSearch approaches
- Base DL decent, but too shallow, lots of handling left

## Preprocessing

- Need to avoid data-leakage
- Careful handling of val/test data

# Must reference all schemas from core/schemas.py ; that must remain single source of truth

