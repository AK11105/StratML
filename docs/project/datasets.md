# Curated Datasets

Datasets selected to demonstrate specific system capabilities. All are CSV-loadable and compatible with the existing `DataLoader → Profiler → Preprocessor → Pipeline` flow without code changes.

---

## 1. Binary Classification — Pima Indians Diabetes

**Source:** [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Size:** 768 samples, 8 features, binary target  
**Key signals triggered:** `imbalance_ratio > 2.0`, `overfitting`, `add_preprocessing: oversample`

The class imbalance (~35% positive) fires the `imbalance_ratio` branch in the action generator, triggering SMOTE oversampling. The system visibly cycles underfitting → regularization → model switching across LogisticRegression, GradientBoosting, and SVC. The counterfactual log captures each branch decision.

---

## 2. Multiclass Classification — UCI Wine Quality (Red)

**Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)  
**Size:** 1,599 samples, 11 features, 6 classes  
**Key signals triggered:** `underfitting`, full model registry sweep

Multi-class structure stresses `log_loss` computation. The decision engine detects underfitting early (many classes, few samples per class) and iterates through LDA, GaussianNB, RandomForest, and GradientBoosting. Good for showing the breadth of the model registry.

---

## 3. Regression — California Housing

**Source:** `sklearn.datasets.fetch_california_housing`  
**Size:** 20,640 samples, 8 features, continuous target  
**Key signals triggered:** `too_slow` (SVR), `r2`/`rmse` metrics path, budget-aware termination

```python
from sklearn.datasets import fetch_california_housing
fetch_california_housing(as_frame=True).frame.to_csv("data/california_housing.csv", index=False)
```

Runtime differences between Ridge/Lasso vs. tree ensembles vs. SVR are measurable. The `too_slow` signal fires on SVR, demonstrating the efficiency agent terminating expensive branches.

---

## 4. High-Dimensional Classification — MNIST Tabular

**Source:** [OpenML #554](https://www.openml.org/d/554)  
**Size:** 70,000 samples, 784 features, 10 classes  
**Key signals triggered:** `too_slow` (KNN, SVC), ML vs. DL pipeline comparison

Forces the efficiency agent to terminate KNN and SVC branches. Tree ensembles converge. Also the best dataset for comparing the ML pipeline against the DL pipeline (MLP/CNN1D) on the same data — TensorBoard curves show the difference.

---

## 5. Overfitting Showcase — Titanic

**Source:** [Kaggle](https://www.kaggle.com/c/titanic/data)  
**Size:** 891 samples, mixed types (numerical + categorical), binary target  
**Key signals triggered:** `overfitting`, `modify_regularization`, `decrease_model_capacity`, categorical encoding

Small dataset with high-cardinality categorical features causes overfitting on tree models. The `overfitting` signal fires and the system applies regularization and capacity reduction. Exercises the full preprocessing path (onehot encoding, imputation). The counterfactual log shows "what if we hadn't switched from DecisionTree."

---

## 6. Plateau / Stagnation Detection — Credit Card Fraud

**Source:** [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Size:** 284,807 samples, 30 features, binary target (0.17% fraud)  
**Key signals triggered:** `stagnating`, misleading accuracy plateau

Extreme imbalance (~580:1) causes accuracy to plateau near 0.998 immediately — a deceptive result. The decision engine detects stagnation and the system must reason beyond naive metric improvement. **Set `primary_metric: f1_score` in config.** Showcases the system's ability to handle deceptive performance signals.

---

## 7. DL Pipeline — Appliances Energy Prediction

**Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)  
**Size:** 19,735 samples, 27 features, regression target (energy in Wh)  
**Key signals triggered:** early stopping (RNN), TensorBoard training curves, MLP vs. CNN1D vs. RNN comparison

Continuous target with temporal structure. All three DL architectures produce meaningfully different results. Early stopping fires on the RNN path. Best dataset for demonstrating TensorBoard curve divergence across architectures.

---

## Summary

| # | Capability | Dataset | Key Signal |
|---|---|---|---|
| 1 | Binary classification | Pima Diabetes | `imbalance_ratio`, `overfitting` |
| 2 | Multiclass classification | Wine Quality (Red) | `underfitting`, model switching |
| 3 | Regression | California Housing | `too_slow` (SVR), `r2`/`rmse` |
| 4 | High-dimensional | MNIST Tabular | `too_slow`, ML vs. DL |
| 5 | Overfitting showcase | Titanic | `overfitting`, categorical encoding |
| 6 | Plateau/stagnation | Credit Card Fraud | `stagnating`, misleading accuracy |
| 7 | DL pipeline | Appliances Energy | Early stopping, TensorBoard curves |
