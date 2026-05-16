# External Datasets (Manual Download Required)

These datasets require a Kaggle account. Download and place the CSV files here (`data/external/`).

---

## Titanic

**URL:** https://www.kaggle.com/c/titanic/data  
**File to download:** `train.csv`  
**Rename to:** `titanic.csv`  
**Target column:** `Survived`

```bash
kaggle competitions download -c titanic -f train.csv
mv train.csv data/external/titanic.csv
```

---

## Credit Card Fraud Detection

**URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**File to download:** `creditcard.csv`  
**Place as:** `creditcard.csv`  
**Target column:** `Class`

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip creditcard.csv -d data/external/
```

> Note: Set `primary_metric: f1_score` in config when running this dataset.
> Accuracy is misleading due to extreme class imbalance (~0.17% fraud).
