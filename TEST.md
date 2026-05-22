# TEST.md — Full Demo Script

Run these commands in order during the presentation.

---

## 1. Install

```powershell
powershell -ExecutionPolicy Bypass -File stratml\cli\install.ps1
```

Restart terminal, then verify:

```bash
stratml doctor
```

---

## 2. Profile a dataset

```bash
stratml profile-data data/external/titanic.csv Survived
```

---

## 3. Validate config

```bash
stratml validate-config config.yaml
```

---

## 4. Dry run

```bash
stratml run config.yaml --path data/external/titanic.csv --dry-run
```

---

## 5. ML demos

```bash
stratml run config.yaml --path data/external/titanic.csv
stratml run config.yaml --path data/raw/wine_quality_red.csv
stratml run config.yaml --path data/raw/california_housing.csv
```

---

## 6. DL demo

```bash
stratml run config.yaml --path data/raw/cifar10.csv
```
