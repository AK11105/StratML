# Decision Engine — Test Commands

Run from the project root: `multi-agent-auto-ml/`

---

## Setup (one-time)

```bash
# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate venv (bash/WSL)
source .venv/Scripts/activate

# Install pytest if not present
pip install pytest
```

---

## Run All Decision Engine Tests

```bash
pytest tests/unit/test_decision_engine.py -v
```

---

## Run by Component

```bash
# 1. Candidate Action Generator
pytest tests/unit/test_decision_engine.py::TestActionGenerator -v

# 2. Decision Dataset Builder
pytest tests/unit/test_decision_engine.py::TestDatasetBuilder -v

# 3. Value Model (stub)
pytest tests/unit/test_decision_engine.py::TestValueModel -v

# 4. Calibration Layer (stub)
pytest tests/unit/test_decision_engine.py::TestCalibration -v

# 5. Uncertainty Estimator (stub)
pytest tests/unit/test_decision_engine.py::TestUncertainty -v

# 6. Performance Agent
pytest tests/unit/test_decision_engine.py::TestPerformanceAgent -v

# 7. Efficiency Agent
pytest tests/unit/test_decision_engine.py::TestEfficiencyAgent -v

# 8. Stability Agent
pytest tests/unit/test_decision_engine.py::TestStabilityAgent -v

# 9. Coordinator Agent
pytest tests/unit/test_decision_engine.py::TestCoordinatorAgent -v

# 10. Action Selector (Policy)
pytest tests/unit/test_decision_engine.py::TestActionSelector -v

# 11. Decision Logger
pytest tests/unit/test_decision_engine.py::TestDecisionLogger -v

# 12. Counterfactual Validator (stub)
pytest tests/unit/test_decision_engine.py::TestCounterfactual -v
```

---

## Run Full Unit Suite (all existing + decision)

```bash
pytest tests/unit/ -v
```

---

## Run with Coverage

```bash
pip install pytest-cov

pytest tests/unit/test_decision_engine.py \
  --cov=stratml/decision \
  --cov-report=term-missing \
  -v
```

---

## Expected: All Pass

| # | Class | Tests |
|---|---|---|
| 1 | `TestActionGenerator` | 8 |
| 2 | `TestDatasetBuilder` | 3 |
| 3 | `TestValueModel` | 3 |
| 4 | `TestCalibration` | 2 |
| 5 | `TestUncertainty` | 3 |
| 6 | `TestPerformanceAgent` | 3 |
| 7 | `TestEfficiencyAgent` | 3 |
| 8 | `TestStabilityAgent` | 3 |
| 9 | `TestCoordinatorAgent` | 4 |
| 10 | `TestActionSelector` | 5 |
| 11 | `TestDecisionLogger` | 3 |
| 12 | `TestCounterfactual` | 3 |
| | **Total** | **46** |
