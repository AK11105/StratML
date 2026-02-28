# 🚀 Multi-Agent Controlled AutoML Framework

> A decision-aware AutoML system that models machine learning experimentation as an explicit, controllable process.

---

## 📌 Overview

The **Multi-Agent Controlled AutoML Framework** is an end-to-end experimentation system that integrates classical machine learning and deep learning pipelines under a structured decision engine.

Instead of relying solely on blind hyperparameter search, this framework:

* Treats experimentation as a **state-driven control problem**
* Separates **execution** from **decision logic**
* Supports **budget-aware experimentation**
* Provides full **traceability and observability**
* Enables reproducible and interpretable model selection

The system is modular, extensible, and designed for research and advanced experimentation workflows.

---

## 🧠 Core Concept

The framework separates the system into two distinct layers:

### 🔹 Experiment Engine

Runs ML/DL pipelines and produces structured results.

### 🔹 Decision Engine

Analyzes results and determines the next experiment to execute.

This separation ensures:

* Clean system boundaries
* Interpretable decision-making
* Scalable architecture
* Controlled experimentation

---

## 🏗 Architecture

```
User (CLI / Config)
        ↓
Orchestrator
        ↓
Decision Engine (LangGraph + Rules)
        ↓
Execution Engine (ML / DL Pipelines)
        ↓
Metrics + Logs
        ↺ (Feedback Loop)
```

### Key Components

* **LangGraph Controller** – Manages decision flow
* **Rule Engine** – Deterministic policy layer
* **ML Pipelines** – scikit-learn models
* **DL Pipelines** – PyTorch templates
* **MLflow** – Experiment tracking
* **TensorBoard** – Training visualization
* **LangSmith** – Decision trace logging

---

## 📁 Project Structure

```
multi_agent_automl/
│
├── cli/              # CLI interface
├── config/           # Configuration files
├── core/             # Shared schemas & contracts
├── execution/        # ML/DL experiment engine
├── decision/         # State, rules, controller
├── orchestration/    # Connects decision + execution
├── tracking/         # MLflow, TensorBoard, LangSmith
├── analysis/         # Evaluation & comparisons
└── outputs/          # Saved models & artifacts
```

Each module has a single responsibility to prevent architectural sprawl.

---

## ⚙️ Features

### 🔬 Execution Engine

* Classical ML pipelines
* Template-based DL models
* Early stopping
* Cost estimation
* Structured experiment outputs

### 🧠 Decision Engine

* State abstraction from metrics
* Underfitting / overfitting detection
* Budget-aware control
* Rule-based experiment selection
* Full decision traceability

### 📊 Observability

* Experiment comparison (MLflow)
* Training curves (TensorBoard)
* Decision traces (LangSmith)

---

## 🔄 Experiment Lifecycle

1. User submits dataset & configuration
2. Execution engine runs baseline
3. Metrics are logged
4. Decision engine evaluates state
5. Next action is selected
6. Loop continues until termination

---

## 🎯 Design Principles

* Interpretability over blind automation
* Explicit state and action modeling
* Budget-aware experimentation
* Modular extensibility
* Reproducibility by design

---

## 🔮 Roadmap

Planned extensions include:

* Learned decision advisors
* Hybrid rule + predictive arbitration
* Expanded model templates
* User-registered custom pipelines
* Policy comparison framework

---

## 🧩 Tech Stack

* Python
* PyTorch
* scikit-learn
* LangChain
* LangGraph
* MLflow
* TensorBoard
* LangSmith

---

## 🚀 Getting Started

```bash
python cli/main.py run --dataset path/to/data.csv
```

(Full installation and configuration guide coming soon.)

---

## 💡 Project Vision

Transform AutoML from a black-box optimization process into a structured, decision-aware experimentation system.
