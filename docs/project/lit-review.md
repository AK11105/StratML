# Literature Review

Research that directly informs the design and improvement of this system. Papers are
grouped by the system concern they address, not by publication date.

---

## 1. Multi-Agent AutoML Frameworks

### AutoML-Agent — Multi-Agent LLM Framework for Full-Pipeline AutoML
**Trirat, Jeong, Hwang — ICML 2025**

Proposes a multi-agent LLM framework covering the complete ML pipeline from data
retrieval to deployment. Uses specialized agents for prompt parsing, data handling, model
development, and code generation, coordinated by an Agent Manager. A
retrieval-augmented planning strategy generates multiple candidate workflows; task
decomposition enables parallel execution; multi-stage verification detects implementation
errors. Evaluated on seven tasks across fourteen datasets, showing higher success rates
than prior LLM-based AutoML approaches.

**Relevance to this system:**
- Validates the coordinator + specialist agent architecture we use
- Retrieval-augmented planning directly motivates the cross-run MetaMemory improvement
  (improvements-roadmap item 8)
- Multi-stage verification is the conceptual basis for the evaluator agent

*IEEE: P. Trirat, W. Jeong, and S. J. Hwang, "AutoML-Agent: A Multi-Agent LLM Framework
for Full-Pipeline AutoML," Proc. ICML, vol. 267, 2025.*

---

### I-MCTS — Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search
**Liang, Wei, Xu, Chen, Qian, Wu — arXiv 2025**

Improves ML pipeline search using Introspective MCTS. Unlike prior methods with static
search spaces, I-MCTS dynamically expands nodes by analyzing parent and sibling
solutions. Introduces a hybrid reward mechanism combining LLM-estimated node quality
with actual validation performance to guide exploration. Achieves 4% absolute improvement
over strong AutoML baselines on 20 datasets while maintaining computational efficiency.

**Relevance to this system:**
- The hybrid reward mechanism (LLM estimate + actual validation) is the principled
  version of what our value model + calibration pipeline approximates
- Dynamic node expansion is the theoretical basis for Thompson sampling over uncertainty
  estimates (improvements-roadmap item 5)
- The 4% improvement figure is a useful benchmark for what better exploration policy
  can realistically achieve

*IEEE: Z. Liang et al., "I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo
Tree Search," arXiv preprint, 2025.*

---

## 2. Decision Quality & Evaluation

### Evaluation Agent for AutoML Pipelines
**Du, Ahlawat, Liu, Wu — arXiv 2026**

Proposes an Evaluation Agent (EA) that performs post-hoc audits of intermediate decisions
in AutoML pipelines — not just final model accuracy. Audits four dimensions: decision
validity, reasoning consistency, model quality risks beyond accuracy, and counterfactual
decision impact. Detected faulty decisions with F1 of 0.919 and identified performance
impacts ranging from −4.9% to +8.3%.

**Relevance to this system:**
- Direct blueprint for the evaluator agent (improvements-roadmap item 3)
- The four audit dimensions map exactly onto our planned `EvaluationRecord` schema
- The −4.9% to +8.3% impact range quantifies why auditing decisions matters — faulty
  decisions are not just suboptimal, they can actively hurt performance

*IEEE: G. Du, A. Ahlawat, X. Liu, and J. Wu, "A Framework for Assessing AI Agent
Decisions and Outcomes in AutoML Pipelines," arXiv preprint, 2026.*

---

## 3. Hyperparameter Optimization & Budget Awareness

### SMAC3 — Versatile Bayesian Optimization for HPO
**Lindauer et al. — JMLR 2022**

A robust and flexible Bayesian Optimization framework for hyperparameter configuration.
Combines random forests as surrogate models with acquisition functions for efficient
search. Supports multi-fidelity evaluation, warm-starting from prior runs, and
algorithm configuration beyond just hyperparameters.

**Relevance to this system:**
- Our value model (RandomForestRegressor predicting action gain) is a simplified version
  of SMAC3's surrogate model approach — the same principle applied to action selection
  rather than hyperparameter search
- SMAC3's warm-starting from prior runs is the established technique behind our
  cross-run MetaMemory improvement (item 8)
- The surrogate model + acquisition function pattern is the formal grounding for
  replacing our greedy action selector with Thompson sampling (item 5)

*IEEE: M. Lindauer et al., "SMAC3: A Versatile Bayesian Optimization Package for
Hyperparameter Optimization," J. Mach. Learn. Res., vol. 23, no. 54, pp. 1–9, 2022.*

---

### Auto-Sklearn 2.0 — Hands-Free AutoML via Meta-Learning
**Feurer et al. — JMLR 2022**

Extends Auto-sklearn with meta-learning for warm-starting, portfolio-based initial
configurations, and a bandit strategy for budget allocation across pipelines. Introduces
a meta-feature-free meta-learning technique that selects initial configurations based on
dataset similarity without requiring expensive meta-feature computation. Achieves strong
performance under rigid time limits on large datasets.

**Relevance to this system:**
- The bandit strategy for budget allocation across pipelines is the formal version of
  what our efficiency agent approximates with heuristic cost proxies
- Meta-feature-free warm-starting is a simpler alternative to our planned MetaMemory
  module — worth considering as a fallback if cosine similarity over meta-features proves
  noisy
- Portfolio-based initial configurations directly informs how bootstrap candidates should
  be selected (currently just two random models)

*IEEE: M. Feurer et al., "Auto-Sklearn 2.0: Hands-Free AutoML via Meta-Learning,"
J. Mach. Learn. Res., vol. 23, no. 261, pp. 1–61, 2022.*

---

## 4. Agentic Experimentation & Workflow Automation

### ADE — Agent Driven Experiments for Deep Learning
**Black — 2026**

Presents a lightweight framework for automating repetitive DL experimentation tasks:
monitoring metrics, detecting anomalies, restarting failed jobs, adjusting hyperparameters,
and logging outcomes. Uses LangChain-based agents, YAML configs, and markdown preference
rules. Reduces manual operational overhead without replacing researcher judgment.

**Relevance to this system:**
- Validates the design philosophy of this system — automate the loop, not the researcher
- The markdown preference rules pattern is a simpler alternative to our LLM-based signal
  extraction for users who want deterministic behaviour
- The anomaly detection component maps onto our `diverging` and `unstable_training`
  signals

*IEEE: S. Black, "Agentic AI for Modern Deep Learning Experimentation," Feb. 2026.*

---

## 5. AutoML Surveys & Scope Definition

### AutoML Systematic Review with NAS
**Salehin et al. — Journal of Information and Intelligence, 2024**

Systematic review of 175 papers covering AutoML stages: data preparation, feature
engineering, model generation, HPO, and evaluation. Compares NAS methods on CIFAR-10,
CIFAR-100, ImageNet. Identifies open challenges: search space complexity, computational
cost, scalability.

**Relevance to this system:**
- Confirms that our system's scope (data prep → model selection → HPO → evaluation) is
  the correct full pipeline definition
- The identified open challenges (search space complexity, compute cost) are exactly what
  our budget-aware decision engine addresses
- NAS is explicitly out of scope for this system — this paper justifies that boundary

*IEEE: I. Salehin et al., "AutoML: A systematic review on automated machine learning with
neural architecture search," J. Inf. Intell., vol. 2, pp. 52–81, 2024.*

---

### Automated Deep Learning — NAS Is Not the End
**Dong, Kedziora, Musial, Gabrys — ACM Computing Surveys, 2022**

Argues that NAS alone does not capture the full scope of DL automation. Examines
automation across the complete DL lifecycle: task formulation, data engineering,
architecture design, HPO, deployment, continuous maintenance. Proposes ten evaluation
criteria including efficiency, scalability, reproducibility, interpretability, and
eco-friendliness.

**Relevance to this system:**
- The ten evaluation criteria are a useful checklist for assessing this system's maturity
- The argument for end-to-end autonomous systems over isolated NAS advances supports our
  full-pipeline approach
- Interpretability and reproducibility as first-class criteria validate our decision
  traceability design

*IEEE: X. Dong, D. J. Kedziora, K. Musial, and B. Gabrys, "Automated Deep Learning:
Neural Architecture Search Is Not the End," ACM Comput. Surv., vol. 55, no. 6, 2022.*

---

### AutoML in Software Engineering
**Calefato, Quaranta, Lanubile, Kalinowski — ICSE 2023**

Benchmarks 12 AutoML tools on SE text-classification datasets and surveys 45
practitioners. AutoKeras and DataRobot matched or outperformed researcher-developed
models. Current AutoML platforms mainly automate model training and evaluation — weak
support for data preparation and deployment.

**Relevance to this system:**
- The weak data preparation support finding is a gap this system addresses through
  adaptive preprocessing decisions
- Practitioner survey results confirm that interpretability and control are the primary
  reasons practitioners distrust AutoML — directly motivating our decision traceability
  and rule-based fallback design

*IEEE: F. Calefato, L. Quaranta, F. Lanubile, and M. Kalinowski, "Assessing the Use of
AutoML for Data-Driven Software Engineering," Proc. ICSE, 2023.*

---

## Summary Table

| Paper | Primary gap addressed |
|---|---|
| AutoML-Agent [1] | Agent council architecture, retrieval-augmented planning |
| I-MCTS [3] | Exploration policy, hybrid reward for value model |
| Evaluation Agent [2] | Evaluator agent design (item 3 in roadmap) |
| SMAC3 [new] | Surrogate model grounding, warm-starting |
| Auto-Sklearn 2.0 [new] | Budget allocation, meta-learning warm-start |
| ADE [4] | Agentic experimentation philosophy |
| AutoML Survey [5] | Scope definition, NAS boundary |
| AutoDL Survey [7] | Evaluation criteria, interpretability |
| AutoML in SE [6] | Practitioner trust, data prep gap |

---

## IEEE References

[1] P. Trirat, W. Jeong, and S. J. Hwang, "AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML," Proc. ICML, vol. 267, 2025.

[2] G. Du, A. Ahlawat, X. Liu, and J. Wu, "A Framework for Assessing AI Agent Decisions and Outcomes in AutoML Pipelines," arXiv preprint, 2026.

[3] Z. Liang, F. Wei, W. Xu, L. Chen, Y. Qian, and X. Wu, "I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search," arXiv preprint, 2025.

[4] S. Black, "Agentic AI for Modern Deep Learning Experimentation," Feb. 2026.

[5] I. Salehin et al., "AutoML: A systematic review on automated machine learning with neural architecture search," J. Inf. Intell., vol. 2, pp. 52–81, 2024.

[6] F. Calefato, L. Quaranta, F. Lanubile, and M. Kalinowski, "Assessing the Use of AutoML for Data-Driven Software Engineering," Proc. ICSE, 2023.

[7] X. Dong, D. J. Kedziora, K. Musial, and B. Gabrys, "Automated Deep Learning: Neural Architecture Search Is Not the End," ACM Comput. Surv., vol. 55, no. 6, 2022.

[8] M. Lindauer et al., "SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization," J. Mach. Learn. Res., vol. 23, no. 54, 2022.

[9] M. Feurer et al., "Auto-Sklearn 2.0: Hands-Free AutoML via Meta-Learning," J. Mach. Learn. Res., vol. 23, no. 261, 2022.
