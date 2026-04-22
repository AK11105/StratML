# To-Do

1. ~~Initial Data Profile handling for task selection and initial model decision~~ ✅ `imbalance_ratio`, `feature_variance_mean`, `class_entropy` added to `DataProfile` and profiler
2. Decide exhaustive set of actions allowed in the system
3. Output structure and folder refined
4. Use single source of truth for schemas (`stratml/core/schemas.py`) — `execution/schemas.py` still duplicates `ExperimentMetrics`, `ResourceUsage`, `ArtifactRefs`, `ExperimentResult`, `ActionDecision`
5. Better PDFs — training curve charts (matplotlib embeds) not yet added
6. Agent integration on decision side (post StateObject)
7. model.py needs to be optional (config/cli), model.pkl always
8. Verify generated objects (all types) are consistent w.r.t `docs/schemas.md`
9. Wire `tune=True` through CLI (`--tune` flag) and config builder — `RandomizedSearchCV` is implemented but unreachable from user-facing commands
10. DL test set evaluation — orchestrator currently skips it; needs model rebuild from saved state + predict on `X_test`
11. TensorBoard wiring — `SummaryWriter` calls not yet added inside `dl_pipeline.py` epoch loop
12. DL artifact saving — explicit `.pth` state-dict + optimizer state not yet implemented
