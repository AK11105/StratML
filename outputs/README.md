# outputs/

This directory holds all structured outputs.

## Layout

```
outputs/
  <dataset_name>/
      data_profile.json          # Phase 2 — DataProfile sent to Team B (once)
      experiments/
          <experiment_id>/
              config.json        # ExperimentConfig used for this run
              metrics.json       # Final ExperimentMetrics snapshot
              experiment_result.json  # Full ExperimentResult sent to Team B
```

## Lifecycle

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `data_profile.json` | Phase 2 (profiler) | Team B — populates StateObject.dataset_meta_features |
| `config.json` | Phase 4 (config builder) | Artifact reference, reproducibility |
| `metrics.json` | Phase 6 (metrics processor) | Team B — ExperimentResult.metrics |
| `experiment_result.json` | Phase 8 (result assembler) | Team B — state_builder.py |
