# DL Pipeline Additions

## Current State

Three tabular-only architectures: `MLP`, `CNN1D`, `RNN`.  
No vision or text support. No pretrained model usage.  
Mutations cover capacity, dropout, and LR only.  
Action generator has no DL-aware branching.

---

## Proposed Additions

### Tabular

| Arch | Type | Pretrained | Trigger |
|---|---|---|---|
| `MLP` | existing | No | baseline |
| `CNN1D` | existing | No | local feature correlations |
| `RNN` | existing | No | ordered/sequential features |
| `ResidualMLP` | new | No | deep underfitting, vanishing gradient |
| `TabTransformer` | new | No | high-cardinality categoricals, large tabular |

**ResidualMLP** — MLP with skip connections every 2 layers. Fixes the degradation problem when `increase_model_capacity` keeps adding layers but val loss stops improving.

**TabTransformer** — column-wise self-attention on categorical embeddings concatenated with numeric features. Current SOTA on tabular benchmarks. Defensible when dataset has many categorical columns.

---

### Vision

| Arch | Type | Pretrained | Trigger |
|---|---|---|---|
| `CNN2D` | new | No | baseline, small datasets |
| `ResNet18` | new | ImageNet (`torchvision`) | medium datasets, transfer learning |
| `EfficientNetB0` | new | ImageNet (`torchvision`) | best accuracy/compute tradeoff |
| `MobileNetV3` | new | ImageNet (`torchvision`) | `too_slow` fires on larger models |

All pretrained vision models replace only the final classifier layer with `nn.Linear(feature_dim, output_dim)`. Backbone starts frozen. `increase_model_capacity` progressively unfreezes the last N blocks.

**Reference dataset:** CIFAR-10 — 60k 32×32 RGB images, 10 classes, available via `torchvision.datasets`.

**Decision chain:** `CNN2D` → `ResNet18` → `EfficientNetB0`. If `too_slow` fires: → `MobileNetV3`.

---

### Text

| Arch | Type | Pretrained | Trigger |
|---|---|---|---|
| `TextCNN` | new | No (random embeddings) | baseline, fast |
| `BiLSTM` | new | No (random embeddings) | sequential context, stagnating TextCNN |
| `DistilBERT` | new | HuggingFace (`distilbert-base-uncased`) | best accuracy, budget allows |
| `TinyBERT` | new | HuggingFace (`huawei-noah/TinyBERT_General_4L_312D`) | `too_slow` fires on DistilBERT |

**TextCNN** — Kim (2014): embedding layer + parallel Conv1d with kernel sizes [3,4,5] + max-pool + linear head.

**BiLSTM** — `nn.Embedding` + bidirectional LSTM + last hidden state → linear head.

**DistilBERT / TinyBERT** — HuggingFace `AutoModel` with a linear classification head. Backbone frozen initially; `increase_model_capacity` unfreezes transformer layers.

**Reference dataset:** IMDb sentiment — 50k reviews, binary classification. Download via `datasets` library (`load_dataset("imdb")`).

**Decision chain:** `TextCNN` → `BiLSTM` → `DistilBERT`. If `too_slow` fires: → `TinyBERT`.

---

## Pretrained Model Strategy

The decision engine treats pretrained models as a distinct escalation tier:

1. **Bootstrap** — always starts with a fast custom architecture (`CNN2D`, `TextCNN`, `MLP`)
2. **Underfitting persists** — `switch_model` escalates to a pretrained backbone (frozen)
3. **Still underfitting** — `increase_model_capacity` unfreezes the last N backbone layers
4. **`too_slow` fires** — `switch_model` drops to a smaller pretrained model (`MobileNetV3`, `TinyBERT`)
5. **Budget exhausted** — `terminate`

This mirrors standard transfer learning practice and is budget-aware: the engine will not unfreeze a 66M-parameter backbone if only 1 iteration remains.

---

## Code Changes Required

### `dl_architectures.py`
Add: `ResidualMLP`, `TabTransformer`, `CNN2D`, `ResNet18Finetune`, `EfficientNetB0Finetune`, `MobileNetV3Finetune`, `TextCNN`, `BiLSTM`, `DistilBERTFinetune`, `TinyBERTFinetune`.

Each pretrained wrapper follows the same pattern:
```python
class ResNet18Finetune(nn.Module):
    def __init__(self, output_dim, frozen=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.fc = nn.Linear(512, output_dim)

    def unfreeze_last(self, n_blocks):
        # unfreeze last n layer groups for progressive fine-tuning
        ...
```

### `dl_pipeline.py`
Three new DataLoader paths:
- **Vision** — reshape `(batch, C*H*W)` → `(batch, C, H, W)`, apply ImageNet normalization for pretrained models
- **Text** — input is int64 token IDs; BERT models need `attention_mask` passed separately
- **Tabular** — unchanged (existing path)

### `dl_mutations.py`
Two new functions:
```python
def unfreeze_backbone(hp, n_layers): ...   # progressive unfreezing
def switch_architecture(hp, new_arch): ... # swap arch, reset incompatible params
```

### `experiment_config_builder.py`
Add all new arch names to `_DL_MODELS`.

### `action_generator.py`
Add DL-aware branch in `_rule_candidates`. When `model_type == "dl"`, propose from the correct modality pool:

```python
_DL_VISION_MODELS = ["CNN2D", "ResNet18", "EfficientNetB0", "MobileNetV3"]
_DL_TEXT_MODELS   = ["TextCNN", "BiLSTM", "DistilBERT", "TinyBERT"]
_DL_TABULAR_MODELS = ["MLP", "CNN1D", "RNN", "ResidualMLP", "TabTransformer"]
```

Signal → action mapping for DL:

| Signal | Modality | Action |
|---|---|---|
| `underfitting` | any | `switch_model` to next tier, or `increase_model_capacity` |
| `overfitting` | any | `modify_regularization` (dropout ↑), `decrease_model_capacity` |
| `too_slow` | vision | `switch_model` → `MobileNetV3` |
| `too_slow` | text | `switch_model` → `TinyBERT` |
| `stagnating` | any | `switch_model` to pretrained tier |
| `converged` + `well_fitted` | any | `terminate` |

### New demo scripts

**`demo_cifar10.py`**
```
CNN2D (underfitting, r2≈0.72)
→ ResNet18 pretrained frozen (jump to 0.89, overfitting)
→ EfficientNetB0 pretrained + dropout (0.91, converged)
→ terminate
```

**`demo_imdb.py`**
```
TextCNN (stagnating at 0.84)
→ BiLSTM (0.87, improvement)
→ DistilBERT frozen (0.91, too_slow fires)
→ TinyBERT (0.89, converged, fast)
→ terminate
```

**`demo_mnist_dl.py`**
```
MLP flat 784 (underfitting vs spatial models, 0.965)
→ CNN2D (0.991, overfitting)
→ ResNet18 pretrained (0.993, too_slow)
→ MobileNetV3 (0.992, converged)
→ terminate
```

### `main.py` `_DEMO_MAP`
```python
"cifar10": "demo.demo_cifar10",
"imdb":    "demo.demo_imdb",
```

### `data/scripts/`
```
download_cifar10.py   — torchvision.datasets.CIFAR10, save as flat CSV
download_imdb.py      — datasets.load_dataset("imdb"), save train/test CSV
```

---

## Dependencies

| Package | Purpose | Already installed |
|---|---|---|
| `torchvision` | ResNet18, EfficientNetB0, MobileNetV3, CIFAR-10 | Yes |
| `transformers` | DistilBERT, TinyBERT | No — `pip install transformers` |
| `datasets` | IMDb download | No — `pip install datasets` |

---

## Why This Is Defensible

- **Pretrained models are not magic** — the system uses them as a deliberate escalation decision triggered by specific signals (`underfitting` after custom arch, `too_slow` for model selection). The decision is traceable in the LangSmith trace and decision log.
- **Frozen → unfreeze progression** mirrors the standard transfer learning literature (Howard & Ruder 2018, ULMFiT). The budget-awareness constraint (don't unfreeze if 1 iteration remains) is a concrete design decision.
- **Three modalities** cover the standard ML problem taxonomy. Each has a fast baseline, a mid-tier custom model, and a pretrained SOTA option — the decision engine can demonstrate the full escalation chain in a single run.
- **Metrics are consistent** — vision and text both use accuracy + F1 as primary metrics, same schema as tabular classification. The decision engine's signal extraction is modality-agnostic.
