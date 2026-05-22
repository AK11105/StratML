"""
dl_architectures.py
-------------------
PyTorch nn.Module definitions for all supported DL architectures.

Adding a new architecture:
  1. Define a new nn.Module class here.
  2. Register it in build_model().
  3. Add its name to _DL_MODELS in experiment_config_builder.py.
  No other files need to change.

Modalities:
  Tabular : MLP, CNN1D, RNN, ResidualMLP, TabTransformer
  Vision  : CNN2D, ResNet18, EfficientNetB0, MobileNetV3
  Text    : TextCNN, BiLSTM, DistilBERT, TinyBERT
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Tabular ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Fully-connected network for tabular data."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float, batch_norm: bool):
        super().__init__()
        dims = [input_dim] + [hidden_units] * layers
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """1-D convolutional network for tabular data with local feature structure."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 dropout: float, batch_norm: bool):
        super().__init__()
        h2 = max(1, hidden_units // 2)
        layers: list[nn.Module] = [nn.Conv1d(1, hidden_units, kernel_size=3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_units))
        layers += [nn.ReLU(), nn.Dropout(dropout),
                   nn.Conv1d(hidden_units, h2, kernel_size=3, padding=1)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(h2))
        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(h2 * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.head(self.conv(x).flatten(1))


class RNN(nn.Module):
    """LSTM-based network for ordered/sequential feature sets."""
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_units,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class ResidualMLP(nn.Module):
    """
    MLP with skip connections every 2 layers.
    Fixes degradation when increase_model_capacity keeps adding layers
    but val loss stops improving.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float, batch_norm: bool):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_units)
        self.blocks = nn.ModuleList()
        for _ in range(max(1, layers // 2)):
            block = nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                nn.BatchNorm1d(hidden_units) if batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, hidden_units),
                nn.BatchNorm1d(hidden_units) if batch_norm else nn.Identity(),
            )
            self.blocks.append(block)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.input_proj(x))
        for block in self.blocks:
            x = self.relu(block(x) + x)
        return self.head(x)


class TabTransformer(nn.Module):
    """
    Column-wise self-attention on categorical embeddings concatenated with
    numeric features. Current SOTA on tabular benchmarks with many categoricals.

    For simplicity, treats all input features as numeric (no separate embedding
    table). The transformer operates on the feature sequence directly.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 layers: int, dropout: float):
        super().__init__()
        nhead = max(1, min(8, hidden_units // 16))
        # Ensure hidden_units is divisible by nhead
        hidden_units = nhead * max(1, hidden_units // nhead)
        self.input_proj = nn.Linear(1, hidden_units)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=nhead, dropout=dropout,
            dim_feedforward=hidden_units * 2, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers))
        self.head = nn.Linear(hidden_units * input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) → (batch, features, 1) → project → transformer
        x = self.input_proj(x.unsqueeze(-1))   # (B, F, H)
        x = self.transformer(x)                 # (B, F, H)
        return self.head(x.flatten(1))


# ── Vision ────────────────────────────────────────────────────────────────────

class CNN2D(nn.Module):
    """
    Simple 2-D CNN for image classification.
    Expects input as flat vector of length C*H*W; reshapes internally.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_units: int,
                 dropout: float, image_shape: tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        h2 = max(1, hidden_units // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(C, hidden_units, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units, h2, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        conv_out = h2 * (H // 4) * (W // 4)
        self.head = nn.Linear(conv_out, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = self.image_shape
        x = x.view(-1, C, H, W)
        return self.head(self.conv(x).flatten(1))


class ResNet18Finetune(nn.Module):
    """ResNet-18 with ImageNet weights; only the final FC layer is trained initially."""
    def __init__(self, output_dim: int, frozen: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
        )

    def unfreeze_last(self, n_blocks: int) -> None:
        """Progressively unfreeze the last n layer groups."""
        layers = [self.backbone.layer4, self.backbone.layer3,
                  self.backbone.layer2, self.backbone.layer1]
        for layer in layers[:n_blocks]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB0Finetune(nn.Module):
    """EfficientNet-B0 with ImageNet weights; best accuracy/compute tradeoff."""
    def __init__(self, output_dim: int, frozen: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, output_dim),
        )

    def unfreeze_last(self, n_blocks: int) -> None:
        features = list(self.backbone.features.children())
        for block in features[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetV3Finetune(nn.Module):
    """MobileNetV3-Small — triggered when too_slow fires on larger vision models."""
    def __init__(self, output_dim: int, frozen: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, output_dim)

    def unfreeze_last(self, n_blocks: int) -> None:
        features = list(self.backbone.features.children())
        for block in features[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Text ──────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Kim (2014) TextCNN: embedding + parallel Conv1d with kernel sizes [3,4,5]
    + max-pool + linear head.
    """
    def __init__(self, vocab_size: int, output_dim: int, embed_dim: int = 128,
                 hidden_units: int = 128, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, hidden_units, kernel_size=k) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_units * 3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) int64
        emb = self.embedding(x).permute(0, 2, 1)  # (B, E, L)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.head(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Bidirectional LSTM for sequential text context."""
    def __init__(self, vocab_size: int, output_dim: int, embed_dim: int = 128,
                 hidden_units: int = 128, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_units, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_units * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.head(out[:, -1, :])


class DistilBERTFinetune(nn.Module):
    """DistilBERT with a linear classification head. Backbone frozen initially."""
    def __init__(self, output_dim: int, frozen: bool = True, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        if frozen:
            for p in self.bert.parameters():
                p.requires_grad = False
        hidden = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def unfreeze_last(self, n_layers: int) -> None:
        transformer_layers = self.bert.transformer.layer
        for layer in transformer_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


class TinyBERTFinetune(nn.Module):
    """TinyBERT — triggered when too_slow fires on DistilBERT."""
    def __init__(self, output_dim: int, frozen: bool = True, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        if frozen:
            for p in self.bert.parameters():
                p.requires_grad = False
        hidden = self.bert.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def unfreeze_last(self, n_layers: int) -> None:
        transformer_layers = self.bert.encoder.layer
        for layer in transformer_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])


# ── Factory ───────────────────────────────────────────────────────────────────

_VISION_PRETRAINED = {"RESNET18", "EFFICIENTNETB0", "MOBILENETV3"}
_TEXT_PRETRAINED   = {"DISTILBERT", "TINYBERT"}

def build_model(arch: str, input_dim: int, output_dim: int, hp: dict) -> nn.Module:
    """
    Instantiate the correct architecture from the arch string and hyperparameters.
    Unknown arch strings fall back to MLP.
    """
    key        = arch.upper().replace("-", "").replace("_", "")
    hidden     = int(hp.get("hidden_units", 64))
    layers     = int(hp.get("layers", 2))
    dropout    = float(hp.get("dropout", 0.0))
    batch_norm = bool(hp.get("batch_norm", False))
    frozen     = bool(hp.get("frozen", True))

    # ── Tabular ──
    if key == "CNN1D":
        return CNN1D(input_dim, output_dim, hidden, dropout, batch_norm)
    if key == "RNN":
        return RNN(input_dim, output_dim, hidden, layers, dropout)
    if key == "RESIDUALMLP":
        return ResidualMLP(input_dim, output_dim, hidden, layers, dropout, batch_norm)
    if key == "TABTRANSFORMER":
        return TabTransformer(input_dim, output_dim, hidden, layers, dropout)

    # ── Vision ──
    if key == "CNN2D":
        image_shape = tuple(hp.get("image_shape", (1, 28, 28)))
        return CNN2D(input_dim, output_dim, hidden, dropout, image_shape)
    if key == "RESNET18":
        return ResNet18Finetune(output_dim, frozen=frozen, dropout=dropout)
    if key == "EFFICIENTNETB0":
        return EfficientNetB0Finetune(output_dim, frozen=frozen, dropout=dropout)
    if key == "MOBILENETV3":
        return MobileNetV3Finetune(output_dim, frozen=frozen, dropout=dropout)

    # ── Text ──
    vocab_size = int(hp.get("vocab_size", 30522))
    embed_dim  = int(hp.get("embed_dim", 128))
    if key == "TEXTCNN":
        return TextCNN(vocab_size, output_dim, embed_dim, hidden, dropout)
    if key == "BILSTM":
        return BiLSTM(vocab_size, output_dim, embed_dim, hidden, layers, dropout)
    if key == "DISTILBERT":
        return DistilBERTFinetune(output_dim, frozen=frozen, dropout=dropout)
    if key == "TINYBERT":
        return TinyBERTFinetune(output_dim, frozen=frozen, dropout=dropout)

    # ── Default ──
    return MLP(input_dim, output_dim, hidden, layers, dropout, batch_norm)
