from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torchvision import models


def build_resnet(backbone: str = "resnet50", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """Build the ResNet classifier used for otoscopy frame classification."""
    backbone = backbone.lower()
    weight_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    }
    ctor_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if backbone not in ctor_map:
        raise ValueError(f"Unsupported backbone: {backbone}")

    weights = weight_map[backbone] if pretrained else None
    model = ctor_map[backbone](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _strip_prefixes(state_dict: dict[str, torch.Tensor], prefixes: Iterable[str]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_resnet_checkpoint(
    checkpoint: str | Path,
    backbone: str = "resnet50",
    num_classes: int = 7,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load either a plain PyTorch state dict or a PyTorch-Lightning checkpoint."""
    checkpoint = Path(checkpoint)
    raw = torch.load(checkpoint, map_location=device)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = _strip_prefixes(state, prefixes=("model.", "module."))

    model = build_resnet(backbone=backbone, num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing checkpoint keys: {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected checkpoint keys: {unexpected[:10]}")
    model.to(device)
    model.eval()
    return model

