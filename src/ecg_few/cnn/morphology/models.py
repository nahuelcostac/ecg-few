"""Models for ECG experiments."""

from __future__ import annotations

from torch import nn
from torchvision import models

from ecg_few.findings import LABEL_NAMES


def build_resnet(resnet_weights: str) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if resnet_weights == "default" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_NAMES))
    return model


def build_model(*, resnet_weights: str) -> nn.Module:
    return build_resnet(resnet_weights)
