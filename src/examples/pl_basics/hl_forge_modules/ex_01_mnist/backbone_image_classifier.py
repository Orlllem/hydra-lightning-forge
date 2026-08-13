"""Backbone image classifier example using BaseLightningModule and BaseLightningDataModule."""

import torch
from torch import nn


class Backbone(nn.Module):
    """Feature extraction backbone for image classification.

    Args:
        in_features: Number of flattened input features (defaults to 28 * 28 for MNIST).
        hidden_dim: Intermediate feature representation dimension.
    """

    def __init__(self, in_features: int = 28 * 28, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))


class ClassificationHead(nn.Module):
    """Linear classification projection head.

    Args:
        in_features: Hidden feature representation dimension.
        num_classes: Number of target output classes.
    """

    def __init__(self, in_features: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
