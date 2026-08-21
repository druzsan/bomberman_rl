"""Small residual Q-network for the 15x15 Bomberman board."""

from __future__ import annotations

import torch
from torch import nn

from .state import GLOBAL_FEATURES, SPATIAL_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class QNetwork(nn.Module):
    def __init__(
        self,
        spatial_channels: int = SPATIAL_CHANNELS,
        global_features: int = GLOBAL_FEATURES,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4 + global_features, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, spatial: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(spatial).flatten(1)
        return self.head(torch.cat((embedding, globals_), dim=1))
