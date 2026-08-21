"""Convolutional action-value model and Bellman target helper."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import (
    BOMB_TIMER_SCALE,
    EXPLOSION_SCALE,
    INPUT_CHANNELS,
    INTERIOR_SIZE,
)


class DQN(nn.Module):
    """Small CNN adapted from Atari DQN to the 15 x 15 playable board."""

    def __init__(self, action_count: int) -> None:
        super().__init__()
        self.register_buffer(
            "channel_scales",
            torch.tensor(
                [1, 1, 1, 1, 1, 1, BOMB_TIMER_SCALE, EXPLOSION_SCALE, 1, 1],
                dtype=torch.float32,
            ).view(1, INPUT_CHANNELS, 1, 1),
        )
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        feature_side = (INTERIOR_SIZE + 1) // 2
        feature_side = (feature_side + 1) // 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_side * feature_side, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 4:
            raise ValueError(
                "DQN expects (batch, channels, height, width), "
                f"got {tuple(observations.shape)}"
            )
        expected = (INPUT_CHANNELS, INTERIOR_SIZE, INTERIOR_SIZE)
        if tuple(observations.shape[1:]) != expected:
            raise ValueError(
                f"DQN expects observation shape {expected}, "
                f"got {tuple(observations.shape[1:])}"
            )
        normalized = observations.to(dtype=torch.float32) / self.channel_scales
        return self.head(self.features(normalized))


def bellman_targets(
    rewards: Tensor,
    next_q_values: Tensor,
    dones: Tensor,
    gamma: float,
) -> Tensor:
    """Compute one-step Q-learning targets without terminal bootstrapping."""
    return rewards + gamma * next_q_values * (~dones)
