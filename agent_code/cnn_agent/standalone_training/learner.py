from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..model import QNetwork
from .replay import ReplayBuffer


class DoubleDQNLearner:
    def __init__(
        self,
        replay: ReplayBuffer,
        learning_rate: float,
        batch_size: int,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.online = QNetwork().to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = replay
        self.batch_size = batch_size
        self.updates = 0

    def update(self) -> float:
        batch = self.replay.sample(self.batch_size)
        spatial = self._tensor(batch["spatial"])
        globals_ = self._tensor(batch["globals"])
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = self._tensor(batch["rewards"])
        next_spatial = self._tensor(batch["next_spatial"])
        next_globals = self._tensor(batch["next_globals"])
        next_masks = torch.from_numpy(batch["next_masks"]).to(self.device)
        terminated = self._tensor(batch["terminated"])
        discounts = self._tensor(batch["discounts"])

        predicted = (
            self.online(spatial, globals_).gather(1, actions[:, None]).squeeze(1)
        )
        with torch.no_grad():
            online_next = self.online(next_spatial, next_globals).masked_fill(
                ~next_masks, -torch.inf
            )
            next_actions = online_next.argmax(dim=1)
            target_next = (
                self.target(next_spatial, next_globals)
                .gather(1, next_actions[:, None])
                .squeeze(1)
            )
            expected = rewards + (1.0 - terminated) * discounts * target_next

        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self.device)
