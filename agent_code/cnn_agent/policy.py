from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import QNetwork
from .state import action_mask, encode_state
from .types import Action, GameState

ACTIONS: tuple[Action, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")


class NetworkPolicy:
    def __init__(
        self,
        network: QNetwork,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.network = network.to(device)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> NetworkPolicy:
        network = QNetwork()
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        state_dict = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        network.load_state_dict(state_dict)
        network.eval()
        return cls(network, device)

    def act(
        self, game_state: GameState, epsilon: float = 0.0, bombs_enabled: bool = True
    ) -> Action:
        return self.act_many([game_state], epsilon, bombs_enabled)[0]

    def act_many(
        self,
        game_states: list[GameState],
        epsilon: float = 0.0,
        bombs_enabled: bool = True,
    ) -> list[Action]:
        """Choose independently explored actions with one batched network pass."""
        if not game_states:
            return []
        encoded = [encode_state(game_state) for game_state in game_states]
        masks = np.stack([action_mask(game_state) for game_state in game_states])
        if not bombs_enabled:
            masks[:, ACTIONS.index("BOMB")] = False
        with torch.inference_mode():
            q_values = self.network(
                torch.from_numpy(np.stack([item[0] for item in encoded])).to(
                    self.device
                ),
                torch.from_numpy(np.stack([item[1] for item in encoded])).to(
                    self.device
                ),
            )
            masked_q_values = q_values.masked_fill(
                ~torch.from_numpy(masks).to(self.device), -torch.inf
            )
            greedy_actions = masked_q_values.argmax(dim=1).cpu().numpy()

        actions: list[Action] = []
        for index, legal in enumerate(masks):
            if epsilon > 0 and self.rng.random() < epsilon:
                action_index = int(self.rng.choice(np.flatnonzero(legal)))
            else:
                action_index = int(greedy_actions[index])
            actions.append(ACTIONS[action_index])
        return actions
