"""Self-contained tournament inference callbacks."""

from __future__ import annotations

from pathlib import Path

from .model import QNetwork
from .policy import NetworkPolicy
from .types import Action, AgentContext, GameState


def setup(self: AgentContext) -> None:
    model_path = Path(__file__).with_name("model.pt")
    if model_path.exists():
        self.policy = NetworkPolicy.load(model_path)
    else:
        self.logger.warning("No model.pt found; using untrained weights")
        self.policy = NetworkPolicy(QNetwork())


def act(self: AgentContext, game_state: GameState) -> Action:
    epsilon = 0.05 if self.train else 0.0
    return self.policy.act(game_state, epsilon=epsilon)
