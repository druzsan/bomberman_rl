"""Engine-facing state and persistent callback-context types."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import DQN
    from .replay import ReplayMemory, TransitionKey

type Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
type Coordinate = tuple[int, int]
type PlayerState = tuple[str, int, bool, Coordinate]
type BombState = tuple[Coordinate, int]
type Field = NDArray[np.int_]
type ExplosionMap = NDArray[np.float64]


class GameState(TypedDict):
    round: int
    step: int
    field: Field
    self: PlayerState
    others: list[PlayerState]
    bombs: list[BombState]
    coins: list[Coordinate]
    user_input: Action | None
    explosion_map: ExplosionMap


class AgentContext(Protocol):
    """Attributes owned by the engine or initialized by this agent."""

    logger: logging.Logger
    train: bool
    device: torch.device
    rng: np.random.Generator
    model: DQN
    target_model: DQN
    optimizer: Optimizer
    replay: ReplayMemory
    loaded_checkpoint: dict[str, Any] | None
    environment_step: int
    optimizer_step: int
    completed_rounds: int
    epsilon: float
    last_transition_key: TransitionKey | None
    last_transition_index: int | None
    round_return: float
    round_steps: int
    round_events: Counter[str]
    round_losses: list[float]
    round_gradient_norms: list[float]
    round_action_times: list[float]
    round_q_values: list[float]
    round_started_at: float
