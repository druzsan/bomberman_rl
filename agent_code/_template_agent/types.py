"""Types exposed by the game engine to agent callbacks.

This module deliberately lives beside the callbacks so the entire directory can
be copied and renamed when starting a new agent.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

type Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
type Coordinate = tuple[int, int]
type PlayerState = tuple[str, int, bool, Coordinate]
type BombState = tuple[Coordinate, int]
type Field = NDArray[np.int_]
type ExplosionMap = NDArray[np.float64]


class GameState(TypedDict):
    """Snapshot passed to ``act`` and the training callbacks."""

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
    """Engine-owned object passed as the first callback argument.

    Add attributes here when your implementation stores a model or other state
    on the context during ``setup`` or ``setup_training``.
    """

    logger: logging.Logger
    train: bool
