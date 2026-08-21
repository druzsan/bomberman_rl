from __future__ import annotations

import logging
from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

type Action = Literal["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
type Coordinate = tuple[int, int]
type PlayerState = tuple[str, int, bool, Coordinate]
type BombState = tuple[Coordinate, int]


class GameState(TypedDict):
    round: int
    step: int
    field: NDArray[np.integer]
    self: PlayerState
    others: list[PlayerState]
    bombs: list[BombState]
    coins: list[Coordinate]
    user_input: Action | None
    explosion_map: NDArray[np.floating]


class AgentContext(Protocol):
    logger: logging.Logger
    train: bool
