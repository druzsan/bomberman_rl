"""Shared factorized state encoder used by training and tournament inference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import GameState

BOARD_SIZE = 15
SPATIAL_CHANNELS = 12
GLOBAL_FEATURES = 2

WALLS = 0
CRATES = 1
COINS = 2
SELF = 3
OPPONENTS = 4
OPPONENT_CAN_BOMB = 5
BOMB_TIMER_START = 6
EXPLOSIONS = 11


def encode_state(
    game_state: GameState,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Encode the playable field as image-order ``(channel, y, x)`` planes."""
    field = np.asarray(game_state["field"])
    if field.shape != (17, 17):
        raise ValueError(f"Expected a 17x17 field, got {field.shape}")

    planes = np.zeros((SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    interior = field[1:-1, 1:-1].T
    planes[WALLS] = interior == -1
    planes[CRATES] = interior == 1

    def mark(channel: int, position: tuple[int, int], value: float = 1.0) -> None:
        x, y = position
        if not (1 <= x <= BOARD_SIZE and 1 <= y <= BOARD_SIZE):
            raise ValueError(f"Entity outside playable interior: {(x, y)}")
        planes[channel, y - 1, x - 1] = value

    for coin in game_state["coins"]:
        mark(COINS, coin)
    mark(SELF, game_state["self"][3])
    for opponent in game_state["others"]:
        mark(OPPONENTS, opponent[3])
        if opponent[2]:
            mark(OPPONENT_CAN_BOMB, opponent[3])
    for position, countdown in game_state["bombs"]:
        if not 0 <= countdown <= 4:
            raise ValueError(f"Bomb countdown must be in [0, 4], got {countdown}")
        mark(BOMB_TIMER_START + countdown, position)

    explosion_map = np.asarray(game_state["explosion_map"])[1:-1, 1:-1].T
    planes[EXPLOSIONS] = explosion_map > 0
    globals_ = np.asarray(
        [float(game_state["self"][2]), min(game_state["step"], 400) / 400.0],
        dtype=np.float32,
    )
    return planes, globals_


def action_mask(game_state: GameState) -> NDArray[np.bool_]:
    """Return actions legal under the observable, current state."""
    field = np.asarray(game_state["field"])
    x, y = game_state["self"][3]
    occupied = {position for position, _ in game_state["bombs"]}
    occupied.update(player[3] for player in game_state["others"])
    destinations = ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
    moves = [field[pos] == 0 and pos not in occupied for pos in destinations]
    can_bomb = game_state["self"][2] and (x, y) not in occupied
    return np.asarray([*moves, True, can_bomb], dtype=np.bool_)
