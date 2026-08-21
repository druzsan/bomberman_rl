"""Direct spatial encoding of the engine-provided game state."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import BOARD_SIZE, INPUT_SHAPE
from .types import Coordinate, GameState

EncodedState = NDArray[np.uint8]


def _image_coordinate(coordinate: Coordinate) -> tuple[int, int]:
    """Convert an interior engine coordinate (x, y) to image (row, column)."""
    x, y = coordinate
    if not (1 <= x < BOARD_SIZE - 1 and 1 <= y < BOARD_SIZE - 1):
        raise ValueError(f"Entity coordinate {coordinate} lies on/outside the border")
    return y - 1, x - 1


def _validate_border(field: np.ndarray) -> None:
    if field.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"Expected a {BOARD_SIZE}x{BOARD_SIZE} field, got {field.shape}"
        )
    border_is_stone = (
        np.all(field[0, :] == -1)
        and np.all(field[-1, :] == -1)
        and np.all(field[:, 0] == -1)
        and np.all(field[:, -1] == -1)
    )
    if not border_is_stone:
        raise ValueError("The field border is not made entirely of stone walls")


def encode_state(game_state: GameState) -> EncodedState:
    """Encode raw state fields without pathfinding or derived tactical features."""
    field = np.asarray(game_state["field"])
    explosion_map = np.asarray(game_state["explosion_map"])
    _validate_border(field)
    if explosion_map.shape != field.shape:
        raise ValueError(
            "Explosion map shape does not match field shape: "
            f"{explosion_map.shape} != {field.shape}"
        )

    encoded = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    # Engine arrays use [x, y]; Conv2d images use [row=y, column=x].
    interior = field[1:-1, 1:-1].T
    encoded[0] = interior == -1
    encoded[1] = interior == 1

    for coordinate in game_state["coins"]:
        encoded[2][_image_coordinate(coordinate)] = 1

    _, _, can_bomb, own_coordinate = game_state["self"]
    own_pixel = _image_coordinate(own_coordinate)
    encoded[3][own_pixel] = 1
    encoded[8][own_pixel] = int(can_bomb)

    for _, _, opponent_can_bomb, coordinate in game_state["others"]:
        pixel = _image_coordinate(coordinate)
        encoded[4][pixel] = 1
        encoded[9][pixel] = int(opponent_can_bomb)

    for coordinate, timer in game_state["bombs"]:
        if not 0 <= timer <= 254:
            raise ValueError(f"Bomb timer cannot be encoded as uint8: {timer}")
        pixel = _image_coordinate(coordinate)
        encoded[5][pixel] = 1
        encoded[6][pixel] = timer + 1

    explosion_interior = explosion_map[1:-1, 1:-1].T
    if np.any(explosion_interior < 0) or np.any(explosion_interior > 255):
        raise ValueError("Explosion map values must fit in uint8")
    encoded[7] = explosion_interior.astype(np.uint8)
    return encoded
