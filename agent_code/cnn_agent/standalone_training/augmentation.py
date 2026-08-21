from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..policy import ACTIONS


def transform_spatial(spatial: NDArray, symmetry: int) -> NDArray:
    """Apply one of four rotations, optionally followed by a horizontal flip."""
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be in [0, 7]")
    transformed = np.rot90(spatial, k=symmetry % 4, axes=(-2, -1))
    if symmetry >= 4:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def transform_action(action: int, symmetry: int) -> int:
    if action >= 4:
        return action
    # Image-coordinate vectors in ACTIONS order: up, right, down, left.
    row, col = ((-1, 0), (0, 1), (1, 0), (0, -1))[action]
    for _ in range(symmetry % 4):
        row, col = -col, row  # np.rot90: old right becomes old up.
    if symmetry >= 4:
        col = -col
    return ((-1, 0), (0, 1), (1, 0), (0, -1)).index((row, col))


def transform_mask(mask: NDArray[np.bool_], symmetry: int) -> NDArray[np.bool_]:
    output = np.empty_like(mask)
    for action in range(len(ACTIONS)):
        output[transform_action(action, symmetry)] = mask[action]
    return output
