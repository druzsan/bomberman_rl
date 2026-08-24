"""The dihedral group D4 acting on boards, directions and actions.

The wall layout of every scenario is invariant under all eight symmetries of the
square (the start corners map onto each other), so each of them is a valid data
augmentation and a valid way to fold a Q-table.

Group elements are indexed ``g = 4 * flip + rot``: mirror about the vertical
axis first (if ``flip``), then rotate ``rot`` quarter turns clockwise.  Screen
coordinates are used throughout, i.e. ``y`` grows downwards.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_G = 8

#: ``DIR_MAP[g, d]`` is the image of move direction ``d`` under ``g``.
DIR_MAP = np.array(
    [[(rot + ((4 - d) % 4 if flip else d)) % 4 for d in range(4)]
     for flip in (0, 1) for rot in range(4)],
    dtype=np.int8,
)

#: Images of a value in ``{0..3 directions, 4, 5}`` where 4 and 5 are
#: direction-free sentinels (``HERE``/``WAIT`` and ``NONE``).
DIR6_MAP = np.concatenate([DIR_MAP, np.full((N_G, 2), [4, 5], dtype=np.int8)], axis=1)
DIR5_MAP = np.concatenate([DIR_MAP, np.full((N_G, 1), 4, dtype=np.int8)], axis=1)

#: Images of the 6 actions (moves permute, ``WAIT``/``BOMB`` are fixed).
ACTION_MAP = DIR6_MAP.copy()

#: ``BITS4_MAP[g, m]`` permutes a 4-bit direction mask.
BITS4_MAP = np.zeros((N_G, 16), dtype=np.int8)
for _g in range(N_G):
    for _m in range(16):
        _v = 0
        for _d in range(4):
            if _m & (1 << _d):
                _v |= 1 << int(DIR_MAP[_g, _d])
        BITS4_MAP[_g, _m] = _v

#: ``INVERSE[g]`` is the group element undoing ``g``.
INVERSE = np.zeros(N_G, dtype=np.int8)
for _g in range(N_G):
    for _h in range(N_G):
        if all(DIR_MAP[_h, DIR_MAP[_g, _d]] == _d for _d in range(4)):
            INVERSE[_g] = _h
            break

#: ``COMPOSE[g, h]`` applies ``h`` first, then ``g``.
COMPOSE = np.zeros((N_G, N_G), dtype=np.int8)
for _g in range(N_G):
    for _h in range(N_G):
        img = [int(DIR_MAP[_g, DIR_MAP[_h, _d]]) for _d in range(4)]
        for _k in range(N_G):
            if all(DIR_MAP[_k, _d] == img[_d] for _d in range(4)):
                COMPOSE[_g, _h] = _k
                break


def transform_plane(plane: NDArray, g: int) -> NDArray:
    """Apply ``g`` to the last two axes of a board-shaped array."""
    flip, rot = divmod(int(g), 4)
    out = plane
    if flip:
        out = out[..., ::-1, :]
    if rot:
        # Clockwise quarter turn in screen coordinates: (x, y) -> (n-1-y, x).
        out = np.rot90(out, k=-rot, axes=(-1, -2))
    return np.ascontiguousarray(out)


def transform_coord(x: int, y: int, shape: tuple[int, int], g: int) -> tuple[int, int]:
    """Apply ``g`` to a single coordinate on a board of the given shape."""
    flip, rot = divmod(int(g), 4)
    cols, rows = shape
    if flip:
        x = cols - 1 - x
    for _ in range(rot):
        x, y = rows - 1 - y, x
        cols, rows = rows, cols
    return int(x), int(y)


def transform_state(game_state, g: int):
    """Apply ``g`` to a whole ``game_state`` dict (board planes and coordinates).

    Used by the symmetry unit tests and by D4 data augmentation for S3/S4.
    """
    shape = game_state["field"].shape
    name, score, bombs_left, (sx, sy) = game_state["self"]
    out = dict(game_state)
    out["field"] = transform_plane(game_state["field"], g)
    out["explosion_map"] = transform_plane(game_state["explosion_map"], g)
    out["self"] = (name, score, bombs_left, transform_coord(int(sx), int(sy), shape, g))
    out["others"] = [(n, sc, bl, transform_coord(int(x), int(y), shape, g))
                     for n, sc, bl, (x, y) in game_state["others"]]
    out["bombs"] = [(transform_coord(int(x), int(y), shape, g), t)
                    for (x, y), t in game_state["bombs"]]
    out["coins"] = [transform_coord(int(x), int(y), shape, g) for x, y in game_state["coins"]]
    return out
