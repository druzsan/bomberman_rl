"""Board constants and direction tables.

The game-rule constants are *local copies* of the stock ``settings.py`` values.
Training monkeypatches ``settings`` (board size, crate density, step limit);
those knobs never change the rules encoded here, and duplicating them keeps the
shipped agent independent of a patched module.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

WALL = -1
FREE = 0
CRATE = 1

BOMB_TIMER = 4
BOMB_POWER = 3
EXPLOSION_TIMER = 2

#: Index order used everywhere: the first four entries are the moves, clockwise
#: starting at UP, which is what the D4 direction maps in :mod:`lib.symmetry`
#: assume.
ACTIONS: tuple[str, ...] = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

UP, RIGHT, DOWN, LEFT, WAIT, BOMB = range(N_ACTIONS)

#: ``(dx, dy)`` per move index. ``y`` grows downwards, as in the GUI.
DELTAS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
MOVE_ACTIONS: tuple[str, ...] = ACTIONS[:4]


def neighbours(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """The four von-Neumann neighbours of ``(x, y)`` in ``DELTAS`` order."""
    return ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))


def spread(a: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Dilate a boolean plane by one step in the four move directions.

    Operates on the last two axes, so a stack of planes can be expanded at once.
    Unlike ``np.roll`` this does not wrap around the border.
    """
    out = a.copy()
    out[..., 1:, :] |= a[..., :-1, :]
    out[..., :-1, :] |= a[..., 1:, :]
    out[..., :, 1:] |= a[..., :, :-1]
    out[..., :, :-1] |= a[..., :, 1:]
    return out


def crate_adjacent(field: NDArray[np.int_]) -> NDArray[np.bool_]:
    """Free tiles that have at least one crate as a von-Neumann neighbour."""
    return spread(field == CRATE) & (field == FREE)


def assert_stock_layout(field: NDArray[np.int_]) -> None:
    """Sanity-check the wall structure the danger model relies on.

    The outer ring must be solid wall (so blast rays and neighbour loops never
    need bounds checks) and interior walls sit exactly at even/even coordinates.
    Called once from ``setup``; cheap enough to leave enabled.
    """
    cols, rows = field.shape
    assert (field[0, :] == WALL).all() and (field[-1, :] == WALL).all()
    assert (field[:, 0] == WALL).all() and (field[:, -1] == WALL).all()
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    pillars = ((xs + 1) * (ys + 1) % 2 == 1)
    assert (field[pillars] == WALL).all()
