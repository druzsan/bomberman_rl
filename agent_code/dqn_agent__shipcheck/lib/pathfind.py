"""Deterministic breadth-first search utilities.

Unlike ``rule_based_agent``'s ``look_for_targets`` these never shuffle, so a
feature vector is a pure function of the game state and experiments reproduce.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .board import DELTAS, spread
from .types import Coordinate

UNREACHABLE = np.int16(-1)


def distance_field(
    passable: NDArray[np.bool_],
    sources: NDArray[np.bool_],
    stop_at: Coordinate | None = None,
    max_dist: int = 64,
) -> NDArray[np.int16]:
    """Multi-source BFS distance in steps, ``-1`` where unreachable.

    ``sources`` themselves get distance 0 even when they are not passable (a
    crate or an opponent can be a target without being walkable); expansion
    beyond the first ring only goes through ``passable`` tiles.  ``stop_at``
    ends the search early once that tile has been labelled, which is what makes
    this cheap enough to call several times per step.
    """
    dist = np.full(passable.shape, UNREACHABLE, dtype=np.int16)
    frontier = sources
    dist[frontier] = 0
    if stop_at is not None and dist[stop_at] >= 0:
        return dist
    for d in range(1, max_dist + 1):
        frontier = spread(frontier) & passable & (dist < 0)
        if not frontier.any():
            break
        dist[frontier] = d
        if stop_at is not None and dist[stop_at] >= 0:
            break
    return dist


def step_towards(
    dist: NDArray[np.int16],
    start: Coordinate,
    blocked: NDArray[np.bool_] | None = None,
) -> int:
    """First move index of a shortest path down ``dist``, or ``-1``.

    ``dist`` is a field produced by :func:`distance_field` with the objective as
    source.  Ties break in ``DELTAS`` order (UP, RIGHT, DOWN, LEFT) so the
    feature is deterministic.  Returns ``-1`` when the objective is unreachable
    and ``4`` (``WAIT``) when the agent already stands on it.
    """
    sx, sy = int(start[0]), int(start[1])
    d = int(dist[sx, sy])
    if d < 0:
        return -1
    if d == 0:
        return 4
    for k, (dx, dy) in enumerate(DELTAS):
        nx, ny = sx + dx, sy + dy
        if dist[nx, ny] == d - 1 and (blocked is None or not blocked[nx, ny]):
            return k
    return -1


def mask_of(shape: tuple[int, int], coords) -> NDArray[np.bool_]:
    """Boolean plane with the given coordinates set."""
    out = np.zeros(shape, dtype=bool)
    for c in coords:
        out[int(c[0]), int(c[1])] = True
    return out


def step_mask(dist: NDArray[np.int16], start: Coordinate) -> int:
    """Bit mask of all move directions that start a shortest path down ``dist``.

    :func:`step_towards` picks the lowest set bit of this mask.  The mask itself
    is exactly D4-equivariant (a permutation of its bits), which is what the
    symmetry test checks; the single-direction feature can only differ between
    mirrored states when several directions tie, and either choice is a valid
    shortest path.
    """
    sx, sy = int(start[0]), int(start[1])
    d = int(dist[sx, sy])
    if d <= 0:
        return 0
    mask = 0
    for k, (dx, dy) in enumerate(DELTAS):
        if dist[sx + dx, sy + dy] == d - 1:
            mask |= 1 << k
    return mask
