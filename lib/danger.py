"""The threat model: who dies where, and when.

Two engine facts drive everything here and are the ones naive implementations
get wrong (see ``dev/plan.md`` §2.3/§2.4):

* **Crates do not block blasts.** ``Bomb.get_blast_coords`` only breaks on
  ``field == -1``, so a bomb reaches three tiles per direction through crates
  and you can be killed through a crate.
* **A blast is lethal for two consecutive steps.** A bomb observed with timer
  ``t`` detonates at the end of the step ``t`` actions from now and its tiles
  kill at ``t`` *and* ``t + 1``. The second step is only ever visible through
  ``explosion_map``, never through ``bombs``.

Time is measured in ``tau``: ``tau = 0`` is the step whose action we are about
to choose. "Lethal at ``tau``" means *standing on that tile at the end of step
now + tau kills you*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .board import BOMB_POWER, DELTAS, WALL, spread
from .types import BombState, Coordinate, ExplosionMap, Field, PlayerState

#: Highest ``tau`` representable in the uint8 bit field.
MAX_TAU = 7
#: A freshly dropped bomb kills at ``tau = 4`` and ``tau = 5``.
FRESH_BOMB_TAU = 4


def blast_coords(field: Field, x: int, y: int, power: int = BOMB_POWER) -> list[Coordinate]:
    """Tiles hit by a bomb at ``(x, y)``. Only ``WALL`` blocks; crates do not."""
    coords = [(x, y)]
    for dx, dy in DELTAS:
        for i in range(1, power + 1):
            nx, ny = x + i * dx, y + i * dy
            if field[nx, ny] == WALL:
                break
            coords.append((nx, ny))
    return coords


def lethal_bits(
    field: Field,
    bombs: list[BombState],
    explosion_map: ExplosionMap | None,
    extra_bombs: tuple[Coordinate, ...] = (),
) -> NDArray[np.uint8]:
    """Bit ``tau`` is set iff being on that tile at the end of step ``now+tau`` kills.

    ``extra_bombs`` are hypothetical bombs dropped *this* step (used to answer
    "what if I bomb here?"); they contribute bits ``4`` and ``5``.
    """
    out = np.zeros(field.shape, dtype=np.uint8)
    if explosion_map is not None:
        out[explosion_map >= 1] |= 1
    for (bx, by), t in bombs:
        if t > MAX_TAU:
            continue
        mask = np.uint8((1 << int(t)) | (1 << min(int(t) + 1, MAX_TAU)))
        for cx, cy in blast_coords(field, int(bx), int(by)):
            out[cx, cy] |= mask
    if extra_bombs:
        mask = np.uint8((1 << FRESH_BOMB_TAU) | (1 << (FRESH_BOMB_TAU + 1)))
        for bx, by in extra_bombs:
            for cx, cy in blast_coords(field, int(bx), int(by)):
                out[cx, cy] |= mask
    return out


def blocked_mask(
    field: Field,
    bombs: list[BombState],
    extra_bombs: tuple[Coordinate, ...] = (),
) -> NDArray[np.bool_]:
    """Tiles that cannot be *entered*: walls, crates and bombs.

    Standing still is always legal, so this never applies to the tile the agent
    is already on.  Opponents are handled separately (:func:`agent_mask`)
    because they only block for the current step.
    """
    out = field != 0
    for (bx, by), _ in bombs:
        out[int(bx), int(by)] = True
    for bx, by in extra_bombs:
        out[int(bx), int(by)] = True
    return out


def agent_mask(field: Field, others: list[PlayerState]) -> NDArray[np.bool_]:
    """Tiles occupied by living opponents."""
    out = np.zeros(field.shape, dtype=bool)
    for _, _, _, (ox, oy) in others:
        out[int(ox), int(oy)] = True
    return out


def escape_taus(
    lethal: NDArray[np.uint8],
    blocked: NDArray[np.bool_],
    start: Coordinate,
    blocked_now: NDArray[np.bool_] | None = None,
    horizon: int = MAX_TAU,
    threat: NDArray[np.int16] | None = None,
) -> NDArray[np.int16]:
    """Time-expanded reachability search over ``(tile, tau)``, per first action.

    Returns an ``int16`` array of length 5 indexed by ``UP, RIGHT, DOWN, LEFT,
    WAIT``: the smallest ``tau`` at which that first action can put the agent on
    a tile that is safe from ``tau`` onwards (``lethal >> tau == 0``), or ``-1``
    if no such escape exists within ``horizon``.

    ``blocked`` applies from ``tau = 0`` on; ``blocked_now`` adds obstacles that
    only hold for the first step (opponents, which move afterwards).  The search
    is conservative in our favour on two counts: crates that the pending blast
    will destroy still count as walls, and bombs never disappear.

    ``threat`` is an opponent reach-time field (see
    :func:`lib.pathfind.distance_field` from the opponents): a tile is treated
    as unavailable from ``tau >= threat[tile]``, i.e. once an opponent could be
    standing on it.  Measurement drove this in: with opponents modelled as
    *static* obstacles the expert bombed itself into one-wide corridors that an
    opponent then walked into.
    """
    sx, sy = int(start[0]), int(start[1])
    n_first = 5
    alive = np.zeros((n_first,) + lethal.shape, dtype=bool)
    first_blocked = blocked if blocked_now is None else (blocked | blocked_now)
    for k, (dx, dy) in enumerate(DELTAS):
        nx, ny = sx + dx, sy + dy
        if not first_blocked[nx, ny]:
            alive[k, nx, ny] = True
    alive[4, sx, sy] = True  # WAIT is always legal

    out = np.full(n_first, -1, dtype=np.int16)
    pending = np.ones(n_first, dtype=bool)
    for tau in range(horizon + 1):
        if tau > 0:
            moved = spread(alive)
            moved[:, blocked] = False
            alive = alive | moved
            if threat is not None:
                alive[:, (threat >= 0) & (threat <= tau)] = False
        alive[:, (lethal & (1 << tau)) != 0] = False
        safe = (lethal >> tau) == 0
        reached = (alive & safe).any(axis=(1, 2)) & pending
        out[reached] = tau
        pending &= ~reached
        if not pending.any() or not alive.any():
            break
    return out


def survivable(taus: NDArray[np.int16]) -> bool:
    """Does any first action lead to a tile that is safe from then on?"""
    return bool((taus >= 0).any())


def survives_bomb_here(
    field: Field,
    bombs: list[BombState],
    explosion_map: ExplosionMap | None,
    others: list[PlayerState],
    pos: Coordinate,
    block_others: bool = False,
    threat: NDArray[np.int16] | None = None,
) -> bool:
    """Would dropping a bomb at ``pos`` still leave an escape route?

    The hypothetical bomb also blocks its own tile from ``tau >= 1``, which is
    exactly what the engine does (``tile_is_free`` counts bombs as obstacles).

    ``block_others`` treats opponents as obstacles for the *whole* escape, not
    just the current step.  Measured on 30 rounds, the optimistic model (the
    default) lets the agent bomb itself into a one-wide corridor that an
    opponent then walks into, which caused 18 suicides in 30 rounds; the
    pessimistic model is the right one for deciding to *create* danger, while
    the optimistic one stays right for escaping danger that already exists.
    """
    extra = ((int(pos[0]), int(pos[1])),)
    lethal = lethal_bits(field, bombs, explosion_map, extra_bombs=extra)
    blocked = blocked_mask(field, bombs, extra_bombs=extra)
    occupied = agent_mask(field, others)
    if block_others:
        taus = escape_taus(lethal, blocked | occupied, pos, threat=threat)
    else:
        taus = escape_taus(lethal, blocked, pos, blocked_now=occupied, threat=threat)
    return survivable(taus)


def bomb_value(field: Field, others: list[PlayerState], pos: Coordinate) -> tuple[int, int]:
    """``(crates destroyed, opponents currently inside the blast)`` for a bomb at ``pos``."""
    coords = blast_coords(field, int(pos[0]), int(pos[1]))
    crates = sum(1 for cx, cy in coords if field[cx, cy] == 1)
    opp = {(int(ox), int(oy)) for _, _, _, (ox, oy) in others}
    hits = sum(1 for c in coords if c in opp)
    return crates, hits


def own_tile_danger(lethal: NDArray[np.uint8], pos: Coordinate) -> int:
    """Smallest ``tau`` at which the agent's own tile becomes lethal, or ``-1``."""
    bits = int(lethal[int(pos[0]), int(pos[1])])
    if bits == 0:
        return -1
    return int((bits & -bits).bit_length() - 1)
