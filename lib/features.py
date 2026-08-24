"""State analysis and discrete feature extraction.

:func:`analyse` does all the expensive work once (danger model, escape search,
objective BFS) and returns a :class:`StateInfo`; the feature tuple, the reward
shaping and ``bfs_expert`` all read from that single object so nothing is
recomputed.

Feature blocks (``dev/plan.md`` §6):

===  ==========================  =========================================  ====
Id   Name                        Values                                     Card
===  ==========================  =========================================  ====
F1   ``target_dir``              0-3 move, 4 here, 5 none                      6
F2   ``escape_dir``              0-3 move, 4 wait, 5 doomed, 6 already safe    7
F3   ``danger_now``              0-3 tau, 4 tau>=4, 5 safe                     6
F4   ``walkable``                4-bit mask over the move directions          16
F5   ``bomb_ready``              bool                                          2
F6   ``bomb_here_value``         0 suicide, 1 pointless, 2 ok, 3 good          4
F7   ``opp_dir``                 0-3 move, 4 none                              5
F8   ``opp_dist``                0 adjacent, 1 <=3, 2 <=6, 3 far/none          4
F9   ``in_dead_end``             bool                                          2
===  ==========================  =========================================  ====
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from numpy.typing import NDArray

from . import danger, pathfind
from .board import DELTAS, N_ACTIONS, crate_adjacent
from .symmetry import ACTION_MAP, BITS4_MAP, DIR5_MAP, DIR6_MAP, INVERSE, N_G
from .types import Coordinate, GameState

# F1 / F2 sentinels
DIR_HERE = 4
DIR_NONE = 5
ESC_WAIT = 4
ESC_DOOMED = 5
ESC_SAFE = 6

OBJ_NONE, OBJ_COIN, OBJ_CRATE, OBJ_OPPONENT = 0, 1, 2, 3

#: Number of distinct values per feature block, keyed by block name.
CARDINALITY = {
    "target_dir": 6,
    "escape_dir": 7,
    "danger_now": 6,
    "walkable": 16,
    "bomb_ready": 2,
    "bomb_here_value": 4,
    "opp_dir": 5,
    "opp_dist": 4,
    "in_dead_end": 2,
}

#: How each block transforms under a D4 group element.
_MAPS = {
    "target_dir": DIR6_MAP,
    "escape_dir": np.concatenate(
        [DIR6_MAP[:, :4], np.full((N_G, 3), [4, 5, 6], dtype=np.int8)], axis=1
    ),
    "danger_now": np.tile(np.arange(6, dtype=np.int8), (N_G, 1)),
    "walkable": BITS4_MAP,
    "bomb_ready": np.tile(np.arange(2, dtype=np.int8), (N_G, 1)),
    "bomb_here_value": np.tile(np.arange(4, dtype=np.int8), (N_G, 1)),
    "opp_dir": DIR5_MAP,
    "opp_dist": np.tile(np.arange(4, dtype=np.int8), (N_G, 1)),
    "in_dead_end": np.tile(np.arange(2, dtype=np.int8), (N_G, 1)),
}

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "TQ-A": ("target_dir", "walkable"),
    "TQ-B": ("target_dir", "escape_dir", "danger_now", "walkable", "bomb_ready",
             "bomb_here_value"),
    "TQ-C": ("target_dir", "escape_dir", "danger_now", "walkable", "bomb_ready",
             "bomb_here_value", "opp_dir", "opp_dist"),
    "TQ-D": ("target_dir", "escape_dir", "danger_now", "walkable", "bomb_ready",
             "bomb_here_value", "opp_dir", "opp_dist", "in_dead_end"),
}


@dataclass(slots=True)
class StateInfo:
    """Everything derived from one ``game_state``, computed once."""

    pos: Coordinate
    bombs_left: bool
    step: int
    n_others: int

    field: NDArray[np.int_]
    passable: NDArray[np.bool_]
    lethal: NDArray[np.uint8]
    blocked: NDArray[np.bool_]
    occupied: NDArray[np.bool_]

    danger_tau: int
    escape: NDArray[np.int16]
    escape_possible: bool
    safe_here: bool

    objective: int
    obj_dist: int
    target_dir: int

    opp_dist: int
    opp_dir_raw: int
    #: Steps for the nearest opponent to reach each tile (-1 beyond the horizon).
    threat: NDArray[np.int16] | None

    #: Bit masks of *all* tied optimal directions.  Exactly D4-equivariant, and
    #: therefore what the symmetry test asserts on; the scalar direction
    #: features below pick the lowest set bit, which may differ between mirrored
    #: states when directions tie.
    target_mask: int
    opp_mask: int
    escape_mask: int

    bomb_crates: int
    bomb_hits: int
    #: Escape exists treating opponents as obstacles only for the current step.
    bomb_safe: bool
    #: Escape exists even if every opponent stands still and blocks its tile,
    #: and even against tiles opponents could reach in time.
    bomb_safe_strict: bool
    #: Escape exists with one step of slack: it still works if every blast went
    #: off one step earlier than it will.  Survives a blocked move.
    bomb_safe_slack: bool

    in_dead_end: bool
    legal: NDArray[np.bool_]
    blocks: dict[str, int] = dc_field(default_factory=dict)


def analyse(game_state: GameState, *, with_opponents: bool = True) -> StateInfo:
    """Compute the full state analysis for one snapshot."""
    field = game_state["field"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    _, _, bombs_left, (sx, sy) = game_state["self"]
    sx, sy = int(sx), int(sy)
    others = game_state["others"]
    coins = game_state["coins"]

    lethal = danger.lethal_bits(field, bombs, explosion_map)
    blocked = danger.blocked_mask(field, bombs)
    occupied = danger.agent_mask(field, others)
    passable = ~blocked

    danger_tau = danger.own_tile_danger(lethal, (sx, sy))
    safe_here = lethal[sx, sy] == 0
    if safe_here:
        escape = np.full(5, -1, dtype=np.int16)
        escape_possible, escape_mask = True, 0
    else:
        escape = danger.escape_taus(lethal, blocked, (sx, sy), blocked_now=occupied)
        escape_possible = danger.survivable(escape)
        escape_mask = 0
        if escape_possible:
            best_tau = int(escape[escape >= 0].min())
            escape_mask = int(((escape == best_tau) * (1 << np.arange(5))).sum())

    # Objective: nearest reachable coin, else a free tile next to a crate, else
    # the nearest opponent.  Each fallback only costs a BFS when the previous
    # one found nothing.
    objective, obj_dist, target_dir, target_mask = OBJ_NONE, -1, DIR_NONE, 0
    if coins:
        dist = pathfind.distance_field(passable, pathfind.mask_of(field.shape, coins),
                                       stop_at=(sx, sy))
        if dist[sx, sy] >= 0:
            objective, obj_dist = OBJ_COIN, int(dist[sx, sy])
            target_mask = pathfind.step_mask(dist, (sx, sy))
            target_dir = _dir_from(pathfind.step_towards(dist, (sx, sy)))
    if objective == OBJ_NONE:
        near_crate = crate_adjacent(field) & passable
        if near_crate.any():
            dist = pathfind.distance_field(passable, near_crate, stop_at=(sx, sy))
            if dist[sx, sy] >= 0:
                objective, obj_dist = OBJ_CRATE, int(dist[sx, sy])
                target_mask = pathfind.step_mask(dist, (sx, sy))
                target_dir = _dir_from(pathfind.step_towards(dist, (sx, sy)))

    threat = None
    if others:
        threat = pathfind.distance_field(passable | occupied, occupied,
                                         max_dist=danger.MAX_TAU + 1)

    opp_dist, opp_dir_raw, opp_mask = -1, 4, 0
    if others and (with_opponents or objective == OBJ_NONE):
        dist = pathfind.distance_field(passable | occupied, occupied, stop_at=(sx, sy))
        if dist[sx, sy] >= 0:
            opp_dist = int(dist[sx, sy])
            opp_mask = pathfind.step_mask(dist, (sx, sy))
            k = pathfind.step_towards(dist, (sx, sy))
            opp_dir_raw = k if 0 <= k < 4 else 4
            if objective == OBJ_NONE:
                objective, obj_dist = OBJ_OPPONENT, opp_dist
                target_mask, target_dir = opp_mask, _dir_from(k)

    if bombs_left:
        bomb_crates, bomb_hits = danger.bomb_value(field, others, (sx, sy))
        bomb_safe = danger.survives_bomb_here(field, bombs, explosion_map, others, (sx, sy))
        bomb_safe_strict = bomb_safe and (
            not others
            or danger.survives_bomb_here(field, bombs, explosion_map, others, (sx, sy),
                                         block_others=True, threat=threat))
        bomb_safe_slack = bomb_safe and _has_slack(field, bombs, explosion_map,
                                                   occupied, (sx, sy))
    else:
        bomb_crates = bomb_hits = 0
        bomb_safe = bomb_safe_strict = bomb_safe_slack = False

    legal = np.zeros(N_ACTIONS, dtype=bool)
    for k, (dx, dy) in enumerate(DELTAS):
        legal[k] = not (blocked[sx + dx, sy + dy] or occupied[sx + dx, sy + dy])
    legal[4] = True
    legal[5] = bool(bombs_left)

    n_free_neighbours = sum(1 for dx, dy in DELTAS if field[sx + dx, sy + dy] == 0)

    info = StateInfo(
        pos=(sx, sy), bombs_left=bool(bombs_left), step=int(game_state["step"]),
        n_others=len(others), field=field, passable=passable, lethal=lethal,
        blocked=blocked, occupied=occupied, danger_tau=danger_tau, escape=escape,
        escape_possible=escape_possible, safe_here=bool(safe_here),
        objective=objective, obj_dist=obj_dist, target_dir=target_dir,
        opp_dist=opp_dist, opp_dir_raw=opp_dir_raw, threat=threat,
        target_mask=target_mask, opp_mask=opp_mask, escape_mask=escape_mask,
        bomb_crates=bomb_crates,
        bomb_hits=bomb_hits, bomb_safe=bomb_safe, bomb_safe_strict=bomb_safe_strict,
        bomb_safe_slack=bomb_safe_slack,
        in_dead_end=n_free_neighbours <= 1, legal=legal,
    )
    info.blocks = _blocks(info)
    return info


def _has_slack(field, bombs, explosion_map, occupied, pos) -> bool:
    """Would the escape from a bomb at ``pos`` survive losing one step?

    Shifting every lethal bit one step earlier (``lethal >> 1``) is exactly
    "all blasts go off one step sooner", so an escape that still exists under
    the shifted map tolerates one blocked move or one wasted step.
    """
    extra = ((int(pos[0]), int(pos[1])),)
    lethal = danger.lethal_bits(field, bombs, explosion_map, extra_bombs=extra) >> 1
    blocked = danger.blocked_mask(field, bombs, extra_bombs=extra)
    return danger.survivable(
        danger.escape_taus(lethal, blocked, pos, blocked_now=occupied))


def _dir_from(k: int) -> int:
    """Map a :func:`pathfind.step_towards` result onto the F1 encoding."""
    if k < 0:
        return DIR_NONE
    if k == 4:
        return DIR_HERE
    return k


def _blocks(info: StateInfo) -> dict[str, int]:
    sx, sy = info.pos
    if info.safe_here:
        escape_dir = ESC_SAFE
    elif not info.escape_possible:
        escape_dir = ESC_DOOMED
    else:
        best = (info.escape_mask & -info.escape_mask).bit_length() - 1
        escape_dir = ESC_WAIT if best == 4 else best

    danger_now = 5 if info.danger_tau < 0 else min(info.danger_tau, 4)

    walkable = 0
    for k, (dx, dy) in enumerate(DELTAS):
        nx, ny = sx + dx, sy + dy
        if (not info.blocked[nx, ny] and not info.occupied[nx, ny]
                and not (info.lethal[nx, ny] & 1)):
            walkable |= 1 << k

    if not info.bombs_left or not info.bomb_safe_strict:
        bomb_here = 0
    elif info.bomb_hits > 0 or info.bomb_crates >= 3:
        bomb_here = 3
    elif info.bomb_crates >= 1:
        bomb_here = 2
    else:
        bomb_here = 1

    if info.opp_dist < 0:
        opp_dist = 3
    elif info.opp_dist <= 1:
        opp_dist = 0
    elif info.opp_dist <= 3:
        opp_dist = 1
    elif info.opp_dist <= 6:
        opp_dist = 2
    else:
        opp_dist = 3

    return {
        "target_dir": info.target_dir,
        "escape_dir": escape_dir,
        "danger_now": danger_now,
        "walkable": walkable,
        "bomb_ready": int(info.bombs_left),
        "bomb_here_value": bomb_here,
        "opp_dir": info.opp_dir_raw,
        "opp_dist": opp_dist,
        "in_dead_end": int(info.in_dead_end),
    }


class FeatureSpec:
    """A named subset of the feature blocks plus its index arithmetic."""

    def __init__(self, name: str):
        if name not in FEATURE_SETS:
            raise KeyError(f"unknown feature set {name!r}")
        self.name = name
        self.blocks: tuple[str, ...] = FEATURE_SETS[name]
        self.cards = tuple(CARDINALITY[b] for b in self.blocks)
        strides = []
        acc = 1
        for c in reversed(self.cards):
            strides.append(acc)
            acc *= c
        self.strides = tuple(reversed(strides))
        self.n_states = acc
        self.needs_opponents = any(b.startswith("opp") for b in self.blocks)
        # Ragged (different cardinalities), and plain lists index far faster
        # than numpy scalars in the per-step hot path.
        self._maps = [[[int(v) for v in row] for row in _MAPS[b]] for b in self.blocks]

    def tuple_of(self, info: StateInfo) -> tuple[int, ...]:
        return tuple(info.blocks[b] for b in self.blocks)

    def index(self, values) -> int:
        idx = 0
        for v, s in zip(values, self.strides):
            idx += int(v) * s
        return idx

    def transform(self, values, g: int) -> tuple[int, ...]:
        return tuple(m[g][v] for m, v in zip(self._maps, values))

    def canonical(self, values) -> tuple[tuple[int, ...], int]:
        """Lexicographically smallest D4 image and the element producing it."""
        best, best_g = tuple(int(v) for v in values), 0
        for g in range(1, N_G):
            cand = self.transform(values, g)
            if cand < best:
                best, best_g = cand, g
        return best, best_g

    def canonical_index(self, info: StateInfo) -> tuple[int, int]:
        values, g = self.canonical(self.tuple_of(info))
        return self.index(values), g


def action_in_frame(action_index: int, g: int) -> int:
    """Map an action index into the frame produced by ``g``."""
    return int(ACTION_MAP[g, action_index])


def action_from_frame(action_index: int, g: int) -> int:
    """Map an action index chosen in the ``g`` frame back to the world frame."""
    return int(ACTION_MAP[INVERSE[g], action_index])
