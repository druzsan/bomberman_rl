"""Feature vector for the linear model (S2a).

The same state analysis as the tabular agent, but encoded as a real vector
instead of a table index: the discrete blocks become one-hot groups and the
distances that the tabular agent had to bucket away are kept as normalised
scalars.  ``Q(s, a) = w_a . phi(s)`` with about 280 parameters, against roughly
600 reachable table entries -- so it should need far fewer samples and it
generalises to states the table has never seen.

The vector is built from the **canonicalised** block values, so the D4 fold
applies here exactly as it does to the table: one update covers all eight
symmetry images.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .features import CARDINALITY, StateInfo

#: Blocks encoded as a one-hot group, in this order.
ONEHOT_BLOCKS = ("target_dir", "wait_ok", "danger_now", "bomb_ready",
                 "bomb_here_value", "opp_dir", "opp_dist", "in_dead_end")

#: ``move_status`` is a base-3 code over four directions; a 81-wide one-hot
#: would waste parameters, so it is expanded as four one-hot(3) groups.
MOVE_STATUS_GROUPS = 4
MOVE_STATUS_LEVELS = 3

#: Normalised scalars, plus a bias. These are what the tabular agent had to
#: discard: it can only see "opponent within 3 tiles", never "2.0 tiles".
SCALARS = ("obj_dist", "opp_dist", "step", "others_alive", "bomb_crates",
           "safe_moves", "bias")

DISTANCE_CAP = 20.0
STEP_CAP = 400.0


def feature_dim(spec) -> int:
    n = sum(CARDINALITY[b] for b in ONEHOT_BLOCKS if b in spec.blocks)
    if "move_status" in spec.blocks:
        n += MOVE_STATUS_GROUPS * MOVE_STATUS_LEVELS
    return n + len(SCALARS)


def build(spec, values, info: StateInfo, out: NDArray[np.float32] | None = None):
    """Encode one canonicalised block tuple plus its scalars.

    ``values`` must already be in the canonical frame; the scalars below are all
    D4 invariants, so they need no transformation.
    """
    d = feature_dim(spec)
    phi = np.zeros(d, dtype=np.float32) if out is None else out
    phi[:] = 0.0
    at = 0
    lookup = dict(zip(spec.blocks, values))
    for block in ONEHOT_BLOCKS:
        if block not in spec.blocks:
            continue
        phi[at + int(lookup[block])] = 1.0
        at += CARDINALITY[block]
    if "move_status" in spec.blocks:
        code = int(lookup["move_status"])
        for k in range(MOVE_STATUS_GROUPS):
            phi[at + (code // 3 ** k) % 3] = 1.0
            at += MOVE_STATUS_LEVELS
    phi[at + 0] = min(max(info.obj_dist, 0), DISTANCE_CAP) / DISTANCE_CAP if info.obj_dist >= 0 else 1.0
    phi[at + 1] = min(max(info.opp_dist, 0), DISTANCE_CAP) / DISTANCE_CAP if info.opp_dist >= 0 else 1.0
    phi[at + 2] = min(info.step, STEP_CAP) / STEP_CAP
    phi[at + 3] = info.n_others / 3.0
    phi[at + 4] = min(info.bomb_crates, 12) / 12.0
    phi[at + 5] = float(info.survivable.sum()) / 5.0
    phi[at + 6] = 1.0
    return phi
