"""Board-plane encoding for the deep models (S3/S4).

The tabular and linear agents read a hand-designed discrete tuple
(:mod:`lib.features`); the deep model instead gets the board itself and learns
its own features.  This module is the single definition of that encoding, used
by three parties that must never disagree:

* the shipped ``callbacks.act`` (one state, built directly);
* the actor processes (one state, built and then **bit-packed** into the replay
  buffer);
* the learner (a whole minibatch, unpacked on the GPU).

Keeping one function produce the bits and letting both the packer and the
direct path use it is what makes the three consistent by construction -- the
classic "training-time and inference-time features drifted apart" bug has no
place to hide.

Layout: ``(channel, x, y)``, ``x`` the column, ``y`` growing downwards, matching
``game_state["field"]`` and the D4 maps in :mod:`lib.symmetry`.

===  ==========================  ==================================================
Ch   Plane                       Meaning
===  ==========================  ==================================================
0    ``wall``                    ``field == -1``
1    ``crate``                   ``field == 1``
2    ``coin``                    collectable coins
3    ``self``                    our tile (also used as the read-out selector)
4    ``others``                  every living opponent
5    ``others_bomb``             opponents that still have a bomb available
6-9  ``bomb_t0 .. bomb_t3``      one-hot bomb timer at the bomb's tile
10   ``explosion``               ``explosion_map >= 1`` (lethal *now*)
11-16 ``lethal_tau0 .. tau5``    decoded from :func:`lib.danger.lethal_bits`
17   ``bombs_left``              broadcast constant, our bomb availability
18   ``step``                    broadcast constant, ``step / MAX_STEPS``
===  ==========================  ==================================================

Channels 0-16 are binary and are what gets bit-packed; 17-18 are reconstructed
from two small integers.  Channels 11-16 are the one piece of domain knowledge
injected into the deep model -- a network *can* learn blast timing from 6-10,
but it costs samples.  :data:`PLANE_SETS` exposes the ablation ("full" vs
"nodanger") as a decode-time channel selection, so both variants read the *same*
replay buffer and the comparison needs no second data collection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import danger
from .board import CRATE, WALL
from .types import GameState

#: Binary planes, in channel order.
BINARY_PLANES: tuple[str, ...] = (
    "wall", "crate", "coin", "self", "others", "others_bomb",
    "bomb_t0", "bomb_t1", "bomb_t2", "bomb_t3", "explosion",
    "lethal_tau0", "lethal_tau1", "lethal_tau2", "lethal_tau3",
    "lethal_tau4", "lethal_tau5",
)
#: Broadcast scalar planes, appended after the binary ones.
SCALAR_PLANES: tuple[str, ...] = ("bombs_left", "step")

N_BINARY = len(BINARY_PLANES)
N_SCALAR = len(SCALAR_PLANES)
N_PLANES = N_BINARY + N_SCALAR

PLANE_INDEX = {name: i for i, name in enumerate(BINARY_PLANES + SCALAR_PLANES)}

#: The danger block, i.e. everything a "no domain priors" ablation removes.
DANGER_CHANNELS = tuple(PLANE_INDEX[f"lethal_tau{t}"] for t in range(6))

#: Named channel subsets.  The buffer always stores every channel; a run picks
#: one of these at decode time, so an ablation is a config diff.
PLANE_SETS: dict[str, tuple[int, ...]] = {
    "full": tuple(range(N_PLANES)),
    "nodanger": tuple(c for c in range(N_PLANES) if c not in DANGER_CHANNELS),
}

BOARD = 17
CELLS = BOARD * BOARD
#: Bytes needed for the bit-packed binary planes of one state.
PACKED_BYTES = (N_BINARY * CELLS + 7) // 8

#: Normalisation for the broadcast step counter; a local copy of the stock
#: ``settings.MAX_STEPS`` so a curriculum monkeypatch cannot change the encoding.
MAX_STEPS = 400


def n_channels(plane_set: str = "full") -> int:
    return len(PLANE_SETS[plane_set])


def binary_planes(game_state: GameState, lethal: NDArray[np.uint8] | None = None,
                  ) -> NDArray[np.bool_]:
    """The ``(17, cols, rows)`` boolean stack.

    ``lethal`` is :func:`lib.danger.lethal_bits` for this state; passing the
    value already computed by :func:`lib.features.analyse` avoids recomputing
    the most expensive part of the encoding.
    """
    field = game_state["field"]
    shape = field.shape
    out = np.zeros((N_BINARY,) + shape, dtype=bool)

    out[0] = field == WALL
    out[1] = field == CRATE
    for cx, cy in game_state["coins"]:
        out[2, int(cx), int(cy)] = True
    _, _, _, (sx, sy) = game_state["self"]
    out[3, int(sx), int(sy)] = True
    for _, _, other_bomb, (ox, oy) in game_state["others"]:
        out[4, int(ox), int(oy)] = True
        if other_bomb:
            out[5, int(ox), int(oy)] = True
    for (bx, by), timer in game_state["bombs"]:
        # Observed timers are 0..3 (measured); anything else is clamped rather
        # than dropped, so a rule change cannot silently blank the plane.
        out[6 + min(max(int(timer), 0), 3), int(bx), int(by)] = True
    out[10] = game_state["explosion_map"] >= 1

    if lethal is None:
        lethal = danger.lethal_bits(field, game_state["bombs"],
                                    game_state["explosion_map"])
    for tau in range(6):
        out[11 + tau] = (lethal & np.uint8(1 << tau)) != 0
    return out


def scalars(game_state: GameState) -> tuple[float, float]:
    """The two broadcast constants."""
    _, _, bombs_left, _ = game_state["self"]
    return float(bool(bombs_left)), min(int(game_state["step"]), MAX_STEPS) / MAX_STEPS


def planes(game_state: GameState, lethal: NDArray[np.uint8] | None = None,
           plane_set: str = "full") -> NDArray[np.float32]:
    """``(C, cols, rows)`` float32 input for the network, one state."""
    binary = binary_planes(game_state, lethal)
    shape = binary.shape[1:]
    bombs_left, step = scalars(game_state)
    full = np.empty((N_PLANES,) + shape, dtype=np.float32)
    full[:N_BINARY] = binary
    full[N_BINARY] = bombs_left
    full[N_BINARY + 1] = step
    if plane_set == "full":
        return full
    return np.ascontiguousarray(full[list(PLANE_SETS[plane_set])])


def pack(game_state: GameState, lethal: NDArray[np.uint8] | None = None,
         ) -> NDArray[np.uint8]:
    """Bit-pack the binary planes of one state into :data:`PACKED_BYTES` bytes.

    615 bytes per state instead of 22 kB of float planes is what makes a
    million-transition replay buffer fit in memory without any compression
    scheme of our own.
    """
    binary = binary_planes(game_state, lethal)
    if binary.shape[1:] != (BOARD, BOARD):
        raise ValueError(f"packed storage assumes a {BOARD}x{BOARD} board, "
                         f"got {binary.shape[1:]}")
    return np.packbits(binary.reshape(-1))


def unpack_batch(packed: NDArray[np.uint8], step: NDArray[np.uint16],
                 bombs_left: NDArray[np.uint8],
                 plane_set: str = "full") -> NDArray[np.float32]:
    """Inverse of :func:`pack` for a whole batch: ``(B, C, 17, 17)`` float32.

    The GPU path in :mod:`training.dqn_learner` mirrors this in torch; this
    numpy version is what the conformance test compares it against.
    """
    bits = np.unpackbits(packed, axis=1)[:, :N_BINARY * CELLS]
    out = np.empty((packed.shape[0], N_PLANES, BOARD, BOARD), dtype=np.float32)
    out[:, :N_BINARY] = bits.reshape(-1, N_BINARY, BOARD, BOARD)
    out[:, N_BINARY] = bombs_left.astype(np.float32)[:, None, None]
    out[:, N_BINARY + 1] = (np.minimum(step, MAX_STEPS) / MAX_STEPS
                            ).astype(np.float32)[:, None, None]
    if plane_set == "full":
        return out
    return np.ascontiguousarray(out[:, list(PLANE_SETS[plane_set])])


def bits_to_mask(values: NDArray[np.bool_]) -> int:
    """Pack a boolean per-action array into an integer bit mask."""
    return int(np.packbits(values, bitorder="little")[0])


def mask_to_bits(mask: int, n: int) -> NDArray[np.bool_]:
    """Inverse of :func:`bits_to_mask`."""
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
