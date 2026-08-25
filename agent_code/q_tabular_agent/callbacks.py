"""Inference for the tabular Q-learning agent.

The model is a plain ``(n_states, 6)`` float32 table over the discrete feature
tuple of :mod:`lib.features`, folded over the eight symmetries of the square:
the feature tuple is canonicalised, the table is read at the canonical entry,
and the chosen action is mapped back through the inverse group element.  That
makes every experience count eight times and shrinks the table by the same
factor.

Inference is pure numpy, roughly 0.25 ms per step -- about 2000x inside the
tournament's 0.5 s budget -- and needs no torch.

The training driver injects live shared-memory tables through
:data:`TRAINING_TABLES`; when that is ``None`` (always, in the tournament) the
model is loaded from ``model.npz`` next to this file.  The engine chdirs into
the agent directory around every callback, so the relative path is correct.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .lib import danger
from .lib.board import ACTIONS, N_ACTIONS
from .lib.features import FeatureSpec, analyse
from .lib.symmetry import ACTION_MAP, INVERSE
from .lib.types import Action, AgentContext, GameState

#: Default artifact name, resolved relative to the agent directory (the engine
#: chdirs there around every callback).
MODEL_FILE = "model.npz"

#: Set by ``training.worker`` to a namespace with ``q``/``n``/``ctrl``/
#: ``feature_set``/``fold``. Keeping the injection here means the shipped agent
#: never imports ``multiprocessing``.
TRAINING_TABLES = None

#: Tie-breaking seed. The reference opponents call ``np.random.seed()`` in their
#: own ``setup``, reseeding the global numpy RNG from entropy; owning a
#: generator keeps our behaviour reproducible and theirs undisturbed.
SEED = 20260824


def setup(self: AgentContext) -> None:
    self.rng = np.random.default_rng(SEED)
    if TRAINING_TABLES is not None:
        tables = TRAINING_TABLES
        self.tables = tables
        self.Q = tables.q
        self.N = tables.n
        self.ctrl = tables.ctrl
        self.spec = FeatureSpec(tables.feature_set)
        self.fold = bool(tables.fold)
        self.use_mask = False
    else:
        # The evaluation harness points one process at one checkpoint through
        # this variable; the graders never set it, so the default must work.
        path = Path(os.environ.get("BOMBERMAN_MODEL", MODEL_FILE))
        with np.load(path) as data:
            self.spec = FeatureSpec(str(data["feature_set"]))
            self.fold = bool(data["fold"])
            # Stored sparsely: only states the agent ever visited carry values,
            # which keeps a 15 MB table under a megabyte on disk.
            self.Q = np.zeros((self.spec.n_states, N_ACTIONS), dtype=np.float32)
            self.Q[data["states"]] = data["values"]
            self.N = None
            self.use_mask = bool(data["use_mask"]) if "use_mask" in data.files else False
        self.ctrl = None
        self.logger.info("loaded %s: %s, %d states", path, self.spec.name, self.spec.n_states)
    self._cache_key = None
    self._cache_info = None
    self.current_round = -1


def state_info(self: AgentContext, game_state: GameState):
    """Analyse the state ``act`` was called with, memoising the most recent one.

    The cache exists because ``game_events_occurred``'s *new* state is the very
    state the next ``act`` will see, so analysing it twice would double the
    feature cost during training.

    The cache key must **not** be ``(round, step)`` alone: the engine stamps
    ``new_game_state`` with the step that just finished, so the old and the new
    state of one transition carry the *same* step number.  Keying on it silently
    returns the old analysis as the new one, which turns every TD update into
    ``Q(s,a) <- r + gamma * max_a Q(s,a)`` -- self-bootstrapping that still
    ranks actions by immediate reward (so a coin-collection stage looks fine)
    but can never learn to escape a bomb.  :func:`cache_next` therefore files
    the new state under ``step + 1``, which is the number ``act`` will see.
    """
    key = (game_state["round"], game_state["step"])
    if self._cache_key == key:
        return self._cache_info
    info = analyse(game_state)
    self._cache_key, self._cache_info = key, info
    return info


def cache_next(self: AgentContext, game_state: GameState):
    """Analyse a transition's *new* state and file it for the next ``act``."""
    info = analyse(game_state)
    self._cache_key = (game_state["round"], game_state["step"] + 1)
    self._cache_info = info
    return info


def entry_of(self: AgentContext, info) -> tuple[int, int]:
    """``(table index, group element)`` for a state."""
    if self.fold:
        return self.spec.canonical_index(info)
    return self.spec.index(self.spec.tuple_of(info)), 0


def legal_in_frame(legal: np.ndarray, g: int) -> np.ndarray:
    """Permute a per-action mask into the frame produced by ``g``."""
    out = np.zeros(N_ACTIONS, dtype=bool)
    out[ACTION_MAP[g]] = legal
    return out


def survivable_mask(info, g: int) -> np.ndarray:
    """Actions after which a survival route can still be *proven* (S5a).

    This is a mask, not a policy: it usually leaves several actions open and
    says nothing about which of them is good.  Used to shield exploration
    during training, and optionally at inference -- but only if the ablation
    shows the unmasked policy is already near-safe, so the mask is insurance
    rather than the policy.
    """
    escape = danger.escape_taus(info.lethal, info.blocked, info.pos,
                                blocked_now=info.occupied)
    world = np.zeros(N_ACTIONS, dtype=bool)
    world[:5] = escape >= 0
    world[5] = info.bombs_left and info.bomb_safe_strict
    return legal_in_frame(world, g)


def greedy(self: AgentContext, q: np.ndarray, mask: np.ndarray) -> int:
    """Argmax over allowed actions, ties broken by our own generator."""
    values = np.where(mask, q, -np.inf)
    best = np.flatnonzero(values == values.max())
    return int(best[self.rng.integers(len(best))])


def act(self: AgentContext, game_state: GameState) -> Action:
    info = state_info(self, game_state)
    idx, g = entry_of(self, info)
    mask = legal_in_frame(info.legal, g)

    if self.train:
        from .train import behaviour_action

        a_frame = behaviour_action(self, info, idx, g, mask)
    else:
        if self.use_mask:
            safe = mask & survivable_mask(info, g)
            if safe.any():
                mask = safe
        a_frame = greedy(self, self.Q[idx], mask)
    return ACTIONS[int(ACTION_MAP[INVERSE[g], a_frame])]
