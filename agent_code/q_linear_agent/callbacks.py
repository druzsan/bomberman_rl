"""Inference for the linear Q agent.

``Q(s, a) = w_a . phi(s)`` -- about 280 parameters against the tabular agent's
~600 table entries, on exactly the same state analysis. Two things this buys:
it generalises to states never seen in training, and the weights are directly
readable, which a table of 600 independent entries is not.

The weight matrix is stored transposed, as ``(feature, action)``, so the same
checkpoint and shared-memory machinery serves both agents.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .lib import danger
from .lib.board import ACTIONS, N_ACTIONS
from .lib.features import FeatureSpec, analyse
from .lib.linear import build, feature_dim
from .lib.symmetry import ACTION_MAP, INVERSE
from .lib.types import Action, AgentContext, GameState

MODEL_FILE = "model.npz"
TRAINING_TABLES = None
SEED = 20260824


def setup(self: AgentContext) -> None:
    self.rng = np.random.default_rng(SEED)
    if TRAINING_TABLES is not None:
        tables = TRAINING_TABLES
        self.tables = tables
        self.W = tables.q
        self.N = tables.n
        self.ctrl = tables.ctrl
        self.spec = FeatureSpec(tables.feature_set)
        self.fold = bool(tables.fold)
        self.use_mask = False
    else:
        path = Path(os.environ.get("BOMBERMAN_MODEL", MODEL_FILE))
        with np.load(path) as data:
            self.spec = FeatureSpec(str(data["feature_set"]))
            self.fold = bool(data["fold"])
            self.W = np.zeros((feature_dim(self.spec), N_ACTIONS), dtype=np.float32)
            self.W[data["states"]] = data["values"]
            self.N = None
            self.use_mask = bool(data["use_mask"]) if "use_mask" in data.files else False
        self.ctrl = None
        self.logger.info("loaded %s: %s, %d weights", path, self.spec.name, self.W.size)
    self.phi = np.zeros(feature_dim(self.spec), dtype=np.float32)
    self._cache_key = None
    self._cache_info = None


def state_info(self: AgentContext, game_state: GameState):
    """See ``q_tabular_agent.callbacks.state_info`` for why the key is not the step."""
    key = (game_state["round"], game_state["step"])
    if self._cache_key == key:
        return self._cache_info
    info = analyse(game_state)
    self._cache_key, self._cache_info = key, info
    return info


def cache_next(self: AgentContext, game_state: GameState):
    info = analyse(game_state)
    self._cache_key = (game_state["round"], game_state["step"] + 1)
    self._cache_info = info
    return info


def features_of(self: AgentContext, info, out=None) -> tuple[np.ndarray, int]:
    """``(phi, g)`` in the canonical frame."""
    values = self.spec.tuple_of(info)
    if self.fold:
        values, g = self.spec.canonical(values)
    else:
        g = 0
    return build(self.spec, values, info, out=out), g


def q_values(self: AgentContext, phi: np.ndarray) -> np.ndarray:
    return phi @ self.W


def legal_in_frame(legal: np.ndarray, g: int) -> np.ndarray:
    out = np.zeros(N_ACTIONS, dtype=bool)
    out[ACTION_MAP[g]] = legal
    return out


def survivable_mask(info, g: int) -> np.ndarray:
    escape = danger.escape_taus(info.lethal, info.blocked, info.pos,
                                blocked_now=info.occupied)
    world = np.zeros(N_ACTIONS, dtype=bool)
    world[:5] = escape >= 0
    world[5] = info.bombs_left and info.bomb_safe_strict
    return legal_in_frame(world, g)


def greedy(self: AgentContext, q: np.ndarray, mask: np.ndarray) -> int:
    values = np.where(mask, q, -np.inf)
    best = np.flatnonzero(values == values.max())
    return int(best[self.rng.integers(len(best))])


def act(self: AgentContext, game_state: GameState) -> Action:
    info = state_info(self, game_state)
    phi, g = features_of(self, info, out=self.phi)
    mask = legal_in_frame(info.legal, g)
    if self.train:
        from .train import behaviour_action

        a_frame = behaviour_action(self, info, phi, g, mask)
    else:
        if self.use_mask:
            safe = mask & survivable_mask(info, g)
            if safe.any():
                mask = safe
        a_frame = greedy(self, q_values(self, phi), mask)
    return ACTIONS[int(ACTION_MAP[INVERSE[g], a_frame])]
