"""n-step Expected SARSA with linear function approximation.

Semi-gradient update, for the canonicalised feature vector::

    G = sum_{k<n} gamma^k r_{t+k} + gamma^n * E_pi[Q(s_{t+n}, .)]
    w_a += alpha * (G - w_a . phi(s_t)) * phi(s_t)

**On-policy on purpose.** Off-policy bootstrapping plus function approximation
plus bootstrapping is the textbook "deadly triad" divergence case; with a shared
weight vector a single overestimated action can drag every state with it, which
a table cannot do. Expected SARSA keeps the target on-policy while still
averaging over the whole action distribution rather than one sampled action, so
it is much less noisy than plain SARSA.

The weight matrix lives in the same ``(rows, 6)`` shared array the tabular agent
uses, with rows indexing features instead of states, so the whole driver,
checkpointing and evaluation pipeline is unchanged.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .callbacks import cache_next, features_of, greedy, legal_in_frame, q_values, state_info
from .callbacks import survivable_mask as _survivable_mask
from .lib.board import ACTION_INDEX
from .lib.linear import feature_dim
from .lib.rewards import (
    RewardConfig,
    custom_events,
    reward_from_events,
    transition_reward,
    true_reward,
)
from .lib.symmetry import ACTION_MAP
from .lib.types import Action, AgentContext, GameState

CTRL_SIZE = 64


class CTRL:
    EPSILON = 0
    SHAPING = 1
    ALPHA = 2
    ALPHA_POWER = 3
    GAMMA = 4
    N_STEP = 5
    LEARN = 6
    STOP = 7
    STAGE = 8
    ALGO = 9
    ALLOW_BOMB = 10
    MASK_LETHAL = 11
    WORKER_STEPS = 32


BOMB_ACTION = 5
EPISODE_SINK = None
REWARD_CONFIG = RewardConfig()

#: Gradient clipping. phi has ~17 active entries, so an unlucky TD error can
#: move every one of them at once; a table can only ever move one cell.
MAX_TD_ERROR = 10.0


def n_parameter_rows(spec) -> int:
    """Rows the driver must allocate: features, not states."""
    return feature_dim(spec)


@dataclass(slots=True)
class Transition:
    phi: np.ndarray
    action: int
    reward: float
    next_phi: np.ndarray | None
    next_mask: np.ndarray | None
    terminal: bool


def setup_training(self: AgentContext) -> None:
    self.buffer: deque[Transition] = deque()
    self.processed: set[tuple[int, int]] = set()
    self.last_key = None
    self.last_events: list[str] = []
    self.recent_positions: deque = deque(maxlen=REWARD_CONFIG.loop_window)
    self.visit_counts: Counter = Counter()
    self.episode = _new_episode()
    self.worker_id = getattr(getattr(self, "tables", None), "worker_id", 0)


def _new_episode() -> dict:
    return {"steps": 0, "shaped_return": 0.0, "true_return": 0.0,
            "events": Counter(), "updates": 0}


def behaviour_action(self: AgentContext, info, phi: np.ndarray, g: int,
                     mask: np.ndarray) -> int:
    ctrl = self.ctrl
    allowed = mask.copy()
    if ctrl[CTRL.ALLOW_BOMB] < 0.5:
        allowed[BOMB_ACTION] = False
    level = int(ctrl[CTRL.MASK_LETHAL])
    exploring = self.rng.random() < ctrl[CTRL.EPSILON]
    if level >= 1 and (exploring or level >= 2):
        safe = allowed & _survivable_mask(info, g)
        if safe.any():
            allowed = safe
    if not allowed.any():
        allowed = mask
    if exploring:
        choices = np.flatnonzero(allowed)
        return int(choices[self.rng.integers(len(choices))])
    return greedy(self, q_values(self, phi), allowed)


def game_events_occurred(self: AgentContext, old_game_state: GameState,
                         self_action: Action, new_game_state: GameState,
                         events: list[str]) -> None:
    if old_game_state is None or self_action is None:
        return
    key = (old_game_state["round"], old_game_state["step"])
    if key in self.processed:
        return
    self.processed.add(key)
    old_info = state_info(self, old_game_state)
    new_info = cache_next(self, new_game_state)
    _store(self, old_game_state, old_info, self_action, new_info, list(events),
           terminal=False, key=key)


def end_of_round(self: AgentContext, last_game_state: GameState | None,
                 last_action: Action | None, events: list[str]) -> None:
    if last_game_state is not None and last_action is not None:
        key = (last_game_state["round"], last_game_state["step"])
        if key in self.processed and key == self.last_key and self.buffer:
            extra = _multiset_difference(events, self.last_events)
            bonus = reward_from_events(extra, REWARD_CONFIG) * REWARD_CONFIG.reward_scale
            self.buffer[-1].reward += bonus
            self.buffer[-1].terminal = True
            self.episode["shaped_return"] += bonus
            self.episode["true_return"] += true_reward(extra)
            self.episode["events"].update(extra)
        elif key not in self.processed:
            self.processed.add(key)
            old_info = state_info(self, last_game_state)
            _store(self, last_game_state, old_info, last_action, None, list(events),
                   terminal=True, key=key)
    _drain(self, force=True)
    _report(self, last_game_state)
    _reset_round(self)


def _store(self: AgentContext, old_state, old_info, action: Action, new_info,
           events: list[str], *, terminal: bool, key) -> None:
    if new_info is not None:
        self.recent_positions.append(new_info.pos)
        self.visit_counts = Counter(self.recent_positions)
    extra = custom_events(old_info, new_info, events, REWARD_CONFIG,
                          old_state=old_state, visit_counts=self.visit_counts)
    all_events = events + extra
    reward = transition_reward(old_info, new_info, all_events, REWARD_CONFIG,
                               shaping_scale=self.ctrl[CTRL.SHAPING])

    phi, g = features_of(self, old_info)
    a_frame = int(ACTION_MAP[g, ACTION_INDEX[action]])
    if new_info is None:
        next_phi, next_mask = None, None
    else:
        next_phi, next_g = features_of(self, new_info)
        next_mask = legal_in_frame(new_info.legal, next_g)
    self.buffer.append(Transition(phi.copy(), a_frame, reward, next_phi, next_mask,
                                  terminal))

    self.last_key = key
    self.last_events = events
    self.episode["steps"] += 1
    self.episode["shaped_return"] += reward
    self.episode["true_return"] += true_reward(all_events)
    self.episode["events"].update(all_events)
    self.ctrl[CTRL.WORKER_STEPS + self.worker_id] += 1
    _drain(self)


def _bootstrap(self: AgentContext, t: Transition) -> float:
    if t.terminal or t.next_mask is None or not t.next_mask.any():
        return 0.0
    values = q_values(self, t.next_phi)[t.next_mask]
    eps = float(self.ctrl[CTRL.EPSILON])
    return float((1.0 - eps) * values.max() + eps * values.mean())


def _drain(self: AgentContext, force: bool = False) -> None:
    if self.ctrl[CTRL.LEARN] < 0.5:
        if force:
            self.buffer.clear()
        return
    gamma = float(self.ctrl[CTRL.GAMMA])
    n = max(1, int(self.ctrl[CTRL.N_STEP]))
    buf = self.buffer
    while buf and (len(buf) > n or force):
        m = min(n, len(buf))
        g_return = 0.0
        discount = 1.0
        for j in range(m):
            g_return += discount * buf[j].reward
            discount *= gamma
            if buf[j].terminal:
                m = j + 1
                break
        g_return += discount * _bootstrap(self, buf[m - 1])
        _apply(self, buf[0], g_return)
        buf.popleft()


def _apply(self: AgentContext, t: Transition, target: float) -> None:
    alpha = float(self.ctrl[CTRL.ALPHA])
    prediction = float(t.phi @ self.W[:, t.action])
    error = float(np.clip(target - prediction, -MAX_TD_ERROR, MAX_TD_ERROR))
    # Normalising by ||phi||^2 makes the step size independent of how many
    # features happen to be active, which is what keeps a constant alpha stable.
    scale = alpha / max(float(t.phi @ t.phi), 1.0)
    self.W[:, t.action] += (scale * error) * t.phi
    self.N[:, t.action] += 1
    self.episode["updates"] += 1


def _report(self: AgentContext, last_game_state) -> None:
    if EPISODE_SINK is None:
        return
    ep = self.episode
    events = ep["events"]
    EPISODE_SINK.put({
        "worker": self.worker_id,
        "steps": ep["steps"],
        "shaped_return": ep["shaped_return"],
        "true_return": ep["true_return"],
        "score": (last_game_state["self"][1] if last_game_state else 0),
        "suicide": events.get("KILLED_SELF", 0) > 0,
        "survived": events.get("SURVIVED_ROUND", 0) > 0,
        "updates": ep["updates"],
        "events": dict(events),
    })


def _reset_round(self: AgentContext) -> None:
    self.buffer.clear()
    self.processed.clear()
    self.recent_positions.clear()
    self.visit_counts.clear()
    self.last_key = None
    self.last_events = []
    self.episode = _new_episode()


def _multiset_difference(a: list[str], b: list[str]) -> list[str]:
    remaining = Counter(b)
    out = []
    for item in a:
        if remaining[item]:
            remaining[item] -= 1
        else:
            out.append(item)
    return out
