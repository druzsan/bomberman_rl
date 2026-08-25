"""n-step tabular Q-learning with D4 folding.

Update, for the canonicalised state/action pair::

    G = sum_{k<n} gamma^k r_{t+k} + gamma^n * bootstrap(s_{t+n})
    Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])

with ``bootstrap`` either ``max_a Q`` (off-policy Q-learning) or the
epsilon-greedy expectation (Expected SARSA, on-policy), selected by the driver
so the comparison is a config diff rather than a second implementation.  The
target ``max``/expectation is taken over *legal* actions only, which removes a
large amount of bootstrap noise.

Because the table is folded, one update already covers all eight symmetry images
of the situation -- no explicit augmentation, and no risk of the eight copies
drifting apart.

Two engine subtleties are handled explicitly:

* the last transition of a *surviving* round is delivered twice (once by
  ``game_events_occurred`` and once by ``end_of_round`` with ``SURVIVED_ROUND``
  appended), so transitions are de-duplicated on ``(round, step)`` and the extra
  events are folded into the reward of the already-stored transition;
* a *dying* agent never receives ``game_events_occurred`` for its death step --
  that transition arrives only through ``end_of_round``, and is terminal.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .callbacks import cache_next, entry_of, greedy, legal_in_frame, state_info
from .callbacks import survivable_mask as _survivable_mask
from .lib.board import ACTION_INDEX
from .lib.rewards import (
    RewardConfig,
    custom_events,
    reward_from_events,
    transition_reward,
    true_reward,
)
from .lib.symmetry import ACTION_MAP
from .lib.types import Action, AgentContext, GameState

#: Shared control vector layout. The driver writes the schedules, every worker
#: reads them, so a schedule change takes effect without restarting anything.
CTRL_SIZE = 64


class CTRL:
    EPSILON = 0
    SHAPING = 1
    ALPHA = 2           # 0 -> Robbins-Monro 1/(1+N)**ALPHA_POWER
    ALPHA_POWER = 3
    GAMMA = 4
    N_STEP = 5
    LEARN = 6
    STOP = 7
    STAGE = 8
    ALGO = 9            # 0 Q-learning, 1 Expected SARSA
    ALLOW_BOMB = 10
    MASK_LETHAL = 11    # 0 off, 1 exploration only, 2 exploration and greedy
    WORKER_STEPS = 32   # slots 32.. hold one env-step counter per worker


ALGO_Q_LEARNING = 0
ALGO_EXPECTED_SARSA = 1

BOMB_ACTION = 5  # D4 fixes WAIT and BOMB, so this index is frame-independent

#: Set by ``training.worker`` to a queue; ``None`` outside the driver.
EPISODE_SINK = None
#: Replaced by the driver with the run's reward configuration.
REWARD_CONFIG = RewardConfig()


@dataclass(slots=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    next_mask: np.ndarray | None
    terminal: bool


def setup_training(self: AgentContext) -> None:
    self.buffer: deque[Transition] = deque()
    self.processed: set[tuple[int, int]] = set()
    self.last_key: tuple[int, int] | None = None
    self.last_events: list[str] = []
    self.recent_positions: deque = deque(maxlen=REWARD_CONFIG.loop_window)
    self.visit_counts: Counter = Counter()
    self.episode = _new_episode()
    self.worker_id = getattr(getattr(self, "tables", None), "worker_id", 0)
    if getattr(self, "ctrl", None) is None:
        self.ctrl = _default_ctrl()


def _default_ctrl() -> np.ndarray:
    ctrl = np.zeros(CTRL_SIZE, dtype=np.float64)
    ctrl[CTRL.EPSILON] = 0.1
    ctrl[CTRL.SHAPING] = 1.0
    ctrl[CTRL.ALPHA] = 0.0
    ctrl[CTRL.ALPHA_POWER] = 0.6
    ctrl[CTRL.GAMMA] = 0.95
    ctrl[CTRL.N_STEP] = 3
    ctrl[CTRL.LEARN] = 1.0
    ctrl[CTRL.ALLOW_BOMB] = 1.0
    return ctrl


def _new_episode() -> dict:
    return {"steps": 0, "shaped_return": 0.0, "true_return": 0.0,
            "events": Counter(), "updates": 0}


def behaviour_action(self: AgentContext, info, idx: int, g: int,
                     mask: np.ndarray) -> int:
    """Epsilon-greedy over legal actions, in the canonical frame.

    ``MASK_LETHAL`` controls *safe exploration*.  Escaping a bomb takes four
    consecutive correct moves, so with epsilon = 0.05 a 400-step episode takes a
    random action during an escape almost surely: measured training episodes
    lasted 33 steps and suicided in ~100 % of rounds while the same checkpoint
    evaluated greedily survived all 400 steps.  That mismatch starves the
    interesting part of the state space.  Level 1 removes provably-lethal
    actions from the *exploration* distribution only, leaving the greedy policy
    -- the thing that is actually submitted -- entirely learned; level 2 also
    masks the greedy choice, which is the S5a mask and must be ablated before
    it is shipped.
    """
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
    return greedy(self, self.Q[idx], allowed)


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
            # Surviving round: the same transition was already delivered by
            # game_events_occurred; only the extra events are new.
            extra = _multiset_difference(events, self.last_events)
            self.buffer[-1].reward += reward_from_events(extra, REWARD_CONFIG) \
                * REWARD_CONFIG.reward_scale
            self.buffer[-1].terminal = True
            self.episode["shaped_return"] += reward_from_events(extra, REWARD_CONFIG)
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

    idx, g = entry_of(self, old_info)
    a_frame = int(ACTION_MAP[g, ACTION_INDEX[action]])
    if new_info is None:
        next_idx, next_mask = 0, None
    else:
        next_idx, next_g = entry_of(self, new_info)
        next_mask = legal_in_frame(new_info.legal, next_g)
    self.buffer.append(Transition(idx, a_frame, reward, next_idx, next_mask, terminal))

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
    q = self.Q[t.next_state]
    values = q[t.next_mask]
    if int(self.ctrl[CTRL.ALGO]) == ALGO_EXPECTED_SARSA:
        eps = float(self.ctrl[CTRL.EPSILON])
        k = values.size
        best = float(values.max())
        return (1.0 - eps) * best + eps * float(values.mean()) if k else 0.0
    return float(values.max())


def _drain(self: AgentContext, force: bool = False) -> None:
    if self.ctrl[CTRL.LEARN] < 0.5:
        if force:
            self.buffer.clear()
        return
    gamma = float(self.ctrl[CTRL.GAMMA])
    n = max(1, int(self.ctrl[CTRL.N_STEP]))
    buf = self.buffer
    # One transition is always held back so ``end_of_round`` can still fold
    # the SURVIVED_ROUND reward into it; the n-step window is complete either
    # way because buf[0] only needs buf[0..n-1].
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
        _apply(self, buf[0].state, buf[0].action, g_return)
        buf.popleft()


def _apply(self: AgentContext, state: int, action: int, target: float) -> None:
    count = float(self.N[state, action])
    alpha = float(self.ctrl[CTRL.ALPHA])
    if alpha <= 0.0:
        alpha = 1.0 / (1.0 + count) ** float(self.ctrl[CTRL.ALPHA_POWER])
    self.Q[state, action] += alpha * (target - self.Q[state, action])
    self.N[state, action] = count + 1
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
