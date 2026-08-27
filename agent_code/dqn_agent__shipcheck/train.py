"""Actor-side training callbacks for the deep Q-network agent.

The learning itself happens in a single GPU process (``training.dqn_learner``);
this module is the *actor* half of Ape-X.  Its whole job is to turn the engine's
callback stream into well-formed episodes:

* choose actions with an epsilon-greedy behaviour policy over the live network,
  optionally shielded by the proven-survivable mask (the safety curriculum that
  S1 could not do without -- escaping a bomb needs four consecutive correct
  moves, so an unshielded random policy dies before it ever sees why bombing
  pays);
* compute the shaped reward for every transition with the *same*
  :mod:`lib.rewards` configuration the tabular agent used, so the two models are
  comparable rather than merely both trained;
* bit-pack each state and ship the finished episode to the learner.

n-step returns, prioritisation and the bootstrap are the learner's business:
the actor sends single-step rewards plus episode boundaries, which means
``n_step`` can be changed without restarting the actors.

Two engine subtleties, identical to the tabular agent's:

* the last transition of a *surviving* round is delivered twice (once by
  ``game_events_occurred``, once by ``end_of_round`` with ``SURVIVED_ROUND``
  appended), so transitions are de-duplicated on ``(round, step)`` and the extra
  events are folded into the reward already stored;
* a *dying* agent never receives ``game_events_occurred`` for its death step --
  that transition arrives only through ``end_of_round``, and is terminal.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .callbacks import StateView, greedy, q_values
from .lib import encode
from .lib.board import ACTION_INDEX
from .lib.features import analyse
from .lib.rewards import (
    RewardConfig,
    custom_events,
    reward_from_events,
    transition_reward,
    true_reward,
)
from .lib.types import Action, AgentContext, GameState

CTRL_SIZE = 64


class CTRL:
    """Shared control vector: the driver writes, every actor reads."""

    EPSILON = 0
    SHAPING = 1
    STOP = 7
    STAGE = 8
    ALLOW_BOMB = 10
    MASK_LETHAL = 11    # 0 off, 1 exploration only, 2 exploration and greedy
    WEIGHT_VERSION = 12
    N_ACTORS = 13
    EPS_LADDER = 14
    LEAGUE_VERSION = 15
    WORKER_STEPS = 32   # slots 32.. hold one env-step counter per worker


BOMB_ACTION = 5

#: Set by ``training.dqn_actor``; ``None`` outside a run.
EPISODE_SINK = None
#: Set by ``training.bc`` to a callable ``(self, view, game_state) -> action index``
#: while recording the behaviour-cloning dataset.  Letting the *teacher* drive
#: this module rather than writing a second recorder is what guarantees the
#: expert episodes are byte-identical in format, reward and event handling to
#: the ones the actors produce -- so the same file can seed the DQfD replay.
TEACHER = None
#: Replaced by the driver with the run's reward configuration.
REWARD_CONFIG = RewardConfig()

#: How often an actor pulls fresh weights, in its own env steps.  A full episode
#: is up to 400 steps and the learner does a few hundred updates a second, so
#: refreshing only between rounds would leave the behaviour policy visibly
#: stale.
REFRESH_EVERY = 32


def setup_training(self: AgentContext) -> None:
    self.episode = _new_episode()
    self.processed: set[tuple[int, int]] = set()
    self.last_key: tuple[int, int] | None = None
    self.last_events: list[str] = []
    self.recent_positions: deque = deque(maxlen=REWARD_CONFIG.loop_window)
    self.visit_counts: Counter = Counter()
    self.worker_id = getattr(getattr(self, "policy", None), "worker_id", 0)
    self._info_key = None
    self._info = None
    self._since_refresh = 0
    if getattr(self, "ctrl", None) is None:
        self.ctrl = _default_ctrl()


def _default_ctrl() -> np.ndarray:
    ctrl = np.zeros(CTRL_SIZE, dtype=np.float64)
    ctrl[CTRL.EPSILON] = 0.1
    ctrl[CTRL.SHAPING] = 1.0
    ctrl[CTRL.ALLOW_BOMB] = 1.0
    return ctrl


def _new_episode() -> dict:
    return {"packed": [], "step": [], "action": [], "reward": [], "legal": [],
            "steps": 0, "shaped_return": 0.0, "true_return": 0.0, "events": Counter()}


def state_info(self: AgentContext, game_state: GameState):
    """Full :func:`lib.features.analyse`, memoised across one transition.

    The *new* state of one transition is the *old* state of the next, so caching
    halves the analysis cost.  The key must not be ``(round, step)`` alone: the
    engine stamps ``new_game_state`` with the step that just finished, so both
    states of a transition carry the same number.  New states are therefore
    filed under ``step + 1``.
    """
    key = (game_state["round"], game_state["step"])
    if self._info_key == key:
        return self._info
    info = analyse(game_state)
    self._info_key, self._info = key, info
    return info


def _cache_next(self: AgentContext, game_state: GameState):
    info = analyse(game_state)
    self._info_key = (game_state["round"], game_state["step"] + 1)
    self._info = info
    return info


def _epsilon(self: AgentContext, ctrl: np.ndarray) -> float:
    """This actor's exploration rate.

    With ``EPS_LADDER`` set, the actors spread over the Ape-X ladder
    ``eps_i = eps^(1 + 7*i/(N-1))``: actor 0 explores at the scheduled rate and
    the last actor is almost greedy.  One shared replay buffer then contains
    both the wide exploration that finds new behaviour and the near-greedy
    trajectories that show what the current policy actually does -- which
    matters here because the interesting states (mid-escape, next to an
    opponent) are only reachable by a policy that does not act randomly.
    """
    eps = float(ctrl[CTRL.EPSILON])
    n = int(ctrl[CTRL.N_ACTORS])
    if ctrl[CTRL.EPS_LADDER] < 0.5 or n <= 1 or eps <= 0.0:
        return eps
    return eps ** (1.0 + 7.0 * min(self.worker_id, n - 1) / (n - 1))


def behaviour_action(self: AgentContext, view: StateView, game_state: GameState) -> int:
    """Epsilon-greedy over legal actions, optionally shielded.

    ``MASK_LETHAL`` level 1 removes provably-lethal actions from the
    *exploration* distribution only, leaving the greedy policy -- the thing that
    is actually submitted -- entirely learned.  Level 2 also masks the greedy
    choice, which is the S5a mask and must be ablated before it is shipped.
    """
    if TEACHER is not None:
        return TEACHER(self, view, game_state)

    policy = getattr(self, "policy", None)
    if policy is not None:
        self._since_refresh += 1
        if self._since_refresh >= REFRESH_EVERY:
            self._since_refresh = 0
            policy.maybe_refresh()

    ctrl = self.ctrl
    allowed = view.legal.copy()
    if ctrl[CTRL.ALLOW_BOMB] < 0.5:
        allowed[BOMB_ACTION] = False
    level = int(ctrl[CTRL.MASK_LETHAL])
    exploring = self.rng.random() < _epsilon(self, ctrl)
    if level >= 1 and (exploring or level >= 2):
        safe = allowed & view.survivable()
        if safe.any():
            allowed = safe
    if not allowed.any():
        allowed = view.legal
    if exploring:
        choices = np.flatnonzero(allowed)
        return int(choices[self.rng.integers(len(choices))])
    return greedy(self, q_values(self, view, game_state), allowed)


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
    new_info = _cache_next(self, new_game_state)
    _store(self, old_game_state, old_info, self_action, new_info, list(events), key)


def end_of_round(self: AgentContext, last_game_state: GameState | None,
                 last_action: Action | None, events: list[str]) -> None:
    if last_game_state is not None and last_action is not None:
        key = (last_game_state["round"], last_game_state["step"])
        if key in self.processed and key == self.last_key and self.episode["reward"]:
            # Surviving round: the transition is already stored, only the extra
            # events (SURVIVED_ROUND) are new.
            extra = _multiset_difference(events, self.last_events)
            bonus = reward_from_events(extra, REWARD_CONFIG) * REWARD_CONFIG.reward_scale
            self.episode["reward"][-1] += bonus
            self.episode["shaped_return"] += bonus
            self.episode["true_return"] += true_reward(extra)
            self.episode["events"].update(extra)
        elif key not in self.processed:
            self.processed.add(key)
            old_info = state_info(self, last_game_state)
            _store(self, last_game_state, old_info, last_action, None, list(events), key)
    _flush(self, last_game_state)
    _reset_round(self)


def _store(self: AgentContext, old_state: GameState, old_info, action: Action,
           new_info, events: list[str], key) -> None:
    if new_info is not None:
        self.recent_positions.append(new_info.pos)
        self.visit_counts = Counter(self.recent_positions)
    extra = custom_events(old_info, new_info, events, REWARD_CONFIG,
                          old_state=old_state, visit_counts=self.visit_counts)
    all_events = events + extra
    reward = transition_reward(old_info, new_info, all_events, REWARD_CONFIG,
                               shaping_scale=self.ctrl[CTRL.SHAPING])

    ep = self.episode
    ep["packed"].append(encode.pack(old_state, old_info.lethal))
    ep["step"].append(min(int(old_state["step"]), encode.MAX_STEPS))
    ep["action"].append(ACTION_INDEX[action])
    ep["reward"].append(float(reward))
    ep["legal"].append(encode.bits_to_mask(old_info.legal))

    self.last_key = key
    self.last_events = events
    ep["steps"] += 1
    ep["shaped_return"] += reward
    ep["true_return"] += true_reward(all_events)
    ep["events"].update(all_events)
    self.ctrl[CTRL.WORKER_STEPS + self.worker_id] += 1


def _flush(self: AgentContext, last_game_state) -> None:
    """Ship the finished episode to the learner."""
    ep = self.episode
    if EPISODE_SINK is None or not ep["reward"]:
        return
    events = ep["events"]
    EPISODE_SINK.put({
        "packed": np.asarray(ep["packed"], dtype=np.uint8),
        "step": np.asarray(ep["step"], dtype=np.uint16),
        "action": np.asarray(ep["action"], dtype=np.uint8),
        "reward": np.asarray(ep["reward"], dtype=np.float32),
        "legal": np.asarray(ep["legal"], dtype=np.uint8),
        "worker": self.worker_id,
        "steps": ep["steps"],
        "shaped_return": ep["shaped_return"],
        "true_return": ep["true_return"],
        "score": (last_game_state["self"][1] if last_game_state else 0),
        "suicide": events.get("KILLED_SELF", 0) > 0,
        "survived": events.get("SURVIVED_ROUND", 0) > 0,
        "events": dict(events),
    })


def _reset_round(self: AgentContext) -> None:
    self.episode = _new_episode()
    self.processed.clear()
    self.recent_positions.clear()
    self.visit_counts.clear()
    self.last_key = None
    self.last_events = []
    self._info_key = None
    self._info = None


def _multiset_difference(a: list[str], b: list[str]) -> list[str]:
    remaining = Counter(b)
    out = []
    for item in a:
        if remaining[item]:
            remaining[item] -= 1
        else:
            out.append(item)
    return out
