"""A strong, deterministic rule-based reference agent.

**Not submittable** -- it contains no machine learning.  Its three jobs are:

1. an honest upper reference for the report (``rule_based_agent`` kills itself
   in about half of all rounds, which makes it a misleading ceiling);
2. the teacher for behaviour-cloning pretraining of the deep model;
3. a much stronger sparring partner than ``rule_based_agent``.

The policy is a strict priority list over the *proven-survivable* actions:
``lib.danger.escape_taus`` returns, per first action, the shortest time to a
tile that is safe from then on, so an action with ``-1`` is provably lethal and
is never played.  Everything else is BFS: walk to the nearest coin, else to the
nearest crate, else to the nearest opponent, and bomb only when the bomb
actually pays and an escape route exists.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

from .lib import danger
from .lib.board import ACTIONS, DELTAS
from .lib.features import DIR_HERE, DIR_NONE, analyse
from .lib.rewards import traps_opponent
from .lib.types import Action, AgentContext, GameState

#: Tie-breaking and loop-breaking use the agent's own generator so the global
#: numpy RNG (which the reference opponents reseed in their ``setup``) cannot
#: perturb us, and we cannot perturb them.
SEED = 20260824

LOOP_WINDOW = 8
LOOP_THRESHOLD = 3

#: Which bomb-safety proof to demand before dropping.  Selected by experiment
#: (see ``dev/experiments/002-bfs-expert-safety.md``); the environment variable
#: exists only so variants can be A/B-tested without editing code, and defaults
#: to the winner.  This agent is never submitted, so the knob costs nothing.
SAFETY = os.environ.get("BFS_EXPERT_SAFETY", "slack")
#: How opponents are modelled while planning an escape.  ``opt`` lets them move
#: out of the way (they only block for the current step), ``static`` assumes
#: they stand still, ``threat`` additionally blocks every tile they could reach
#: in time.  Measurement: with ``opt`` the expert routes its escape *through* a
#: stationary opponent and dies -- the bombing check was pessimistic while the
#: escape was optimistic, and the inconsistency is what killed it.
ESCAPE_MODEL = os.environ.get("BFS_EXPERT_ESCAPE", "threat")


def _escape_plan(info):
    """Escape times per first action, pessimistic first, optimistic as fallback.

    Falling back matters: if no escape survives the pessimistic model we are
    probably dead anyway, and an optimistic route is strictly better than
    giving up.
    """
    optimistic = danger.escape_taus(info.lethal, info.blocked, info.pos,
                                    blocked_now=info.occupied)
    if ESCAPE_MODEL == "opt" or not info.occupied.any():
        return optimistic
    threat = info.threat if ESCAPE_MODEL == "threat" else None
    strict = danger.escape_taus(info.lethal, info.blocked | info.occupied, info.pos,
                                threat=threat)
    return strict if (strict >= 0).any() else optimistic


def _may_bomb(info) -> bool:
    if SAFETY == "optimistic":
        return info.bomb_safe
    if SAFETY == "strict":
        return info.bomb_safe_strict
    if SAFETY == "slack":
        return info.bomb_safe_slack
    if SAFETY == "both":
        return info.bomb_safe_slack and info.bomb_safe_strict
    raise ValueError(f"unknown BFS_EXPERT_SAFETY={SAFETY!r}")


def setup(self: AgentContext) -> None:
    self.rng = np.random.default_rng(SEED)
    self.current_round = -1
    self.history = deque(maxlen=LOOP_WINDOW)


def _reset_round(self: AgentContext, game_state: GameState) -> None:
    self.current_round = game_state["round"]
    self.history = deque(maxlen=LOOP_WINDOW)


def act(self: AgentContext, game_state: GameState) -> Action:
    if game_state["round"] != self.current_round:
        _reset_round(self, game_state)

    info = analyse(game_state, with_bomb_slack=True)
    sx, sy = info.pos
    self.history.append((sx, sy))

    # Proven-survivable first actions: index 0-3 are the moves, 4 is WAIT.
    escape = _escape_plan(info)
    survivable = escape >= 0
    if not survivable.any():
        # Doomed within the horizon; still prefer a legal move over an invalid one.
        self.logger.debug("no survivable action, improvising")
        legal = [k for k in range(5) if info.legal[k]]
        return ACTIONS[self.rng.choice(legal) if legal else 4]

    # 1. In danger: run, shortest escape first.
    if not info.safe_here:
        best_tau = int(escape[survivable].min())
        candidates = [k for k in range(5) if survivable[k] and escape[k] == best_tau]
        return ACTIONS[_pick(self, candidates, info)]

    # 2. Bomb, but only when it pays and we can still get out.
    if info.bombs_left and _may_bomb(info):
        kills = info.bomb_hits > 0 and (info.opp_dist <= 1 or traps_opponent(info, game_state))
        if kills or info.bomb_crates >= 1:
            return "BOMB"

    # 3. Walk towards the objective.
    looping = self.history.count((sx, sy)) >= LOOP_THRESHOLD
    if not looping and info.target_dir < 4 and survivable[info.target_dir]:
        return ACTIONS[info.target_dir]

    # 4. Otherwise move somewhere safe; break loops by preferring new tiles.
    moves = [k for k in range(4) if survivable[k]]
    if moves:
        unseen = [k for k in moves
                  if (sx + DELTAS[k][0], sy + DELTAS[k][1]) not in self.history]
        pool = unseen or moves
        return ACTIONS[_pick(self, pool, info)]
    return "WAIT" if survivable[4] else ACTIONS[_pick(self, list(range(4)), info)]


def _pick(self: AgentContext, candidates: list[int], info) -> int:
    """Choose among equally rated actions, preferring progress, then at random.

    The randomness comes from the agent's own generator, so runs are
    reproducible from :data:`SEED` while symmetric stalemates still break.
    """
    if not candidates:
        return 4
    if info.target_dir not in (DIR_HERE, DIR_NONE) and info.target_dir in candidates:
        return info.target_dir
    return int(candidates[self.rng.integers(len(candidates))])
