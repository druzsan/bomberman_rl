"""S5b -- shallow forward search with a learned leaf value.

The shipped agent spends 16.8 ms of a 500 ms think-time budget and then takes
an ``argmax`` over one network evaluation of the current position.  The
dynamics, though, are fully known and deterministic given every agent's action
(:mod:`lib.sim`), so the remaining 483 ms can be spent asking the *same* network
about positions three steps away instead of only about this one.

What is learned and what is search, stated plainly because the report has to be
explicit about it: **the evaluation function is the trained Q-network and
nothing else.**  There is no hand-written notion of a good position anywhere in
this module.  The search contributes the tree and the engine's own reward
constants; every value that decides an action comes out of the network.

**The backup.**  A path's value is

    sum_t gamma^t * r_t  +  gamma^depth * V(leaf),      V(s) = max_{legal a} Q(s, a)

with ``r_t`` the events the engine actually pays for, priced exactly as
:mod:`lib.rewards` priced them during training (a coin 0.3, a kill 1.5, our own
death -2.0 for a suicide and -1.5 otherwise, all after ``reward_scale``).  Our
own actions maximise; the opponents follow :data:`OPPONENT_POLICIES`.

**The one known inconsistency**, stated because it is the reason this has to be
measured rather than assumed to help.  Training used potential-based shaping, so
the network's ``Q`` is ``Q_true - Phi(s)``; the rewards backed up here are the
unshaped ones.  The mismatch is a ``gamma^depth * Phi(leaf)`` term that does not
cancel between siblings.  Recomputing ``Phi`` needs :func:`lib.features.analyse`
at 0.53 ms a node, which the budget cannot take at a few hundred nodes, so the
choice is to leave it out and let the held-out block say whether the search is
worth its latency.  Every leaf sits at the same depth, which is what keeps the
comparison between root actions honest even so.

**Opponents.**  The default model freezes them.  It is not a claim that they
stand still; it is that at depth three the things they can do that matter are
already on the board -- a bomb they drop this step detonates two steps past the
horizon -- while their *movement* only blocks tiles and contests kills.  It also
makes the step's random permutation irrelevant, since ``WAIT`` never contests a
tile, so the search does not have to guess an ordering it cannot know.  A
stronger model is a one-line substitution (:data:`OPPONENT_POLICIES`), and the
plan's other two candidates -- ``bfs_expert``'s action and a learned opponent
head -- fit the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import ACTIONS
from .sim import Sim

#: The engine events a depth-3 search can actually observe, priced as
#: ``lib.rewards.DEFAULT_EVENT_REWARDS`` priced them, before ``reward_scale``.
#: Kept as literals for the same reason ``lib.sim`` keeps the rule constants:
#: ``lib`` is vendored per agent and must not reach out of itself.
COIN_COLLECTED = 3.0
KILLED_OPPONENT = 15.0
KILLED_SELF = -20.0
GOT_KILLED = -15.0
SURVIVED_ROUND = 5.0


@dataclass(slots=True)
class SearchConfig:
    depth: int = 3
    gamma: float = 0.95
    reward_scale: float = 0.1
    opponents: str = "frozen"
    #: Hard cap on network evaluations per step.  Expansion stops one level
    #: short rather than exceed it, which bounds latency **without consulting a
    #: clock**: a wall-clock cut-off would make the policy depend on machine
    #: load, and every comparison in this project is a paired-seed one that
    #: assumes the same input gives the same action.  Measured cost is ~3 ms per
    #: state on one busy thread, so 48 is roughly 150 ms of a 500 ms budget.
    #: One ply is always searched regardless -- it is at most six leaves, it is
    #: what makes the difference between a search and an ``argmax``, and a cap
    #: that could return zero-depth results would silently degrade to the flat
    #: policy on exactly the crowded positions where search is worth most.
    max_leaves: int = 48


@dataclass(slots=True)
class SearchResult:
    values: np.ndarray          #: backed-up value per action, ``-inf`` if illegal
    leaves: int                 #: leaves handed to the network
    nodes: int                  #: positions expanded
    truncated: bool = False


def frozen_opponents(sim: Sim, me: int) -> dict[int, str]:
    """Every opponent waits.  See the module docstring for why this is the default."""
    return {i: "WAIT" for i in sim.active if i != me}


#: Opponent models, all ``(sim, me) -> {agent_index: action}``.
OPPONENT_POLICIES = {"frozen": frozen_opponents}


def legal_actions(sim: Sim, me: int) -> list[str]:
    """Actions that do something.

    An illegal move is a no-op in the engine (it records ``INVALID_ACTION`` and
    leaves the agent where it was), so including one would only add a duplicate
    of the ``WAIT`` subtree at six times the cost.
    """
    a = sim.agents[me]
    out = [d for d, (dx, dy) in (("UP", (0, -1)), ("DOWN", (0, 1)),
                                 ("LEFT", (-1, 0)), ("RIGHT", (1, 0)))
           if sim.tile_is_free(a.x + dx, a.y + dy)]
    out.append("WAIT")
    if a.bombs_left:
        out.append("BOMB")
    return out


def step_reward(before: Sim, after: Sim, me: int, cfg: SearchConfig) -> float:
    """What the engine paid us for this step, in the network's units."""
    r = 0.0
    coins = after.agents[me].score - before.agents[me].score
    kills = sum(1 for victim, killer in after.killed
                if killer == me and victim != me)
    # `score` moved by 1 per coin and 5 per kill; separate them before pricing.
    r += (coins - 5 * kills) * COIN_COLLECTED
    r += kills * KILLED_OPPONENT
    for victim, killer in after.killed:
        if victim == me:
            r += KILLED_SELF if killer == me else GOT_KILLED
    return r * cfg.reward_scale


def search(sim: Sim, me: int, evaluate, cfg: SearchConfig | None = None,
           ) -> SearchResult:
    """Expand our own actions to ``cfg.depth`` and back up ``max``.

    ``evaluate`` takes a list of ``game_state`` dicts and returns a
    ``(n, N_ACTIONS)`` array of Q values -- one batched forward pass, which is
    the whole reason the tree is built breadth-first instead of recursively.
    """
    cfg = cfg or SearchConfig()
    opponent_policy = OPPONENT_POLICIES[cfg.opponents]

    root_actions = legal_actions(sim, me)
    values = np.full(len(ACTIONS), -np.inf)
    if not root_actions:
        return SearchResult(values, 0, 0)

    # (position, index of the root action that led here, accumulated reward)
    frontier: list[tuple[Sim, int, float]] = []
    settled: dict[int, float] = {}      # root action -> best value found so far
    nodes = 0
    truncated = False

    def offer(root: int, value: float) -> None:
        if value > settled.get(root, -np.inf):
            settled[root] = value

    for action in root_actions:
        frontier.append((sim, ACTIONS.index(action), 0.0))

    reached = 0
    for t in range(cfg.depth):
        children: list[tuple[Sim, int, float]] = []
        seen: set[tuple] = set()
        for node, root, acc in frontier:
            # At the root each entry carries its own action; deeper, every legal
            # action is tried from every surviving node.
            todo = [ACTIONS[root]] if t == 0 else legal_actions(node, me)
            others = opponent_policy(node, me)
            for action in todo:
                child = node.copy()
                child.step({me: action, **others})
                nodes += 1
                gained = acc + cfg.gamma ** t * step_reward(node, child, me, cfg)
                if child.agents[me].dead:
                    offer(root, gained)
                    continue
                if child.round_over():
                    offer(root, gained + cfg.gamma ** t
                          * SURVIVED_ROUND * cfg.reward_scale)
                    continue
                key = (root, child.state())
                if key in seen:
                    continue
                seen.add(key)
                children.append((child, root, gained))
        if t > 0 and len(children) > cfg.max_leaves:
            # Keep the level we already have rather than pay for this one.  The
            # deeper level is *discarded*, not sampled: half an expanded level
            # would compare root actions at different depths, which is exactly
            # the bias the uniform-depth backup exists to avoid.  Terminal
            # outcomes found while expanding it are already recorded and stay.
            truncated = True
            break
        frontier = children
        reached = t + 1
        if not frontier:
            break

    if frontier:
        states = [node.to_game_state(me) for node, _, _ in frontier]
        q = np.asarray(evaluate(states))
        discount = cfg.gamma ** reached
        for (node, root, acc), row in zip(frontier, q):
            allowed = set(legal_actions(node, me))
            legal = np.array([a in allowed for a in ACTIONS])
            leaf_v = float(row[legal].max()) if legal.any() else float(row.max())
            offer(root, acc + discount * leaf_v)

    for root, value in settled.items():
        values[root] = value
    return SearchResult(values, len(frontier), nodes, truncated)
