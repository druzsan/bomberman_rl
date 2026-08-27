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

**Why the leaf value carries a potential term.**  Training used potential-based
shaping, so the network's ``Q`` is the value under the *shaped* reward while the
rewards backed up along a path here are the engine's own.  Writing the shaped
return of a ``d``-step path out, the shaping telescopes:

    sum_t gamma^t (r_t + gamma*Phi(s_t+1) - Phi(s_t))
        = sum_t gamma^t r_t  +  gamma^d Phi(s_d)  -  Phi(s_0)

``Phi(s_0)`` is the same for every root action and drops out of an ``argmax``;
``gamma^d * Phi(leaf)`` does not, and it is added.  Paths that end in a death or
a finished round need no correction at all -- ``Phi`` of a terminal state is 0
by convention, which is exactly what makes the telescoping hold.

Leaving it out was the first implementation and it was wrong in a way worth
recording, because the size of the error is not obvious.  Measured over 65 real
positions, ``gamma^3 * Phi(leaf)`` varies by a **mean of 0.100 and up to 0.214
across the siblings of one position**, while the agent only consults the search
when the flat top-2 gap is below **0.05**.  The omitted term was on average
twice the whole margin it was competing with, so the search's disagreements were
substantially an artefact of it.  The correction costs 0.65 ms a leaf against
~3 ms for the network evaluation of the same leaf.

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

_DELTA = ("UP", "DOWN", "LEFT", "RIGHT")

#: The events this search can price, at ``lib.rewards.DEFAULT_EVENT_REWARDS``'s
#: values and before ``reward_scale``.  Literals for the same reason
#: ``lib.sim`` keeps the rule constants: ``lib`` is vendored per agent.
COIN_COLLECTED = 3.0
KILLED_OPPONENT = 15.0
KILLED_SELF = -20.0
GOT_KILLED = -15.0
SURVIVED_ROUND = 5.0
CRATE_DESTROYED = 0.4
MOVED = -0.02
WAITED = -0.05
ESCAPED_DANGER = 1.0
MOVED_INTO_DANGER = -1.0
STAYED_IN_DANGER = -0.05

#: Events the trained reward contains that this search does **not** price, with
#: why.  They are a known residual between the backed-up return and the return
#: the network's ``Q`` actually estimates, and the size of that residual is the
#: main reason the search has to be measured rather than assumed to help.
#:
#: ``COIN_FOUND``            coins under crates are not observable at all.
#: ``GOOD_BOMB``,            the four bomb-quality events need
#: ``USELESS_BOMB``,         ``features.analyse`` on *both* states of every
#: ``SUICIDAL_BOMB``,        transition, at ~1.3 ms a node against ~3 ms for
#: ``BOMB_NEAR_OPPONENT``,   the network evaluation of a leaf -- and
#: ``TRAPPED_OPPONENT``      ``SUICIDAL_BOMB`` in particular is a *proxy* for a
#:                           death that this search observes directly.
#: ``LOOP``,                 need a visit history the search does not carry
#: ``WALKED_INTO_DEAD_END``  across the tree.
#: ``INVALID_ACTION``        never generated: `legal_actions` filters them.
UNPRICED = ("COIN_FOUND", "GOOD_BOMB", "USELESS_BOMB", "SUICIDAL_BOMB",
            "BOMB_NEAR_OPPONENT", "TRAPPED_OPPONENT", "LOOP",
            "WALKED_INTO_DEAD_END")


@dataclass(slots=True)
class SearchConfig:
    depth: int = 3
    gamma: float = 0.95
    reward_scale: float = 0.1
    opponents: str = "frozen"
    #: Add ``gamma^depth * Phi(leaf)`` so the backup is consistent with the
    #: shaped reward the network was trained on.  Off is the (wrong) arithmetic
    #: this module shipped with first, kept only so the ablation can be run.
    use_potential: bool = True
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


def step_reward(before: Sim, after: Sim, me: int, action: str,
                cfg: SearchConfig) -> float:
    """What this step was worth, priced as the training reward priced it.

    Everything here is read off the two positions or off the action, which is
    what keeps it cheap; :data:`UNPRICED` lists what is left out and why.  The
    danger transitions are the one term that costs anything, and it is 0.005 ms
    a state.
    """
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
    # Crates we destroyed.  Only our own bombs detonate into a crate on a step
    # where nobody else's does, in the overwhelming majority of positions; the
    # search attributes the whole arena delta to us, which is the same
    # simplification the frozen opponent model already makes.
    r += CRATE_DESTROYED * int(np.count_nonzero(before.arena == 1)
                               - np.count_nonzero(after.arena == 1))
    r += WAITED if action == "WAIT" else (MOVED if action in _DELTA else 0.0)
    if not after.agents[me].dead:
        was = _in_danger(before, me)
        now = _in_danger(after, me)
        if was and not now:
            r += ESCAPED_DANGER
        elif now and not was:
            r += MOVED_INTO_DANGER
        elif was and now:
            r += STAYED_IN_DANGER
    return r * cfg.reward_scale


def _in_danger(sim: Sim, me: int) -> bool:
    """Is our tile lethal within the danger horizon, as `lib.danger` defines it?"""
    from . import danger

    a = sim.agents[me]
    lethal = danger.lethal_bits(sim.arena, [((b.x, b.y), b.timer) for b in sim.bombs],
                                sim.danger_map().astype(float))
    return bool(lethal[a.x, a.y])


def leaf_potentials(states: list[dict], cfg: SearchConfig) -> np.ndarray:
    """``Phi(s)`` for each leaf, in the network's units.

    The weights come from :class:`lib.rewards.RewardConfig`'s defaults, which is
    what the shipped run trained with; only ``gamma`` and ``reward_scale``,
    which the search already knows, are overridden.  Imported here rather than
    at module scope so a caller that sets ``use_potential=False`` pays nothing.
    """
    from . import features, rewards

    rcfg = rewards.RewardConfig(gamma=cfg.gamma, reward_scale=cfg.reward_scale)
    return np.array([rewards.potential(features.analyse(gs), rcfg) * cfg.reward_scale
                     for gs in states])


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
                gained = acc + cfg.gamma ** t * step_reward(node, child, me,
                                                            action, cfg)
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
        phi = leaf_potentials(states, cfg) if cfg.use_potential else np.zeros(len(states))
        discount = cfg.gamma ** reached
        for (node, root, acc), row, p in zip(frontier, q, phi):
            allowed = set(legal_actions(node, me))
            legal = np.array([a in allowed for a in ACTIONS])
            leaf_v = float(row[legal].max()) if legal.any() else float(row.max())
            offer(root, acc + discount * (leaf_v + p))

    for root, value in settled.items():
        values[root] = value
    return SearchResult(values, len(frontier), nodes, truncated)
