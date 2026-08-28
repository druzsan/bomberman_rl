"""Inference for the deep Q-network agent (S3).

The model is a dueling fully-convolutional network over the 19 board planes of
:mod:`lib.encode`; :mod:`lib.qnet` defines it and is vendored with the agent, so
the architecture is rebuilt from the checkpoint rather than duplicated here.

``act`` is deliberately thin: build the planes, one forward pass, mask illegal
actions, ``argmax``.  Everything the *training* run needs -- reward shaping,
custom events, the replay episode -- lives in ``train.py`` and is imported only
when ``self.train`` is set, so a shipped agent without ``train.py`` still runs.

Cost on one CPU thread (measured): ~0.03 ms to encode, ~2.5 ms for the forward
pass, ~0.03 ms for the danger model -- roughly 2.6 ms per step against the
tournament's 500 ms budget.  ``torch.set_num_threads(1)`` in :func:`setup` is
not optional: torch otherwise starts one worker per core and oversubscribes the
single thread the tournament gives us, which *increases* latency.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .lib import danger, encode
from .lib.board import ACTIONS, N_ACTIONS
from .lib.search import SearchConfig
from .lib.search import search as forward_search
from .lib.sim import Sim
from .lib.symmetry import ACTION_MAP, N_G, transform_plane
from .lib.types import Action, AgentContext, Coordinate, GameState

#: Default artifact name, resolved relative to the agent directory (the engine
#: chdirs there around every callback).
MODEL_FILE = "model.pt"

#: Set by ``training.dqn_actor`` to the live policy holder (a network shared
#: with the learner plus the run's control vector).  ``None`` in the tournament,
#: which is what keeps ``multiprocessing`` out of the shipped agent.
TRAINING_POLICY = None

#: Tie-breaking seed.  The reference opponents call ``np.random.seed()`` in
#: their own ``setup``; owning a generator keeps us reproducible and them
#: undisturbed.
SEED = 20260825


class StateView:
    """The cheap half of :func:`lib.features.analyse`.

    The deep agent needs no objective BFS -- the network reads the board
    directly -- so inference recomputes only the danger model and the legality
    of each action.  Measured: 0.03 ms here versus 0.53 ms for a full
    ``analyse``, on a 2.6 ms step budget.
    """

    __slots__ = (
        "blocked",
        "bombs",
        "bombs_left",
        "explosion_map",
        "field",
        "legal",
        "lethal",
        "occupied",
        "others",
        "pos",
    )

    def __init__(self, game_state: GameState):
        field = game_state["field"]
        _, _, bombs_left, (sx, sy) = game_state["self"]
        self.pos: Coordinate = (int(sx), int(sy))
        self.bombs_left = bool(bombs_left)
        self.field = field
        self.bombs = game_state["bombs"]
        self.explosion_map = game_state["explosion_map"]
        self.others = game_state["others"]
        self.lethal = danger.lethal_bits(field, self.bombs, self.explosion_map)
        self.blocked = danger.blocked_mask(field, self.bombs)
        self.occupied = danger.agent_mask(field, self.others)

        legal = np.zeros(N_ACTIONS, dtype=bool)
        for k, (dx, dy) in enumerate(danger.DELTAS):
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            legal[k] = not (self.blocked[nx, ny] or self.occupied[nx, ny])
        legal[4] = True
        legal[5] = self.bombs_left
        self.legal = legal

    def survivable(self) -> np.ndarray:
        """Actions after which a survival route can still be *proven* (S5a).

        A mask, not a policy: it usually leaves several actions open and says
        nothing about which of them is good.  Used to shield exploration during
        training, and optionally at inference.
        """
        escape = danger.escape_taus(self.lethal, self.blocked, self.pos,
                                    blocked_now=self.occupied)
        out = np.zeros(N_ACTIONS, dtype=bool)
        out[:5] = escape >= 0
        if self.bombs_left:
            threat = None
            if self.others:
                from .lib import pathfind

                threat = pathfind.distance_field(
                    (~self.blocked) | self.occupied, self.occupied,
                    max_dist=danger.MAX_TAU + 1)
            out[5] = danger.survives_bomb_here(
                self.field, self.bombs, self.explosion_map, self.others, self.pos,
                block_others=bool(self.others), threat=threat)
        return out


def setup(self: AgentContext) -> None:
    import torch

    # One thread, always: the tournament gives us a single core and torch's
    # default (one worker per core) makes a 2.5 ms forward pass slower, not
    # faster.  ``tools.check_submission`` asserts this call is present.
    torch.set_num_threads(1)

    self.rng = np.random.default_rng(SEED)
    self.torch = torch
    #: Where we last dropped a bomb, or ``None``; see `act`.
    self.my_bomb: Coordinate | None = None
    #: How often the search actually ran, for the latency write-up.
    self.searched_steps = 0
    self.searched_leaves = 0

    # ``self.train`` matters here, not just the injection: a self-play league
    # runs *frozen* copies of this same agent as opponents inside the actor
    # process, and without this test they would silently share the live network
    # instead of the checkpoint they were pointed at.
    if TRAINING_POLICY is not None and self.train:
        self.policy = TRAINING_POLICY
        self.net = TRAINING_POLICY.net
        self.ctrl = TRAINING_POLICY.ctrl
        # From the policy, not from the network: the behaviour-cloning collector
        # injects a policy with no network at all (the teacher chooses the
        # actions) and must still encode states the same way.
        self.plane_set = TRAINING_POLICY.plane_set
        self.nets = [self.net] if self.net is not None else []
        self.use_mask = False
        self.tta = False
        self.search = None
        self.search_gap = 0.0
    else:
        from .lib.qnet import load_all as load_qnets

        # The evaluation harness points one process at one checkpoint through
        # this variable; the graders never set it, so the default must work.
        path = Path(os.environ.get("BOMBERMAN_MODEL", MODEL_FILE))
        # One artifact can hold several independently trained networks; a
        # single-network file yields a list of one, so this path is the same
        # either way.
        self.nets, meta = load_qnets(path)
        self.net = self.nets[0]
        self.policy = None
        self.ctrl = None
        self.plane_set = self.net.cfg.plane_set
        self.use_mask = bool(meta.get("use_mask", False))
        self.tta = bool(meta.get("tta", False))
        self.search = search_config(meta)
        self.search_gap = float(meta.get("search_gap", 0.05))
        self.logger.info("loaded %s: %s x%d, %d parameters each, planes=%s, "
                         "mask=%s, tta=%s, search=%s",
                         path, type(self.net).__name__, len(self.nets),
                         self.net.n_parameters(), self.plane_set, self.use_mask,
                         self.tta, self.search)


def q_values(self: AgentContext, view: StateView, game_state: GameState) -> np.ndarray:
    """Action values for one state, averaged over every view we can afford.

    Two independent averages, and they compose because they reduce different
    variance:

    * **``tta``** evaluates the state in all eight images of the square's
      symmetry group and maps each frame's actions back to the world frame
      (S6 in ``dev/plan.md``).  The wall layout is D4-invariant, so every image
      is a legitimate view of the same position.
    * **the ensemble** averages over the members of the artifact -- separately
      trained networks answering the same question.

    Averaging Q values directly is only sound because the members share a
    reward function and a discount, so their value scales agree; ``q_mean``
    across the runs bundled here differs by under 2 %.  Members trained under
    different rewards would have to be combined by rank, not by value.

    Cost is ``members x (8 if tta else 1)`` evaluations of ~2 ms each, against a
    measured budget of ~160 per step.
    """
    planes = encode.planes(game_state, view.lethal, self.plane_set)
    with self.torch.inference_mode():
        if not self.tta:
            x = self.torch.from_numpy(planes).unsqueeze(0)
            if len(self.nets) == 1:
                return self.net(x)[0].numpy()
            return np.mean([net(x)[0].numpy() for net in self.nets], axis=0)
        images = self.torch.from_numpy(
            np.stack([transform_plane(planes, g) for g in range(N_G)]))
        frames = np.arange(N_G)[:, None]
        if len(self.nets) == 1:
            q = self.net(images).numpy()
            return q[frames, ACTION_MAP].mean(axis=0)
        return np.mean([net(images).numpy()[frames, ACTION_MAP].mean(axis=0)
                        for net in self.nets], axis=0)


def search_config(meta: dict) -> SearchConfig | None:
    """Read the S5b search settings out of the artifact's ``meta`` block.

    ``None`` -- the default -- is the flat ``argmax`` the model shipped with, so
    an artifact written before this existed behaves exactly as it always did.
    """
    if not meta.get("search"):
        return None
    return SearchConfig(depth=int(meta.get("search_depth", 3)),
                        gamma=float(meta.get("search_gamma", 0.95)),
                        max_leaves=int(meta.get("search_leaves", 48)))


def leaf_values(self: AgentContext, states: list[GameState]) -> np.ndarray:
    """One forward pass over every leaf the search produced.

    Test-time augmentation is deliberately *not* applied here.  It costs eight
    evaluations per state, a leaf costs ~3 ms, and the budget is ~160
    evaluations a step -- so TTA at the leaves and a depth-3 tree cannot both be
    afforded.  They are alternative ways to spend the same budget, which is why
    the two are ablated against each other rather than stacked.
    """
    planes = np.stack([
        encode.planes(gs, danger.lethal_bits(gs["field"], gs["bombs"],
                                             gs["explosion_map"]), self.plane_set)
        for gs in states])
    with self.torch.inference_mode():
        return self.net(self.torch.from_numpy(planes)).numpy()


def searched(self: AgentContext, q: np.ndarray, mask: np.ndarray,
             game_state: GameState) -> np.ndarray:
    """Action values from a depth-limited search, or ``q`` unchanged.

    The search runs only when the flat ``argmax`` is close to a tie.  Two
    reasons, and the second is the one that matters:

    * **Cost.**  A confident position is decided for ~3 ms; the budget is spent
      where it can change something.
    * **Risk.**  The search can then only overturn decisions the network
      already considered close, so a search that is no good is bounded in how
      much damage it can do -- and one that is good keeps every confident call
      the shipped model already gets right.

    ``search_gap`` is in the network's own units, where the whole value range
    at ``gamma = 0.95`` is about 0.4 (E12), so a gap of 0.05 is a real tie
    rather than a nominal one.
    """
    cfg = self.search
    values = np.where(mask, q, -np.inf)
    ordered = np.sort(values[np.isfinite(values)])
    if len(ordered) < 2 or ordered[-1] - ordered[-2] >= self.search_gap:
        return q
    sim = Sim.from_game_state(game_state, self.my_bomb)
    result = forward_search(sim, 0, lambda s: leaf_values(self, s), cfg)
    self.searched_steps += 1
    self.searched_leaves += result.leaves
    if not np.isfinite(result.values).any():
        return q
    return result.values


def greedy(self: AgentContext, q: np.ndarray, mask: np.ndarray) -> int:
    """Argmax over allowed actions, ties broken by our own generator."""
    values = np.where(mask, q, -np.inf)
    best = np.flatnonzero(values == values.max())
    return int(best[self.rng.integers(len(best))])


def act(self: AgentContext, game_state: GameState) -> Action:
    view = StateView(game_state)
    if self.train:
        from .train import behaviour_action

        return ACTIONS[behaviour_action(self, view, game_state)]

    q = q_values(self, view, game_state)
    mask = view.legal
    if self.use_mask:
        safe = mask & view.survivable()
        if safe.any():
            mask = safe
    if self.search is not None:
        # Our own bomb is the one piece of hidden state the search needs and the
        # observation does not carry (`Sim.from_game_state`).  It is forgotten
        # as soon as it is no longer on the board.
        if self.my_bomb not in [(int(x), int(y)) for (x, y), _ in view.bombs]:
            self.my_bomb = None
        q = searched(self, q, mask, game_state)
        # A searched value of -inf means "every line from here is lost", not
        # "illegal"; masking it away again would discard the search's answer.
        if np.isfinite(q[mask]).any():
            mask = mask & np.isfinite(q)
    action = ACTIONS[greedy(self, q, mask)]
    if action == "BOMB":
        self.my_bomb = view.pos
    return action
