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
        self.use_mask = False
        self.tta = False
    else:
        from .lib.qnet import load as load_qnet

        # The evaluation harness points one process at one checkpoint through
        # this variable; the graders never set it, so the default must work.
        path = Path(os.environ.get("BOMBERMAN_MODEL", MODEL_FILE))
        self.net, meta = load_qnet(path)
        self.policy = None
        self.ctrl = None
        self.plane_set = self.net.cfg.plane_set
        self.use_mask = bool(meta.get("use_mask", False))
        self.tta = bool(meta.get("tta", False))
        self.logger.info("loaded %s: %s, %d parameters, planes=%s, mask=%s, tta=%s",
                         path, type(self.net).__name__, self.net.n_parameters(),
                         self.plane_set, self.use_mask, self.tta)


def q_values(self: AgentContext, view: StateView, game_state: GameState) -> np.ndarray:
    """Action values for one state.

    With ``tta`` set, the state is evaluated in all eight images of the square's
    symmetry group and the values are averaged after mapping each frame's
    actions back to the world frame (S6 in ``dev/plan.md``).  The board's wall
    layout is D4-invariant, so every image is a legitimate view of the same
    position and the average is a free variance reduction -- eight images fit in
    one batched forward pass and still cost a fraction of the step budget.
    """
    planes = encode.planes(game_state, view.lethal, self.plane_set)
    with self.torch.inference_mode():
        if not self.tta:
            x = self.torch.from_numpy(planes).unsqueeze(0)
            return self.net(x)[0].numpy()
        images = np.stack([transform_plane(planes, g) for g in range(N_G)])
        q = self.net(self.torch.from_numpy(images)).numpy()
        return q[np.arange(N_G)[:, None], ACTION_MAP].mean(axis=0)


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
    return ACTIONS[greedy(self, q, mask)]
