"""A faithful re-implementation of one ``environment.GenericWorld.do_step``.

The dynamics of this game are **fully known and deterministic** given all four
agents' actions, and the shipped agent uses 16.8 ms of a 500 ms think-time
budget.  This module is what turns the other 483 ms into something usable: a
world model cheap enough to expand a few hundred futures per step and exact
enough that the values backed up through it are values of the real game.

"Exact" is a claim that has to be earned, not asserted, so the module is written
to be *checkable*: :meth:`Sim.from_world` reads a live ``BombeRLeWorld`` and
:meth:`Sim.state` returns a hashable digest, which lets ``tests/test_sim.py``
step a recorded game in both the engine and here and assert they never diverge.
Nothing above this module is worth trusting until that test passes.

Everything in ``do_step`` matters, so everything is reproduced, including the
parts that look like bugs:

* **Order.**  ``poll_and_run_agents`` -> ``collect_coins`` -> ``update_explosions``
  -> ``update_bombs`` -> ``evaluate_explosions``.  A blast created by
  ``update_bombs`` is therefore evaluated in the *same* step that created it.
* **Action order within a step.**  Actions are applied in a random permutation
  of ``active_agents``, and ``tile_is_free`` is evaluated as each one is
  applied -- so whether a move succeeds depends on who moved first, and a bomb
  dropped earlier in the permutation blocks a move later in it.
* **Blasts pass through crates.**  ``get_blast_coords`` stops at walls (``-1``)
  only, which the task sheet does not mention and which changes every safety
  calculation in the game.
* **An explosion is lethal for two steps.**  It is created with
  ``timer = EXPLOSION_TIMER`` at stage 0 and is dangerous only at stage 0;
  ``next_stage`` moves it to stage 1 (``len(Explosion.ASSETS[1]) == 2``, the
  smoke frames) and the stage after that raises ``IndexError`` and retires it.
  The owner's bomb is returned when it reaches stage 1, not when it detonates.
* **Two agents on one coin both score.**  ``collect_coins`` tests
  ``coin.collectable`` in the outer loop only.  Unreachable in the stock game --
  ``tile_is_free`` keeps agents off each other -- and reproduced anyway, because
  the point of this module is to have no opinions.

The one thing deliberately *not* reproduced is the engine's event bookkeeping:
trophies, per-agent event lists and statistics exist for the training callbacks
and cost more than the search can spare.  Scores, deaths and the board are
exact; :attr:`Sim.killed` reports who died this step and to whose bomb, which is
all a leaf evaluation needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import ACTIONS

#: Mirrors ``settings.py``.  Duplicated rather than imported because ``lib`` is
#: vendored into each agent directory and must never reach out to the repo root.
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2
MAX_STEPS = 400
REWARD_KILL = 5
REWARD_COIN = 1

#: ``len(Explosion.ASSETS[stage])`` for each stage, i.e. how many steps an
#: explosion spends in it.  Stage 0 is the lethal one; stage 1 is smoke.
EXPLOSION_STAGES = (4, 2)

WALL = -1
FREE = 0
CRATE = 1

_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


@dataclass(slots=True)
class SimAgent:
    x: int
    y: int
    bombs_left: bool = True
    dead: bool = False
    score: int = 0
    name: str = ""


@dataclass(slots=True)
class SimBomb:
    x: int
    y: int
    owner: int          # index into Sim.agents
    timer: int = BOMB_TIMER


@dataclass(slots=True)
class SimExplosion:
    coords: tuple[tuple[int, int], ...]
    owner: int
    timer: int = EXPLOSION_TIMER
    stage: int = 0

    @property
    def dangerous(self) -> bool:
        return self.stage == 0


def blast_coords(arena: np.ndarray, x: int, y: int,
                 power: int = BOMB_POWER) -> tuple[tuple[int, int], ...]:
    """``items.Bomb.get_blast_coords``: stops at walls, **passes through crates**.

    The order of the returned tiles is the engine's (centre, +x, -x, +y, -y),
    which nothing depends on but which keeps a divergence easy to read.
    """
    out = [(x, y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for i in range(1, power + 1):
            cx, cy = x + i * dx, y + i * dy
            if arena[cx, cy] == WALL:
                break
            out.append((cx, cy))
    return tuple(out)


@dataclass(slots=True)
class Sim:
    """One position, steppable with :meth:`step`.

    ``agents`` keeps the engine's original ordering and never shrinks; ``active``
    is the list of indices into it that ``poll_and_run_agents`` iterates, and it
    is what a permutation indexes into.
    """

    arena: np.ndarray
    agents: list[SimAgent]
    active: list[int]
    bombs: list[SimBomb] = field(default_factory=list)
    explosions: list[SimExplosion] = field(default_factory=list)
    coins: list[tuple[int, int, bool]] = field(default_factory=list)
    step_no: int = 0
    #: ``(victim, killer)`` index pairs from the most recent :meth:`step`;
    #: ``victim == killer`` is a suicide.
    killed: list[tuple[int, int]] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_world(cls, world) -> Sim:
        """Read a live ``BombeRLeWorld`` -- the ground truth the tests compare to."""
        index = {id(a): i for i, a in enumerate(world.agents)}
        agents = [SimAgent(a.x, a.y, bool(a.bombs_left), bool(a.dead), int(a.score),
                           a.name) for a in world.agents]
        sim = cls(
            arena=np.array(world.arena, dtype=np.int8),
            agents=agents,
            active=[index[id(a)] for a in world.active_agents],
            bombs=[SimBomb(b.x, b.y, index[id(b.owner)], int(b.timer))
                   for b in world.bombs],
            explosions=[SimExplosion(tuple(map(tuple, ex.blast_coords)),
                                     index[id(ex.owner)], int(ex.timer),
                                     0 if ex.stage is None else int(ex.stage))
                        for ex in world.explosions],
            coins=[(c.x, c.y, bool(c.collectable)) for c in world.coins],
            step_no=int(world.step),
        )
        return sim

    def copy(self) -> Sim:
        """A deep copy cheap enough to make one per search node."""
        return Sim(
            arena=self.arena.copy(),
            agents=[SimAgent(a.x, a.y, a.bombs_left, a.dead, a.score, a.name)
                    for a in self.agents],
            active=list(self.active),
            bombs=[SimBomb(b.x, b.y, b.owner, b.timer) for b in self.bombs],
            explosions=[SimExplosion(e.coords, e.owner, e.timer, e.stage)
                        for e in self.explosions],
            coins=list(self.coins),
            step_no=self.step_no,
        )

    # -- the step ---------------------------------------------------------
    def tile_is_free(self, x: int, y: int) -> bool:
        if self.arena[x, y] != FREE:
            return False
        for b in self.bombs:
            if b.x == x and b.y == y:
                return False
        for i in self.active:
            a = self.agents[i]
            if a.x == x and a.y == y:
                return False
        return True

    def perform_action(self, i: int, action: str) -> None:
        """``GenericWorld.perform_agent_action``, minus the events.

        An action that cannot be carried out is simply not carried out; the
        engine records ``INVALID_ACTION`` and moves on, and so do we.
        """
        a = self.agents[i]
        if action in _DELTA:
            dx, dy = _DELTA[action]
            if self.tile_is_free(a.x + dx, a.y + dy):
                a.x += dx
                a.y += dy
        elif action == "BOMB" and a.bombs_left:
            self.bombs.append(SimBomb(a.x, a.y, i))
            a.bombs_left = False

    def step(self, actions: dict[int, str] | list[str],
             perm: list[int] | np.ndarray | None = None) -> None:
        """Advance the world one step.

        ``actions`` maps an agent's index in :attr:`agents` to its action (a
        plain list is read as one entry per *active* agent, in ``active``
        order).  ``perm`` is a permutation of ``range(len(active))`` -- the
        engine draws it from ``world.rng`` and records it in the replay, so a
        recorded game can be re-run exactly; passing ``None`` applies actions in
        ``active`` order, which is what a search that does not model the
        ordering should use.
        """
        if not isinstance(actions, dict):
            actions = {i: a for i, a in zip(self.active, actions)}
        order = range(len(self.active)) if perm is None else perm

        self.step_no += 1
        self.killed = []

        for k in order:
            i = self.active[int(k)]
            action = actions.get(i, "WAIT")
            if action in ACTIONS:
                self.perform_action(i, action)

        self._collect_coins()
        self._update_explosions()
        self._update_bombs()
        self._evaluate_explosions()

    def _collect_coins(self) -> None:
        for n, (cx, cy, collectable) in enumerate(self.coins):
            if not collectable:
                continue
            for i in self.active:
                a = self.agents[i]
                if a.x == cx and a.y == cy:
                    # Set inside the inner loop, exactly as the engine does: the
                    # `collectable` test is in the outer loop only.
                    self.coins[n] = (cx, cy, False)
                    a.score += REWARD_COIN

    def _update_explosions(self) -> None:
        remaining = []
        for ex in self.explosions:
            ex.timer -= 1
            if ex.timer <= 0:
                ex.stage += 1
                if ex.stage < len(EXPLOSION_STAGES):
                    ex.timer = EXPLOSION_STAGES[ex.stage]
                    if ex.stage == 1:
                        self.agents[ex.owner].bombs_left = True
                else:
                    continue        # ``next_stage`` -> IndexError -> stage None
            remaining.append(ex)
        self.explosions = remaining

    def _update_bombs(self) -> None:
        alive = []
        for b in self.bombs:
            if b.timer > 0:
                b.timer -= 1
                alive.append(b)
                continue
            coords = blast_coords(self.arena, b.x, b.y)
            for cx, cy in coords:
                if self.arena[cx, cy] == CRATE:
                    self.arena[cx, cy] = FREE
                    for n, (kx, ky, collectable) in enumerate(self.coins):
                        if (kx, ky) == (cx, cy):
                            self.coins[n] = (kx, ky, True)
            self.explosions.append(SimExplosion(coords, b.owner))
        self.bombs = alive

    def _evaluate_explosions(self) -> None:
        """Kill whoever is standing in a stage-0 blast.

        The engine deduplicates *deaths* through a set but awards the kill
        **per explosion**, so an agent caught in two overlapping blasts pays
        once and pays two different owners five points each.  That asymmetry is
        reproduced rather than tidied up.
        """
        hit: dict[int, int] = {}
        for ex in self.explosions:
            if not ex.dangerous:
                continue
            for i in self.active:
                a = self.agents[i]
                if a.dead or (a.x, a.y) not in ex.coords:
                    continue
                hit.setdefault(i, ex.owner)
                if ex.owner != i:
                    self.agents[ex.owner].score += REWARD_KILL
        for i in sorted(hit):
            self.agents[i].dead = True
            self.active.remove(i)
            self.killed.append((i, hit[i]))

    # -- queries ----------------------------------------------------------
    def round_over(self) -> bool:
        """``time_to_stop``, minus the training-agent clause a search cannot see."""
        if not self.active:
            return True
        if (len(self.active) == 1
                and not (self.arena == CRATE).any()
                and not any(c for _, _, c in self.coins)
                and not self.bombs and not self.explosions):
            return True
        return self.step_no >= MAX_STEPS

    def danger_map(self) -> np.ndarray:
        """Tiles lethal *right now*, i.e. covered by a stage-0 explosion."""
        out = np.zeros_like(self.arena, dtype=bool)
        for ex in self.explosions:
            if ex.dangerous:
                for cx, cy in ex.coords:
                    out[cx, cy] = True
        return out

    def state(self):
        """A hashable digest of everything a step can change.

        This is the conformance test's comparison key, so it must cover every
        field the engine mutates and nothing it does not (no trophies, no
        events).  Bombs and explosions are sorted: the engine's list order is an
        insertion artefact and two worlds that differ only in it are the same
        position.
        """
        return (
            self.step_no,
            self.arena.tobytes(),
            tuple((a.x, a.y, a.bombs_left, a.dead, a.score) for a in self.agents),
            tuple(self.active),
            tuple(sorted((b.x, b.y, b.owner, b.timer) for b in self.bombs)),
            tuple(sorted((e.coords, e.owner, e.timer, e.stage)
                         for e in self.explosions)),
            tuple(self.coins),
        )
