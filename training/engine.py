"""Thin wrappers around the stock engine, shared by actors and eval workers.

Two engine facts shape this module:

* Every ``BombeRLeWorld`` and every ``Agent`` construction attaches a new
  ``FileHandler`` to a module-level logger, so building thousands of worlds in
  one process leaks handlers and grinds to a halt.  Therefore: **one world per
  process, reused across rounds** via :func:`play_round`.
* ``world.rng`` drives both the board layout and the per-step action
  permutation, so re-seeding it before ``new_round`` makes a round fully
  reproducible from a single integer.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: Stock rule constants; an eval process asserts these so a curriculum
#: monkeypatch can never silently leak into a reported number.
STOCK_SETTINGS = {
    "COLS": 17, "ROWS": 17, "MAX_STEPS": 400, "BOMB_POWER": 3, "BOMB_TIMER": 4,
    "EXPLOSION_TIMER": 2, "TIMEOUT": 0.5, "REWARD_KILL": 5, "REWARD_COIN": 1,
}
STOCK_SCENARIOS = {
    "empty": {"CRATE_DENSITY": 0, "COIN_COUNT": 0},
    "coin-heaven": {"CRATE_DENSITY": 0, "COIN_COUNT": 50},
    "loot-crate": {"CRATE_DENSITY": 0.75, "COIN_COUNT": 50},
    "classic": {"CRATE_DENSITY": 0.75, "COIN_COUNT": 9},
}


def assert_stock_settings() -> None:
    import settings as s

    for key, want in STOCK_SETTINGS.items():
        got = getattr(s, key)
        if got != want:
            raise AssertionError(f"settings.{key} is {got!r}, expected stock {want!r}")
    for name, want in STOCK_SCENARIOS.items():
        if s.SCENARIOS.get(name) != want:
            raise AssertionError(f"settings.SCENARIOS[{name!r}] is not stock")


def patch_settings(*, cols=None, rows=None, max_steps=None, crate_density=None,
                   coin_count=None, scenario="classic") -> None:
    """Apply curriculum knobs **in this process only**.

    The repository's ``settings.py`` is never edited, which is what keeps the
    submitted agent independent of our training setup.
    """
    import settings as s

    if cols is not None:
        s.COLS = cols
    if rows is not None:
        s.ROWS = rows
    if max_steps is not None:
        s.MAX_STEPS = max_steps
    if crate_density is not None or coin_count is not None:
        info = dict(s.SCENARIOS[scenario])
        if crate_density is not None:
            info["CRATE_DENSITY"] = crate_density
        if coin_count is not None:
            info["COIN_COUNT"] = coin_count
        s.SCENARIOS = dict(s.SCENARIOS)
        s.SCENARIOS[scenario] = info


def make_args(*, scenario="classic", log_dir=None, seed=None,
              continue_without_training=True) -> SimpleNamespace:
    log_dir = log_dir or str(REPO / "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
        replay=None, make_video=False, continue_without_training=continue_without_training,
        log_dir=log_dir, save_stats=False, match_name=None, seed=seed,
        silence_errors=False, scenario=scenario,
    )


def quiet_logging() -> None:
    logging.disable(logging.CRITICAL)


@dataclass
class RoundRecord:
    """Everything the report might want, recorded per round, never aggregated away."""

    seed: int
    score: int = 0
    coins: int = 0
    kills: int = 0
    crates: int = 0
    bombs: int = 0
    invalid: int = 0
    moves: int = 0
    steps_survived: int = 0
    round_steps: int = 0
    suicide: bool = False
    survived: bool = False
    rank: int = 1
    opponent_scores: list[int] = field(default_factory=list)
    think_ms_mean: float = 0.0
    think_ms_p99: float = 0.0
    think_timeouts: int = 0
    events: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class ThinkTimer:
    """Record per-``act`` wall time by patching the runner in *this* process only.

    The tournament timeout is live during evaluation because ``train`` is
    ``False``, so a latency regression shows up as forced ``WAIT``s and a score
    drop.  Measuring it explicitly turns that into a number we can gate on.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.durations: list[float] = []
        self._orig = None

    def __enter__(self):
        from agents import AgentRunner

        self._orig = AgentRunner.process_event
        durations = self.durations
        agent_name = self.agent_name
        orig = self._orig

        def timed(runner, event_name, *args):
            if event_name != "act" or runner.agent_name != agent_name:
                return orig(runner, event_name, *args)
            t0 = time.perf_counter()
            try:
                return orig(runner, event_name, *args)
            finally:
                durations.append(time.perf_counter() - t0)

        AgentRunner.process_event = timed
        return self

    def __exit__(self, *exc):
        from agents import AgentRunner

        AgentRunner.process_event = self._orig
        return False


def make_world(agent_specs, *, scenario="classic", log_dir=None,
               continue_without_training=True):
    """Build one world. Call once per process and reuse it (see module docstring)."""
    from environment import BombeRLeWorld

    args = make_args(scenario=scenario, log_dir=log_dir,
                     continue_without_training=continue_without_training)
    return BombeRLeWorld(args, agent_specs)


def play_round(world, seed: int, *, me: int = 0, timer: ThinkTimer | None = None) -> RoundRecord:
    """Play one round with a controlled world seed and record our agent's outcome."""
    import settings as s

    if timer is not None:
        timer.durations.clear()
    world.rng = np.random.default_rng(seed)
    world.new_round()
    agent = world.agents[me]
    events: Counter[str] = Counter()
    steps_survived = 0
    while world.running:
        world.do_step()
        if not agent.dead:
            steps_survived = world.step
        events.update(agent.events)

    stats = agent.statistics
    scores = [a.score for a in world.agents]
    rank = 1 + sum(1 for i, sc in enumerate(scores) if sc > scores[me])
    rec = RoundRecord(
        seed=seed, score=agent.score, coins=stats.get("coins", 0),
        kills=stats.get("kills", 0), crates=stats.get("crates", 0),
        bombs=stats.get("bombs", 0), invalid=stats.get("invalid", 0),
        moves=stats.get("moves", 0), steps_survived=steps_survived,
        round_steps=world.step, suicide=stats.get("suicides", 0) > 0,
        survived=not agent.dead, rank=rank,
        opponent_scores=[sc for i, sc in enumerate(scores) if i != me],
        events=dict(events),
    )
    if timer is not None and timer.durations:
        d = np.asarray(timer.durations) * 1000.0
        rec.think_ms_mean = float(d.mean())
        rec.think_ms_p99 = float(np.percentile(d, 99))
        rec.think_timeouts = int((d >= s.TIMEOUT * 1000.0).sum())
    return rec


def set_model_env(path: str | os.PathLike | None) -> None:
    """Point the shipped ``callbacks.setup`` at a specific checkpoint artifact."""
    if path is None:
        os.environ.pop("BOMBERMAN_MODEL", None)
    else:
        os.environ["BOMBERMAN_MODEL"] = str(Path(path).resolve())
