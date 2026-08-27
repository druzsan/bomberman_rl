"""Actor process for the deep Q-network: plays rounds, ships episodes.

Unlike the tabular harness -- where the actor *is* the learner, because a
tabular update is a single cell write -- the deep actor learns nothing.  It
holds a CPU copy of the network, refreshes it from shared memory whenever the
learner publishes, and pushes finished episodes down a queue.

Started with the ``spawn`` context on purpose: the driver process owns a CUDA
context, and forking a process that has initialised CUDA is a well-known way to
get a hang that only reproduces under load.

Opponents may be given as ``"<agent>@<path/to/model>"``, which is how the
self-play league works: the frozen checkpoint is loaded by that opponent's own
``setup`` through ``BOMBERMAN_MODEL``.  The variable is set only around the
world construction that needs it, because every agent in the process reads it.

The league itself arrives through a file plus a version counter in the shared
control vector, not through the config: actors are started with ``spawn`` and
therefore hold a *pickled copy* of the stage list, so a driver that mutated its
own copy would have promoted checkpoints into a league no actor ever played.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dqn_shared import AttachedPolicy, PolicySpec
from .engine import (
    REPO,
    disable_file_logging,
    make_world,
    quiet_logging,
    reset_engine_loggers,
    set_model_env,
)

#: Worlds are expensive to build (each one attaches logger handlers), so they
#: are cached; a league with many frozen opponents would otherwise grow the
#: cache without bound.
MAX_WORLDS = 12


@dataclass
class ActorConfig:
    agent: str
    stages: list[dict]
    reward: object
    seed: int
    spec: PolicySpec
    #: Where to look for ``league.json`` when the league version changes.
    run_dir: str = ""


def split_opponent(name: str) -> tuple[str, str | None]:
    """``"dqn_agent@runs/.../model.pt"`` -> ``("dqn_agent", "runs/.../model.pt")``."""
    if "@" in name:
        agent, path = name.split("@", 1)
        return agent, path
    return name, None


class Actor:
    def __init__(self, worker_id: int, cfg: ActorConfig, sink):
        import torch

        torch.set_num_threads(1)
        self.worker_id = worker_id
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed * 1000 + worker_id)
        self.worlds: dict[tuple, object] = {}
        self.stage = -1
        self.league_version = 0.0

        quiet_logging()
        disable_file_logging()
        random.seed(cfg.seed * 977 + worker_id)

        self.policy = AttachedPolicy(cfg.spec, worker_id)
        self.ctrl = self.policy.ctrl
        self.callbacks = importlib.import_module(f"agent_code.{cfg.agent}.callbacks")
        self.train_module = importlib.import_module(f"agent_code.{cfg.agent}.train")
        self.callbacks.TRAINING_POLICY = self.policy
        self.train_module.EPISODE_SINK = sink
        self.train_module.REWARD_CONFIG = cfg.reward
        self.ctrl_slots = self.train_module.CTRL

    # -- curriculum -------------------------------------------------------
    def _apply_stage(self, stage_index: int) -> None:
        from .engine import patch_settings

        stage = self.cfg.stages[stage_index]
        reset_engine_loggers()
        self.worlds.clear()
        patch_settings(cols=stage.get("cols"), rows=stage.get("rows"),
                       max_steps=stage.get("max_steps"),
                       crate_density=stage.get("crate_density"),
                       coin_count=stage.get("coin_count"),
                       scenario=stage["scenario"])
        self.stage = stage_index

    def _world_for(self, stage: dict, opponents: tuple[str, ...]):
        key = (stage["scenario"], opponents)
        world = self.worlds.get(key)
        if world is not None:
            return world
        if len(self.worlds) >= MAX_WORLDS:
            reset_engine_loggers()
            self.worlds.clear()
        parsed = [split_opponent(o) for o in opponents]
        specs = [(self.cfg.agent, True)] + [(name, False) for name, _ in parsed]
        model = next((path for _, path in parsed if path), None)
        set_model_env(model)
        try:
            world = make_world(specs, scenario=stage["scenario"],
                               log_dir=str(REPO / "logs" / f"actor-{os.getpid()}"),
                               continue_without_training=False)
        finally:
            set_model_env(None)
        self.worlds[key] = world
        return world

    def _refresh_league(self) -> None:
        """Adopt a new opponent mixture published by the driver."""
        version = float(self.ctrl[self.ctrl_slots.LEAGUE_VERSION])
        if version == self.league_version or not self.cfg.run_dir:
            return
        path = Path(self.cfg.run_dir) / "league.json"
        if not path.exists():
            return
        published = json.loads(path.read_text())
        self.league_version = version
        for index, mix in published.items():
            self.cfg.stages[int(index)]["opponent_mix"] = [(w, list(o)) for w, o in mix]

    def _sample_opponents(self, stage: dict) -> tuple[str, ...]:
        mix = stage["opponent_mix"]
        weights = np.array([w for w, _ in mix], dtype=float)
        i = int(self.rng.choice(len(mix), p=weights / weights.sum()))
        return tuple(mix[i][1])

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        ctrl = self.ctrl
        while ctrl[self.ctrl_slots.STOP] < 0.5:
            wanted = int(ctrl[self.ctrl_slots.STAGE])
            if wanted != self.stage:
                self._apply_stage(wanted)
            stage = self.cfg.stages[self.stage]
            self._refresh_league()
            self.policy.maybe_refresh()
            world = self._world_for(stage, self._sample_opponents(stage))
            world.rng = np.random.default_rng(int(self.rng.integers(1 << 62)))
            world.new_round()
            while world.running and ctrl[self.ctrl_slots.STOP] < 0.5:
                world.do_step()
            if world.running:
                world.end_round()


def run_actor(worker_id: int, cfg: ActorConfig, sink) -> None:
    try:
        Actor(worker_id, cfg, sink).run()
    except BaseException:
        # Without this the driver only sees "exit 1" and the cause is lost in
        # the noise of twenty simultaneous restarts.
        traceback.print_exc()
        raise
