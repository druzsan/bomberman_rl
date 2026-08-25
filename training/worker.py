"""Actor process: plays rounds through the stock engine and learns in place.

One world per opponent mix, reused forever (constructing a world attaches a new
handler to a module-level logger, so building thousands of them in one process
leaks handlers and grinds to a halt).  Curriculum knobs are monkeypatched in
*this* process only -- ``settings.py`` in the repository is never edited, which
is what keeps the submitted agent independent of our training setup.
"""

from __future__ import annotations

import importlib
import os
import random
import traceback
from dataclasses import dataclass

import numpy as np

from .engine import (
    REPO,
    disable_file_logging,
    make_world,
    quiet_logging,
    reset_engine_loggers,
)
from .shared import AttachedTables, TableSpec


@dataclass
class WorkerConfig:
    agent: str
    stages: list[dict]
    reward: object
    seed: int = 0


class Actor:
    def __init__(self, worker_id: int, cfg: WorkerConfig, spec: TableSpec, sink):
        self.worker_id = worker_id
        self.cfg = cfg
        self.tables = AttachedTables(spec, worker_id)
        self.rng = np.random.default_rng(cfg.seed * 1000 + worker_id)
        self.sink = sink
        self.worlds: dict[tuple, object] = {}
        self.stage = -1

        quiet_logging()
        disable_file_logging()
        random.seed(cfg.seed * 977 + worker_id)

        self.callbacks = importlib.import_module(f"agent_code.{cfg.agent}.callbacks")
        self.train_module = importlib.import_module(f"agent_code.{cfg.agent}.train")
        self.callbacks.TRAINING_TABLES = self.tables
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
        if world is None:
            specs = [(self.cfg.agent, True)] + [(o, False) for o in opponents]
            world = make_world(specs, scenario=stage["scenario"],
                               log_dir=str(REPO / "logs" / f"actor-{os.getpid()}"),
                               continue_without_training=False)
            self.worlds[key] = world
        return world

    def _sample_opponents(self, stage: dict) -> tuple[str, ...]:
        mix = stage["opponent_mix"]
        weights = np.array([w for w, _ in mix], dtype=float)
        i = int(self.rng.choice(len(mix), p=weights / weights.sum()))
        return tuple(mix[i][1])

    # -- main loop --------------------------------------------------------
    def run(self) -> None:
        ctrl = self.tables.ctrl
        while ctrl[self.ctrl_slots.STOP] < 0.5:
            wanted = int(ctrl[self.ctrl_slots.STAGE])
            if wanted != self.stage:
                self._apply_stage(wanted)
            stage = self.cfg.stages[self.stage]
            world = self._world_for(stage, self._sample_opponents(stage))
            world.rng = np.random.default_rng(int(self.rng.integers(1 << 62)))
            world.new_round()
            while world.running and ctrl[self.ctrl_slots.STOP] < 0.5:
                world.do_step()
            if world.running:
                world.end_round()


def run_actor(worker_id: int, cfg: WorkerConfig, spec: TableSpec, sink) -> None:
    try:
        Actor(worker_id, cfg, spec, sink).run()
    except BaseException:
        # Without this the driver only sees "exit 1" and the cause is lost in
        # the noise of twenty simultaneous restarts.
        traceback.print_exc()
        raise
