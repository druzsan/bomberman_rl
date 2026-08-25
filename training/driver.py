"""One training driver for every solution in the ladder.

A run is never a black box that emits a model at the end.  It emits a time
series of checkpoints, each one *actually played in the real environment* with
``train = False`` and the shipped ``callbacks.py``, plus an append-only
``metrics.jsonl`` that every figure and table in the report is derived from.

Invariants (``dev/plan.md`` §9.1):

* progress is counted in **environment steps**, never episodes -- episodes range
  from ~30 to 400 steps and an episode axis makes runs incomparable;
* every evaluated checkpoint runs an **anchor suite** that never changes, so one
  curve spans the whole project, plus a goal-specific **stage suite**;
* evaluation is **asynchronous with a depth-1 drop-stale queue**: it is several
  times more expensive than the training it monitors, so it must lag rather than
  throttle;
* eval processes assert stock ``settings`` on start, so a curriculum board size
  can never silently leak into a reported number.

    uv run python -m training.driver --config training/configs/s1.py --smoke
    uv run python -m training.driver --config training/configs/s1.py --tag main
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.sync_lib import sync as sync_lib

from .checkpoint import copy_checkpoint, git_sha, prune, write_checkpoint
from .engine import REPO
from .evaluate import evaluate, summarise
from .metrics import JsonlSink
from .shared import SharedTables
from .worker import WorkerConfig, run_actor

ANCHOR_SUITE = {"scenario": "classic", "opponents": ["rule_based_agent"] * 3}


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config()


def _jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class EvalService:
    """Background evaluation with a depth-1, drop-stale job queue."""

    def __init__(self, agent: str, rounds: int, workers: int, results: queue.Queue):
        self.agent = agent
        self.rounds = rounds
        self.workers = workers
        self.results = results
        self._job: tuple | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.dropped = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, progress: int, model: Path, suites: dict) -> None:
        with self._lock:
            if self._job is not None:
                self.dropped += 1
            self._job = (progress, model, suites)
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            progress, model, suites = job
            for name, suite in suites.items():
                try:
                    records = evaluate(
                        self.agent, suite["opponents"], scenario=suite["scenario"],
                        rounds=suite.get("rounds", self.rounds),
                        workers=self.workers, model=str(model), strict=True)
                    self.results.put((progress, name, records))
                except Exception as exc:  # an eval failure must never kill a run
                    self.results.put((progress, name, exc))

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


class Driver:
    def __init__(self, cfg: dict, run_dir: Path, args):
        self.cfg = cfg
        self.run = run_dir
        self.args = args
        self.stages = cfg["stages"]
        self.stage_ends = np.cumsum([s["steps"] for s in self.stages])
        self.total_steps = int(self.stage_ends[-1])

        # The control-vector layout lives with the learner, so a second model
        # only has to define its own CTRL rather than edit the driver.
        train_module = importlib.import_module(f"agent_code.{cfg['agent']}.train")
        from lib.features import FeatureSpec

        self.CTRL = train_module.CTRL
        spec = FeatureSpec(cfg["feature_set"])
        n_rows = getattr(train_module, "n_parameter_rows", lambda s: s.n_states)(spec)
        self.tables = SharedTables(n_rows, cfg["feature_set"], cfg["fold"])
        self.n_states = n_rows
        self._init_ctrl()

        self.sink = JsonlSink(run_dir / "metrics.jsonl")
        self.eval_results: queue.Queue = queue.Queue()
        self.evaluator = EvalService(cfg["agent"], cfg["eval_rounds"],
                                     cfg["eval_workers"], self.eval_results)
        self.q_avg = np.zeros_like(self.tables.q)
        self.avg_beta = float(cfg.get("export_average_beta", 0.0))
        self.best_score = -np.inf
        self.best_progress = -1
        self.stage_boundaries: set[int] = set()
        self.stopping = False
        self.actor_restarts = 0

    # -- setup ------------------------------------------------------------
    def _init_ctrl(self) -> None:
        c, C = self.tables.ctrl, self.CTRL
        cfg = self.cfg
        c[C.GAMMA] = cfg["gamma"]
        c[C.N_STEP] = cfg["n_step"]
        c[C.ALPHA] = cfg["alpha"]
        c[C.ALPHA_POWER] = cfg["alpha_power"]
        c[C.ALGO] = cfg["algo"]
        c[C.LEARN] = 1.0
        c[C.SHAPING] = 1.0
        c[C.EPSILON] = self.stages[0]["eps"][0]
        c[C.ALLOW_BOMB] = 1.0 if self.stages[0].get("allow_bomb", True) else 0.0
        c[C.MASK_LETHAL] = float(self.mask_level(0))
        c[C.STAGE] = 0

    def mask_level(self, progress: int) -> int:
        """Safety-curriculum level at this point of the run.

        Escaping a bomb needs four consecutive correct moves, so an untrained
        greedy policy dies within ~20 of the 400 steps and never sees the payoff
        that makes bombing worthwhile -- measured: one suicide per episode and a
        shaped return pinned at the death penalty.  Shielding the policy early
        (level 2) buys full-length episodes to learn from; annealing the shield
        off (level 1, then 0) then lets the agent learn what the shield was
        hiding, so the submitted policy is not dependent on it.

        Configured as ``[(start_fraction, level), ...]``.  ``mask_scope`` picks
        what the fraction is relative to: ``"run"`` shields the beginning of the
        whole curriculum, ``"stage"`` shields the beginning of *each* stage,
        which matters because a new opponent mix makes the old safety knowledge
        partly wrong again.
        """
        schedule = self.cfg.get("mask_schedule")
        if not schedule:
            return int(self.cfg.get("mask_lethal", 0))
        if self.cfg.get("mask_scope", "run") == "stage":
            index = min(int(np.searchsorted(self.stage_ends, progress, side="right")),
                        len(self.stages) - 1)
            start = 0 if index == 0 else int(self.stage_ends[index - 1])
            frac = (progress - start) / max(self.stages[index]["steps"], 1)
        else:
            frac = progress / max(self.total_steps, 1)
        level = schedule[0][1]
        for start, value in schedule:
            if frac >= start:
                level = value
        return int(level)

    def progress(self) -> int:
        C = self.CTRL
        n = self.cfg["actors"]
        return int(self.tables.ctrl[C.WORKER_STEPS:C.WORKER_STEPS + n].sum())

    # -- schedules --------------------------------------------------------
    def update_schedules(self, progress: int) -> int:
        C = self.CTRL
        c = self.tables.ctrl
        stage_index = int(np.searchsorted(self.stage_ends, progress, side="right"))
        stage_index = min(stage_index, len(self.stages) - 1)
        stage = self.stages[stage_index]
        start = 0 if stage_index == 0 else int(self.stage_ends[stage_index - 1])
        frac = np.clip((progress - start) / max(stage["steps"], 1), 0.0, 1.0)

        eps0, eps1 = stage["eps"]
        c[C.EPSILON] = eps0 + (eps1 - eps0) * frac

        # A constant step size on a ~200-state table written by 20 actors makes
        # the greedy policy bistable: measured evaluations swung between 47 and
        # 0.2 coins on consecutive checkpoints. Decaying it settles the policy.
        a0 = self.cfg["alpha"]
        a1 = self.cfg.get("alpha_end", a0)
        if a0 > 0:
            run_frac = min(1.0, progress / max(self.total_steps, 1))
            c[C.ALPHA] = a0 * (a1 / a0) ** run_frac if a1 > 0 else a0
        c[C.ALLOW_BOMB] = 1.0 if stage.get("allow_bomb", True) else 0.0

        level = self.mask_level(progress)
        if int(c[C.MASK_LETHAL]) != level:
            c[C.MASK_LETHAL] = float(level)
            self.log(f"safety mask -> level {level} at {progress} steps")

        anneal = self.cfg.get("shaping_anneal_frac", 0.0)
        if stage_index == len(self.stages) - 1 and anneal > 0 and frac > 1 - anneal:
            c[C.SHAPING] = float(np.clip((1.0 - frac) / anneal, 0.0, 1.0))
        else:
            c[C.SHAPING] = 1.0

        if int(c[C.STAGE]) != stage_index:
            c[C.STAGE] = stage_index
            self.stage_boundaries.add(progress)
            self.log(f"stage -> {stage_index} ({stage['name']}) at {progress} steps")
            return stage_index
        return stage_index

    # -- io ---------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.run / "driver.log", "a") as fh:
            fh.write(line + "\n")

    def emit(self, kind: str, progress: int, metrics: dict, **extra) -> None:
        self.sink.write({"ts": time.time(), "wall_s": time.time() - self.t0,
                         "progress": progress, "unit": "env_steps", "kind": kind,
                         "metrics": metrics, **extra})

    # -- run --------------------------------------------------------------
    def start_actors(self) -> None:
        ctx = mp.get_context("fork")
        self.queue = ctx.Queue(maxsize=10000)
        wcfg = WorkerConfig(agent=self.cfg["agent"], stages=self.stages,
                            reward=self.cfg["reward"], seed=self.cfg["seed"])
        self.actors = []
        for i in range(self.cfg["actors"]):
            p = ctx.Process(target=run_actor, args=(i, wcfg, self.tables.spec, self.queue),
                            daemon=True)
            p.start()
            self.actors.append(p)

    def supervise_actors(self, progress: int) -> None:
        """Restart dead actors; abort the run if they all die immediately."""
        ctx = mp.get_context("fork")
        wcfg = WorkerConfig(agent=self.cfg["agent"], stages=self.stages,
                            reward=self.cfg["reward"], seed=self.cfg["seed"])
        for i, p in enumerate(self.actors):
            if p.is_alive():
                continue
            self.actor_restarts += 1
            if self.actor_restarts > 3 * len(self.actors) and progress == 0:
                raise RuntimeError("actors keep dying without producing a single step; "
                                   "see the traceback above")
            self.log(f"actor {i} died (exit {p.exitcode}), restarting")
            new = ctx.Process(target=run_actor,
                              args=(i, wcfg, self.tables.spec, self.queue), daemon=True)
            new.start()
            self.actors[i] = new

    def collect_episodes(self, window: list) -> None:
        try:
            while True:
                window.append(self.queue.get_nowait())
        except queue.Empty:
            pass

    def train_metrics(self, window: list, progress: int, elapsed: float,
                      steps_delta: int) -> dict:
        C = self.CTRL
        c = self.tables.ctrl
        n = self.tables.n
        visited = n.sum(axis=1)
        m = {
            "train/env_steps_per_s": steps_delta / max(elapsed, 1e-9),
            "train/episodes": len(window),
            "train/eps": float(c[C.EPSILON]),
            "train/shaping_scale": float(c[C.SHAPING]),
            "train/stage": float(c[C.STAGE]),
            "train/mask_level": float(c[C.MASK_LETHAL]),
            "train/alpha": float(c[C.ALPHA]),
            "train/table_nonzero": float((visited > 0).sum()),
            "train/table_coverage_10": float((visited >= 10).sum()),
            "train/q_mean": float(self.tables.q[visited > 0].mean()) if (visited > 0).any() else 0.0,
            "train/q_max": float(self.tables.q.max()),
        }
        if window:
            m["train/episode_len"] = float(np.mean([e["steps"] for e in window]))
            m["train/episode_return_shaped"] = float(np.mean([e["shaped_return"] for e in window]))
            m["train/episode_return_true"] = float(np.mean([e["true_return"] for e in window]))
            m["train/episode_score"] = float(np.mean([e["score"] for e in window]))
            m["train/suicide_rate"] = float(np.mean([e["suicide"] for e in window]))
            m["train/survival_rate"] = float(np.mean([e["survived"] for e in window]))
            names = {k for e in window for k in e["events"]}
            for name in sorted(names):
                m[f"event/{name}_per_episode"] = float(
                    np.mean([e["events"].get(name, 0) for e in window]))
        return m

    def handle_eval_results(self) -> None:
        while True:
            try:
                progress, name, payload = self.eval_results.get_nowait()
            except queue.Empty:
                return
            if isinstance(payload, BaseException):
                self.log(f"eval {name}@{progress} failed: {payload!r}")
                continue
            agg = summarise(payload)
            self.emit("eval", progress, {f"eval/{name}/{k}": v for k, v in agg.items()},
                      suite=name)
            (self.run / "eval" / name).mkdir(parents=True, exist_ok=True)
            (self.run / "eval" / name / f"step_{progress:010d}.json").write_text(
                json.dumps({"aggregate": agg, "records": payload}, default=float))
            self.log(f"eval {name:6s} @{progress}: score {agg['score_mean']:6.2f} "
                     f"[{agg['score_ci95_lo']:.2f},{agg['score_ci95_hi']:.2f}] "
                     f"win {agg['win_rate']:.0%} surv {agg['survival_rate']:.0%} "
                     f"suicide {agg['suicide_rate']:.0%} coins {agg['coins_mean']:.2f} "
                     f"crates {agg['crates_mean']:.1f}")
            if name == "anchor":
                lower = agg["score_ci95_lo"]
                if lower > self.best_score:
                    self.best_score = agg["score_mean"]
                    self.best_progress = progress
                    src = self.run / "checkpoints" / f"step_{progress:010d}"
                    if src.exists():
                        copy_checkpoint(src, self.run / "checkpoints" / "best")
                        self.log(f"promoted step {progress} (score {agg['score_mean']:.2f})")

    def export_table(self) -> np.ndarray:
        """The table a checkpoint ships: Polyak-averaged when enabled."""
        if self.avg_beta <= 0:
            return self.tables.q
        # Never-visited entries stay exactly zero in both, so the average is
        # safe to take element-wise.
        return self.q_avg

    def checkpoint(self, progress: int, stage_index: int) -> Path:
        return write_checkpoint(
            self.run / "checkpoints", progress, self.export_table(), self.tables.n,
            self.cfg["feature_set"], self.cfg["fold"],
            {"stage": stage_index, "wall_s": time.time() - self.t0,
             "eps": float(self.tables.ctrl[self.CTRL.EPSILON]),
             "alpha": float(self.tables.ctrl[self.CTRL.ALPHA]),
             "averaged": self.avg_beta > 0})

    def suites_for(self, stage_index: int) -> dict:
        suites = {"anchor": {**ANCHOR_SUITE, "rounds": self.cfg["eval_rounds"]}}
        stage = self.stages[stage_index]
        if stage.get("eval_suite"):
            suites["stage"] = {**stage["eval_suite"],
                               "rounds": stage["eval_suite"].get("rounds",
                                                                self.cfg["eval_rounds"])}
        return suites

    def run_loop(self) -> None:
        self.t0 = time.time()
        self.start_actors()
        cfg = self.cfg
        window: list = []
        last_log = last_ckpt = last_eval = 0
        last_time = time.time()
        stage_index = 0
        self.log(f"run {self.run.name}: {self.total_steps} env steps, "
                 f"{cfg['actors']} actors, {self.n_states} states, git {git_sha()[:8]}")
        while not self.stopping:
            time.sleep(0.5)
            self.collect_episodes(window)
            self.handle_eval_results()
            progress = self.progress()
            self.supervise_actors(progress)
            stage_index = self.update_schedules(progress)

            if self.avg_beta > 0:
                # Polyak average of the table, which is what gets exported: the
                # raw table tracks the last few hundred samples per state.
                self.q_avg += self.avg_beta * (self.tables.q - self.q_avg)

            if progress - last_log >= cfg["log_every"]:
                now = time.time()
                self.emit("train", progress,
                          {**self.train_metrics(window, progress, now - last_time,
                                                progress - last_log),
                           "driver/actor_restarts": float(self.actor_restarts)})
                last_time, last_log = now, progress
                window = []

            due_stage = self.stage_boundaries and max(self.stage_boundaries) > last_ckpt
            if progress - last_ckpt >= cfg["checkpoint_every"] or due_stage:
                path = self.checkpoint(progress, stage_index)
                last_ckpt = progress
                if progress - last_eval >= cfg["eval_every"] or due_stage:
                    self.evaluator.submit(progress, path / "model.npz",
                                          self.suites_for(stage_index))
                    last_eval = progress
                removed = prune(self.run / "checkpoints", keep_last=5,
                                keep_best={self.run / "checkpoints" / "best"},
                                keep_stages=self.stage_boundaries)
                if removed:
                    self.emit("driver", progress, {"driver/checkpoints_pruned": removed})

            if progress >= self.total_steps:
                break
        self.finish(stage_index)

    def finish(self, stage_index: int) -> None:
        self.tables.ctrl[self.CTRL.STOP] = 1.0
        # Keep draining while joining: an actor blocked on a full metrics queue
        # can never exit, and p.join() would then hang forever.
        drain: list = []
        deadline = time.time() + 30
        while time.time() < deadline and any(p.is_alive() for p in self.actors):
            self.collect_episodes(drain)
            time.sleep(0.2)
        for p in self.actors:
            self.collect_episodes(drain)
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        progress = self.progress()
        final = self.checkpoint(progress, stage_index)
        copy_checkpoint(final, self.run / "checkpoints" / "last")
        self.log(f"final checkpoint at {progress} steps -> {final}")
        self.evaluator.submit(progress, final / "model.npz", self.suites_for(stage_index))
        deadline = time.time() + 600
        seen = 0
        want = len(self.suites_for(stage_index))
        while time.time() < deadline and seen < want:
            before = self.eval_results.qsize()
            self.handle_eval_results()
            seen += before
            time.sleep(1.0)
        self.handle_eval_results()
        self.evaluator.stop()
        self.sink.close()
        # Undelivered items in the metrics queue keep its feeder thread alive,
        # and multiprocessing joins that thread at interpreter exit -- which
        # hangs the process long after the run has finished.
        self.queue.close()
        self.queue.cancel_join_thread()
        best = self.run / "checkpoints" / "best"
        if not best.exists():
            copy_checkpoint(final, best)
        self.log(f"done. best anchor score {self.best_score:.2f} @ {self.best_progress}")
        self.tables.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="run")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--actors", type=int, default=None)
    p.add_argument("--eval-workers", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="override total env steps")
    p.add_argument("--seed", type=int, default=None)
    # One value each, repeatable. Not nargs="*": two greedy list options in a
    # row silently route the second option's arguments into the first, which
    # once cost a whole afternoon of sweeps that all ran the default n_step.
    p.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                   help="config patch, repeat for several")
    p.add_argument("--stage-override", action="append", default=[], metavar="KEY=JSON",
                   help="patch applied to every curriculum stage, repeat for several")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    for item in args.override:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise SystemExit(f"--override {key!r} is not a config key; "
                             f"known keys: {', '.join(sorted(cfg))}")
        cfg[key] = json.loads(value)
    for item in args.stage_override:
        key, value = item.split("=", 1)
        known = {k for s in cfg["stages"] for k in s}
        if key not in known:
            raise SystemExit(f"--stage-override {key!r} is not a stage key; "
                             f"known keys: {', '.join(sorted(known))}")
        for stage in cfg["stages"]:
            stage[key] = json.loads(value)
    if args.actors:
        cfg["actors"] = args.actors
    if args.eval_workers:
        cfg["eval_workers"] = args.eval_workers
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.smoke:
        cfg["actors"] = min(cfg["actors"], 3)
        cfg["eval_workers"] = 2
        cfg["eval_rounds"] = 4
        cfg["log_every"] = 2_000
        cfg["checkpoint_every"] = 5_000
        cfg["eval_every"] = 5_000
        total = 20_000
        share = max(1, total // len(cfg["stages"]))
        for s in cfg["stages"]:
            s["steps"] = share
    if args.steps:
        scale = args.steps / sum(s["steps"] for s in cfg["stages"])
        for s in cfg["stages"]:
            s["steps"] = max(1, int(s["steps"] * scale))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPO / "runs" / f"{stamp}-{cfg['agent']}-{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"config": _jsonable(cfg), "git_sha": git_sha(), "argv": sys.argv,
         "overrides": args.override, "stage_overrides": args.stage_override}, indent=1))

    # The agent imports its *vendored* copy of lib/; a stale copy is otherwise a
    # confusing mass actor crash rather than an error.
    sync_lib(REPO / "agent_code" / cfg["agent"])

    driver = Driver(cfg, run_dir, args)

    def handle_sigterm(signum, frame):
        driver.stopping = True

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    driver.run_loop()
    print(run_dir, flush=True)
    # Every artifact is on disk and every stream is flushed by now.  Leaving
    # through os._exit avoids multiprocessing's atexit handlers, which join
    # queue feeder threads and pool helpers and can block a finished run
    # indefinitely -- a real problem when a sweep runs dozens of these in a row.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
