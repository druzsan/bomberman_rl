"""Ape-X style training driver for the deep Q-network (S3).

Same contract as the tabular driver (``dev/plan.md`` §9): progress is counted in
environment steps, every checkpoint exports the *shipped* inference artifact and
is played in the real engine with ``train = False``, evaluation is asynchronous
on a depth-1 drop-stale queue, and ``metrics.jsonl`` is the single source of
truth.  What differs is where the learning happens: the tabular actors *are* the
learner, while here N CPU actors only generate episodes and one GPU process owns
the network.

    uv run python -m training.dqn_driver --config training/configs/s3.py --smoke
    uv run python -m training.dqn_driver --config training/configs/s3.py --tag main

Layout of one loop iteration: drain the episode queue into the replay buffer,
take a chunk of gradient steps, publish weights if due, then handle schedules,
metrics, checkpoints and evaluation.  The chunk size keeps the queue drained
about fifteen times a second, which is frequent enough that actors never block
on a full queue and rare enough that the GPU is not stalled by Python.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.sync_lib import sync as sync_lib

from .checkpoint import copy_checkpoint, git_sha, prune
from .driver import ANCHOR_SUITE, EvalService
from .engine import REPO
from .evaluate import summarise
from .metrics import JsonlSink
from .schedule import Curriculum


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
    if isinstance(obj, Path):
        return str(obj)
    return obj


class DQNDriver:
    def __init__(self, cfg: dict, run_dir: Path, args):
        from lib.qnet import QNet, QNetConfig

        from .dqn_learner import DQNLearner
        from .dqn_shared import SharedPolicy

        self.cfg = cfg
        self.run = run_dir
        self.args = args
        self.curriculum = Curriculum(cfg["stages"], cfg)
        self.total_steps = self.curriculum.total_steps
        for stage in cfg["stages"]:
            if stage.get("cols") or stage.get("rows"):
                raise SystemExit("the packed replay buffer assumes a 17x17 board; "
                                 "a board-size curriculum is not supported for S3")

        train_module = importlib.import_module(f"agent_code.{cfg['agent']}.train")
        self.CTRL = train_module.CTRL

        net_cfg = QNetConfig(plane_set=cfg["plane_set"], channels=cfg["channels"],
                             blocks=cfg["blocks"])
        self.net_config = asdict(net_cfg)
        n_params = QNet(net_cfg).n_parameters()
        self.shared = SharedPolicy(n_params, self.net_config)
        self.learner = DQNLearner(cfg, device=cfg.get("device", "cuda"),
                                  seed=cfg["seed"])
        self.n_params = n_params

        self.sink = JsonlSink(run_dir / "metrics.jsonl")
        self.eval_results: queue.Queue = queue.Queue()
        self.evaluator = EvalService(cfg["agent"], cfg["eval_rounds"],
                                     cfg["eval_workers"], self.eval_results)
        self.best_score = -np.inf
        self.best_progress = -1
        self.stage_boundaries: set[int] = set()
        self.stopping = False
        self.actor_restarts = 0
        self.episodes_seen = 0
        self.expert_episodes = 0
        self.league: list[str] = []
        self._init_ctrl()

    # -- setup ------------------------------------------------------------
    def _init_ctrl(self) -> None:
        c, C = self.shared.ctrl, self.CTRL
        c[C.EPSILON] = self.cfg["stages"][0]["eps"][0]
        c[C.SHAPING] = 1.0
        c[C.ALLOW_BOMB] = 1.0 if self.cfg["stages"][0].get("allow_bomb", True) else 0.0
        c[C.MASK_LETHAL] = float(self.curriculum.mask_level(0))
        c[C.STAGE] = 0
        c[C.N_ACTORS] = float(self.cfg["actors"])
        c[C.EPS_LADDER] = 1.0 if self.cfg.get("eps_ladder", True) else 0.0

    def progress(self) -> int:
        C = self.CTRL
        n = self.cfg["actors"]
        return int(self.shared.ctrl[C.WORKER_STEPS:C.WORKER_STEPS + n].sum())

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

    # -- actors -----------------------------------------------------------
    def start_actors(self) -> None:
        from .dqn_actor import ActorConfig, run_actor

        # ``spawn``, not ``fork``: this process owns a CUDA context and forking
        # one of those is a reliable way to get a hang that only shows up under
        # load.
        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue(maxsize=self.cfg.get("queue_size", 256))
        self.actor_cfg = ActorConfig(agent=self.cfg["agent"], stages=self.cfg["stages"],
                                     reward=self.cfg["reward"], seed=self.cfg["seed"],
                                     spec=self.shared.spec, run_dir=str(self.run))
        self.actors = []
        for i in range(self.cfg["actors"]):
            p = self.ctx.Process(target=run_actor, args=(i, self.actor_cfg, self.queue),
                                 daemon=True)
            p.start()
            self.actors.append(p)

    def supervise_actors(self, progress: int) -> None:
        from .dqn_actor import run_actor

        for i, p in enumerate(self.actors):
            if p.is_alive():
                continue
            self.actor_restarts += 1
            if self.actor_restarts > 3 * len(self.actors) and progress == 0:
                raise RuntimeError("actors keep dying without producing a single step; "
                                   "see the traceback above")
            self.log(f"actor {i} died (exit {p.exitcode}), restarting")
            new = self.ctx.Process(target=run_actor,
                                   args=(i, self.actor_cfg, self.queue), daemon=True)
            new.start()
            self.actors[i] = new

    def drain(self, window: list) -> int:
        added = 0
        try:
            while True:
                ep = self.queue.get_nowait()
                added += self.learner.add_episode(ep)
                self.episodes_seen += 1
                window.append({k: ep[k] for k in
                               ("steps", "shaped_return", "true_return", "score",
                                "suicide", "survived", "events")})
        except queue.Empty:
            pass
        return added

    # -- schedules --------------------------------------------------------
    def update_schedules(self, progress: int) -> int:
        C = self.CTRL
        c = self.shared.ctrl
        index, frac = self.curriculum.locate(progress)
        stage = self.cfg["stages"][index]

        c[C.EPSILON] = self.curriculum.epsilon(index, frac)
        c[C.ALLOW_BOMB] = 1.0 if stage.get("allow_bomb", True) else 0.0
        c[C.SHAPING] = self.curriculum.shaping(index, frac)

        level = self.curriculum.mask_level(progress)
        if int(c[C.MASK_LETHAL]) != level:
            c[C.MASK_LETHAL] = float(level)
            self.log(f"safety mask -> level {level} at {progress} steps")

        if int(c[C.STAGE]) != index:
            c[C.STAGE] = index
            self.stage_boundaries.add(progress)
            self.log(f"stage -> {index} ({stage['name']}) at {progress} steps")
        return index

    # -- metrics ----------------------------------------------------------
    def train_metrics(self, window: list, elapsed: float, steps_delta: int,
                      grad_delta: int) -> dict:
        C = self.CTRL
        c = self.shared.ctrl
        m = {
            "train/env_steps_per_s": steps_delta / max(elapsed, 1e-9),
            "train/grad_steps_per_s": grad_delta / max(elapsed, 1e-9),
            "train/replay_ratio": grad_delta * self.cfg["batch_size"] /
                                  max(steps_delta, 1),
            "train/episodes": len(window),
            "train/eps": float(c[C.EPSILON]),
            "train/shaping_scale": float(c[C.SHAPING]),
            "train/stage": float(c[C.STAGE]),
            "train/mask_level": float(c[C.MASK_LETHAL]),
            "train/actor_queue_depth": float(self.queue.qsize()),
            "train/league_size": float(len(self.league)),
            **self.learner._last,
        }
        if window:
            m["train/episode_len"] = float(np.mean([e["steps"] for e in window]))
            m["train/episode_return_shaped"] = float(np.mean([e["shaped_return"]
                                                              for e in window]))
            m["train/episode_return_true"] = float(np.mean([e["true_return"]
                                                            for e in window]))
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
                self.promote(progress, agg)

    def promote(self, progress: int, agg: dict) -> None:
        src = self.run / "checkpoints" / f"step_{progress:010d}"
        if agg["score_ci95_lo"] > self.best_score:
            self.best_score = agg["score_mean"]
            self.best_progress = progress
            if src.exists():
                copy_checkpoint(src, self.run / "checkpoints" / "best")
                self.log(f"promoted step {progress} (score {agg['score_mean']:.2f})")
        # A promoted checkpoint joins the self-play league (plan §9.8): the
        # archive and the league are the same thing rather than two systems.
        joins_league = (self.cfg.get("league_after", 0)
                        and progress >= self.cfg["league_after"]
                        and agg["score_mean"] >= self.cfg.get("league_min_score", 0.0)
                        and src.exists())
        if joins_league:
            keep = self.run / "league" / f"step_{progress:010d}.pt"
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "model.pt", keep)
            self.league.append(str(keep))
            self.league = self.league[-self.cfg.get("league_size", 8):]
            self.push_league()

    def push_league(self) -> None:
        """Publish an opponent mixture that includes the frozen checkpoints.

        Written to ``league.json`` and announced by bumping a counter in the
        shared control vector.  Actors are started with ``spawn`` and hold a
        pickled copy of the stage list, so mutating the driver's own copy would
        change nothing that any actor ever sees.
        """
        share = float(self.cfg.get("league_share", 0.3))
        entries = [(share / len(self.league), [f"{self.cfg['agent']}@{p}"] * 3)
                   for p in self.league]
        index = len(self.cfg["stages"]) - 1
        stage = self.cfg["stages"][index]
        base = [(w, o) for w, o in stage["opponent_mix"]
                if not any("@" in x for x in o)]
        total = sum(w for w, _ in base) or 1.0
        stage["opponent_mix"] = [(w * (1 - share) / total, o) for w, o in base] + entries
        (self.run / "league.json").write_text(json.dumps(
            {str(index): stage["opponent_mix"]}, default=str))
        self.shared.ctrl[self.CTRL.LEAGUE_VERSION] += 1
        self.log(f"league now {len(self.league)} opponents "
                 f"({share:.0%} of the last stage's episodes)")

    # -- checkpoints ------------------------------------------------------
    def checkpoint(self, progress: int, stage_index: int) -> Path:
        from .dqn_learner import write_checkpoint

        return write_checkpoint(
            self.run / "checkpoints", progress, self.learner,
            {"stage": stage_index, "wall_s": time.time() - self.t0,
             "grad_steps": self.learner.grad_steps,
             "eps": float(self.shared.ctrl[self.CTRL.EPSILON]),
             "plane_set": self.cfg["plane_set"],
             "use_mask": bool(self.cfg.get("ship_mask", False))},
            with_state=self.cfg.get("save_train_state", True))

    def suites_for(self, stage_index: int) -> dict:
        suites = {"anchor": {**ANCHOR_SUITE, "rounds": self.cfg["eval_rounds"]}}
        stage = self.cfg["stages"][stage_index]
        if stage.get("eval_suite"):
            suites["stage"] = {**stage["eval_suite"],
                               "rounds": stage["eval_suite"].get(
                                   "rounds", self.cfg["eval_rounds"])}
        return suites

    # -- run --------------------------------------------------------------
    def preload(self) -> None:
        path = self.cfg.get("pretrained")
        if not path:
            return
        meta = self.learner.load_pretrained(Path(path))
        self.log(f"initialised from {path} ({meta.get('kind', 'unknown')}, "
                 f"{meta.get('samples', '?')} samples)")

    def seed_expert_replay(self) -> None:
        """Load the demonstration buffer the actors never overwrite (DQfD)."""
        path = self.cfg.get("expert_episodes")
        if not path:
            return
        from .bc import iter_episodes

        capacity = int(self.cfg.get("expert_capacity", 1_000_000))
        added = self.learner.load_expert(iter_episodes(Path(path)), capacity)
        self.log(f"loaded {added} expert transitions from {path} "
                 f"({self.cfg.get('expert_share', 0.25):.0%} of each minibatch, "
                 f"annealed over {self.cfg.get('expert_anneal_frac', 0):.0%} of the run)")

    def run_loop(self) -> None:
        self.t0 = time.time()
        self.preload()
        self.shared.publish(self.learner.flat_weights())
        self.seed_expert_replay()
        self.start_actors()
        cfg = self.cfg
        window: list = []
        last_log = last_ckpt = last_eval = 0
        last_grad = 0
        last_time = time.time()
        stage_index = 0
        self.log(f"run {self.run.name}: {self.total_steps} env steps, "
                 f"{cfg['actors']} actors, {self.n_params} parameters, "
                 f"planes={cfg['plane_set']}, git {git_sha()[:8]}")
        while not self.stopping:
            self.drain(window)
            self.handle_eval_results()
            progress = self.progress()
            self.supervise_actors(progress)
            stage_index = self.update_schedules(progress)
            frac = min(1.0, progress / max(self.total_steps, 1))

            allowed = int(progress * cfg["train_ratio"]) if cfg["train_ratio"] else None
            did = 0
            if self.learner.ready() and (allowed is None
                                         or self.learner.grad_steps < allowed):
                for _ in range(cfg.get("grad_chunk", 8)):
                    self.learner.update(frac)
                    did += 1
                    if allowed is not None and self.learner.grad_steps >= allowed:
                        break
            else:
                time.sleep(0.02)

            if did and self.learner.grad_steps % cfg["broadcast_every"] < did:
                self.shared.publish(self.learner.flat_weights())

            if progress - last_log >= cfg["log_every"]:
                now = time.time()
                self.emit("train", progress,
                          {**self.train_metrics(window, now - last_time,
                                                progress - last_log,
                                                self.learner.grad_steps - last_grad),
                           "driver/actor_restarts": float(self.actor_restarts)})
                last_time, last_log = now, progress
                last_grad = self.learner.grad_steps
                window = []

            due_stage = self.stage_boundaries and max(self.stage_boundaries) > last_ckpt
            if progress - last_ckpt >= cfg["checkpoint_every"] or due_stage:
                path = self.checkpoint(progress, stage_index)
                last_ckpt = progress
                if progress - last_eval >= cfg["eval_every"] or due_stage:
                    self.evaluator.submit(progress, path / "model.pt",
                                          self.suites_for(stage_index))
                    last_eval = progress
                removed = prune(self.run / "checkpoints",
                                keep_last=cfg.get("keep_last", 8),
                                keep_best={self.run / "checkpoints" / "best"},
                                keep_stages=self.stage_boundaries)
                if removed:
                    self.emit("driver", progress, {"driver/checkpoints_pruned": removed})

            if progress >= self.total_steps:
                break
        self.finish(stage_index)

    def finish(self, stage_index: int) -> None:
        self.shared.ctrl[self.CTRL.STOP] = 1.0
        drain: list = []
        deadline = time.time() + 30
        while time.time() < deadline and any(p.is_alive() for p in self.actors):
            self.drain(drain)
            time.sleep(0.2)
        for p in self.actors:
            self.drain(drain)
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        progress = self.progress()
        final = self.checkpoint(progress, stage_index)
        copy_checkpoint(final, self.run / "checkpoints" / "last")
        self.log(f"final checkpoint at {progress} steps, "
                 f"{self.learner.grad_steps} gradient steps -> {final}")
        self.evaluator.submit(progress, final / "model.pt", self.suites_for(stage_index))
        deadline = time.time() + 900
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
        self.queue.close()
        self.queue.cancel_join_thread()
        best = self.run / "checkpoints" / "best"
        if not best.exists():
            copy_checkpoint(final, best)
        self.log(f"done. best anchor score {self.best_score:.2f} @ {self.best_progress}")
        self.shared.close()


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
        cfg["learn_start"] = min(cfg["learn_start"], 2_000)
        cfg["replay_capacity"] = min(cfg["replay_capacity"], 32_768)
        cfg["batch_size"] = min(cfg["batch_size"], 64)
        cfg["target_sync"] = min(cfg["target_sync"], 100)
        cfg["broadcast_every"] = min(cfg["broadcast_every"], 50)
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

    sync_lib(REPO / "agent_code" / cfg["agent"])

    driver = DQNDriver(cfg, run_dir, args)

    def handle_sigterm(signum, frame):
        driver.stopping = True

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    driver.run_loop()
    print(run_dir, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
