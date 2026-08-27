"""Behaviour cloning from ``bfs_expert``, and the dataset it runs on.

The single biggest documented failure mode for this project is "the deep model
had not converged by the deadline".  The mitigation is to not start from noise:
record a few hundred thousand transitions from the strong rule-based expert,
train the *same* network on them supervised, and let DQN fine-tune from there.
It also produces the report's most informative figure -- score against
environment steps for {DQN from scratch, BC only, BC then DQN} -- which shows
exactly where reinforcement learning overtakes its teacher.

Two commands:

    uv run python -m training.bc collect --out data/expert --episodes 3000
    uv run python -m training.bc train  --data data/expert --out runs/bc/main

**Collection** drives the *deep agent's own* training callbacks with the expert
substituted for the behaviour policy (``train.TEACHER``).  Nothing about the
recording path is special-cased: the same ``lib.rewards`` configuration, the
same custom events, the same bit-packing and the same episode format as a real
actor.  The file is therefore usable twice -- as supervised data here, and as
the permanently-resident expert fraction of the DQfD replay buffer in
``training.dqn_driver``.

**Training** is DQfD's pre-training phase: a classification loss that makes the
expert action the ``argmax``, plus an n-step Q regression that gives the value
head a calibrated scale before any environment interaction happens.  Without the
second term the fine-tuning run spends its first hundred thousand steps merely
discovering how large a Q value should be.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from lib.board import ACTION_INDEX
from lib.encode import PACKED_BYTES, PLANE_SETS
from lib.qnet import QNet, QNetConfig
from lib.rewards import RewardConfig
from lib.symmetry import ACTION_MAP

from .engine import REPO, disable_file_logging, make_world, quiet_logging
from .replay import PrioritisedReplay

ARRAYS = ("packed", "step", "action", "reward", "legal")
DTYPES = {"packed": np.uint8, "step": np.uint16, "action": np.uint8,
          "reward": np.float32, "legal": np.uint8}

#: Opponent mixtures the expert plays against while recording.  Coverage of the
#: state space matters more here than difficulty: a dataset recorded only
#: against ``rule_based_agent`` teaches the network nothing about crowded
#: boards, because that opponent removes itself from half of them.
COLLECT_MIX = [
    (0.30, ["rule_based_agent"] * 3),
    (0.30, ["bfs_expert"] * 3),
    (0.20, ["coin_collector_agent"] * 3),
    (0.10, ["peaceful_agent"] * 3),
    (0.10, []),
]


# --------------------------------------------------------------------------
# dataset on disk
# --------------------------------------------------------------------------
def write_dataset(out: Path, episodes: list[dict]) -> dict:
    """Store episodes as flat ``.npy`` arrays plus their lengths."""
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(ep["reward"]) for ep in episodes], dtype=np.int32)
    for name in ARRAYS:
        np.save(out / f"{name}.npy",
                np.concatenate([ep[name] for ep in episodes]).astype(DTYPES[name]))
    np.save(out / "lengths.npy", lengths)
    meta = {"episodes": len(episodes), "transitions": int(lengths.sum()),
            "packed_bytes": PACKED_BYTES}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def append_dataset(out: Path, episodes: list[dict]) -> None:
    """Append a shard to an existing dataset directory."""
    if not (out / "lengths.npy").exists():
        write_dataset(out, episodes)
        return
    lengths = np.concatenate([np.load(out / "lengths.npy"),
                              np.array([len(ep["reward"]) for ep in episodes],
                                       dtype=np.int32)])
    for name in ARRAYS:
        old = np.load(out / f"{name}.npy")
        new = np.concatenate([ep[name] for ep in episodes]).astype(DTYPES[name])
        np.save(out / f"{name}.npy", np.concatenate([old, new]))
    np.save(out / "lengths.npy", lengths)
    (out / "meta.json").write_text(json.dumps(
        {"episodes": len(lengths), "transitions": int(lengths.sum()),
         "packed_bytes": PACKED_BYTES}, indent=1))


def iter_episodes(path: Path):
    """Yield episode dicts from a dataset directory, memory-mapped."""
    lengths = np.load(path / "lengths.npy")
    data = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    start = 0
    for n in lengths:
        n = int(n)
        yield {name: np.asarray(data[name][start:start + n]) for name in ARRAYS}
        start += n


def dataset_size(path: Path) -> tuple[int, int]:
    lengths = np.load(path / "lengths.npy")
    return len(lengths), int(lengths.sum())


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------
class _DummyPolicy:
    """What ``callbacks.setup`` expects from a live run, minus the network."""

    def __init__(self, worker_id: int, ctrl: np.ndarray, plane_set: str = "full"):
        self.worker_id = worker_id
        self.ctrl = ctrl
        self.plane_set = plane_set
        self.net = None

    def maybe_refresh(self) -> None:
        pass


def _teacher(noise: float, seed: int, shield: bool = True):
    """``bfs_expert``'s policy with epsilon-random noise, as a teacher callable.

    The noise is what makes the dataset usable: a deterministic expert only ever
    visits the states its own policy reaches, and a network trained on that has
    no idea what to do the first time it finds itself somewhere else.

    ``shield`` draws that noise from the *proven-survivable* actions only, and
    it is not a detail.  Measured on 8 000 rounds: unshielded epsilon = 0.1
    gives a median episode of 40 steps with 97.6 % of them ending in death,
    because escaping a bomb takes four consecutive correct moves and a 10 %
    chance of ruining each one compounds.  The dataset then teaches the network
    what dying looks like rather than what the expert does.  Shielding the noise
    keeps the state coverage and drops the suicides; the label is the expert's
    own action either way.
    """
    from agent_code.bfs_expert import callbacks as expert

    ctx = SimpleNamespace(logger=None, train=False)

    class _Logger:
        def debug(self, *a, **k):
            pass

        info = warning = error = debug

    ctx.logger = _Logger()
    expert.setup(ctx)
    rng = np.random.default_rng(seed)

    def teach(self, view, game_state) -> int:
        if rng.random() < noise:
            allowed = view.legal
            if shield:
                safe = allowed & view.survivable()
                if safe.any():
                    allowed = safe
            choices = np.flatnonzero(allowed)
            return int(choices[rng.integers(len(choices))])
        return ACTION_INDEX[expert.act(ctx, game_state)]

    return teach


def _collect_slice(job) -> str:
    """Play ``rounds`` rounds and write this worker's shard; return its path.

    The shard is written here rather than returned because a full dataset is
    2.5 M transitions of 615 bytes: returning it would materialise ~1.6 GB in
    the worker, the same again in the pickle, and the same again in the parent's
    concatenation -- about 5 GB of peak for a 1.5 GB result, next to a training
    run holding a replay buffer of its own.
    """
    worker_id, rounds, noise, seed, reward_cfg, scenario, shield, out = job
    quiet_logging()
    disable_file_logging()
    random.seed(seed * 977 + worker_id)
    torch.set_num_threads(1)

    from agent_code.dqn_agent import callbacks, train

    ctrl = np.zeros(64, dtype=np.float64)
    ctrl[train.CTRL.ALLOW_BOMB] = 1.0
    ctrl[train.CTRL.SHAPING] = 1.0
    callbacks.TRAINING_POLICY = _DummyPolicy(worker_id, ctrl)
    train.REWARD_CONFIG = reward_cfg
    train.TEACHER = _teacher(noise, seed * 131 + worker_id, shield)

    collected: list[dict] = []
    train.EPISODE_SINK = SimpleNamespace(put=collected.append)

    rng = np.random.default_rng(seed * 1000 + worker_id)
    weights = np.array([w for w, _ in COLLECT_MIX], dtype=float)
    weights /= weights.sum()
    worlds: dict[tuple, object] = {}
    for _ in range(rounds):
        opponents = tuple(COLLECT_MIX[int(rng.choice(len(COLLECT_MIX), p=weights))][1])
        world = worlds.get(opponents)
        if world is None:
            specs = [("dqn_agent", True)] + [(o, False) for o in opponents]
            world = make_world(specs, scenario=scenario,
                               log_dir=str(REPO / "logs" / f"bc-{os.getpid()}"),
                               continue_without_training=False)
            worlds[opponents] = world
        world.rng = np.random.default_rng(int(rng.integers(1 << 62)))
        world.new_round()
        while world.running:
            world.do_step()
    shard = Path(out) / f".shard-{worker_id:03d}"
    write_dataset(shard, collected)
    return str(shard)


def merge_shards(out: Path, shards: list[Path]) -> dict:
    """Concatenate worker shards into one dataset, one array at a time."""
    out.mkdir(parents=True, exist_ok=True)
    lengths = np.concatenate([np.load(s / "lengths.npy") for s in shards])
    np.save(out / "lengths.npy", lengths)
    for name in ARRAYS:
        parts = [np.load(s / f"{name}.npy", mmap_mode="r") for s in shards]
        total = sum(p.shape[0] for p in parts)
        shape = (total,) + parts[0].shape[1:]
        merged = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                           dtype=DTYPES[name], shape=shape)
        at = 0
        for part in parts:
            merged[at:at + part.shape[0]] = part
            at += part.shape[0]
        merged.flush()
        del merged, parts
    for shard in shards:
        shutil.rmtree(shard, ignore_errors=True)
    return {"episodes": len(lengths), "transitions": int(lengths.sum()),
            "packed_bytes": PACKED_BYTES}


def collect(out: Path, *, episodes: int, workers: int, noise: float, seed: int,
            scenario: str, reward_cfg: RewardConfig, shield: bool = True) -> dict:
    per = max(1, episodes // workers)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, per, noise, seed, reward_cfg, scenario, shield, str(out))
            for i in range(workers)]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        shards = [Path(p) for p in pool.map(_collect_slice, jobs)]
    meta = merge_shards(out, shards)
    meta["wall_s"] = time.perf_counter() - t0
    meta["noise"] = noise
    meta["shield"] = bool(shield)
    meta["scenario"] = scenario
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------
# supervised training
# --------------------------------------------------------------------------
def train_bc(data: Path, out: Path, *, net_cfg: QNetConfig, steps: int,
             batch_size: int, lr: float, n_step: int, gamma: float,
             margin: float, w_value: float, target_sync: int, augment: bool,
             device: str, seed: int, log_every: int = 500) -> dict:
    from .dqn_learner import decode_planes, transform_batch, unpack_masks

    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_episodes, n_trans = dataset_size(data)
    replay = PrioritisedReplay(n_trans, n_step=n_step, gamma=gamma, alpha=0.0, seed=seed)
    for ep in iter_episodes(data):
        replay.add_episode(ep, expert=True)

    net = QNet(net_cfg).to(dev)
    target = QNet(net_cfg).to(dev)
    target.load_state_dict(net.state_dict())
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1.5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 20)
    channels = torch.tensor(PLANE_SETS[net_cfg.plane_set], dtype=torch.long, device=dev)
    action_map = torch.tensor(np.asarray(ACTION_MAP, dtype=np.int64), device=dev)
    rng = np.random.default_rng(seed + 3)
    amp = dev.type == "cuda"

    history = []
    t0 = time.perf_counter()
    for it in range(1, steps + 1):
        batch = replay.sample(batch_size, 0.0)
        packed = torch.from_numpy(batch["packed"]).to(dev)
        step_i = torch.from_numpy(batch["step"].astype(np.int32)).to(dev)
        legal = torch.from_numpy(batch["legal"]).to(dev)
        action = torch.from_numpy(batch["action"].astype(np.int64)).to(dev)
        ret = torch.from_numpy(batch["ret"]).to(dev)
        discount = torch.from_numpy(batch["discount"]).to(dev)
        n_packed = torch.from_numpy(batch["next_packed"]).to(dev)
        n_step_i = torch.from_numpy(batch["next_step"].astype(np.int32)).to(dev)
        n_legal = torch.from_numpy(batch["next_legal"]).to(dev)

        x = decode_planes(packed, step_i, legal, channels)
        x_next = decode_planes(n_packed, n_step_i, n_legal, channels)
        mask = unpack_masks(legal)
        next_mask = unpack_masks(n_legal)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            q_next = target(x_next).float()
        q_next = q_next.masked_fill(~next_mask, -float("inf"))
        boot = torch.nan_to_num(q_next.max(dim=1).values, neginf=0.0)
        value_target = ret + discount * boot

        if augment:
            gs = torch.from_numpy(rng.integers(0, 8, size=x.shape[0])).to(dev)
            for g in range(1, 8):
                sel = gs == g
                if bool(sel.any()):
                    x[sel] = transform_batch(x[sel], g)
            perm = action_map[gs]
            action = perm.gather(1, action.unsqueeze(1)).squeeze(1)
            mask = torch.zeros_like(mask, dtype=torch.int8).scatter(
                1, perm, mask.to(torch.int8)).bool()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            q = net(x).float()
        q_sa = q.gather(1, action.unsqueeze(1)).squeeze(1)

        # Classification: the expert action must win by ``margin`` among the
        # legal ones.  Cross-entropy over masked Q values is the cheap version of
        # DQfD's large-margin loss and behaves better when several actions are
        # genuinely equivalent.
        logits = q.masked_fill(~mask, -1e4)
        ce = nn.functional.cross_entropy(logits / max(margin, 1e-6), action)
        value = nn.functional.smooth_l1_loss(q_sa, value_target)
        loss = ce + w_value * value

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % target_sync == 0:
            target.load_state_dict(net.state_dict())

        if it % log_every == 0 or it == steps:
            acc = float((logits.argmax(dim=1) == action).float().mean())
            row = {"step": it, "loss": float(loss.detach()), "ce": float(ce.detach()),
                   "value": float(value.detach()), "accuracy": acc,
                   "q_mean": float(q_sa.detach().mean()),
                   "grad_norm": float(grad_norm),
                   "lr": sched.get_last_lr()[0],
                   "wall_s": time.perf_counter() - t0}
            history.append(row)
            print(f"  {it:7d}/{steps}  loss {row['loss']:.4f}  ce {row['ce']:.4f}  "
                  f"value {row['value']:.4f}  acc {acc:.3f}  q {row['q_mean']:6.2f}  "
                  f"{row['wall_s']:.0f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "behaviour_cloning", "data": str(data), "episodes": n_episodes,
            "samples": n_trans, "steps": steps, "batch_size": batch_size,
            "n_step": n_step, "gamma": gamma, "accuracy": history[-1]["accuracy"],
            "wall_s": time.perf_counter() - t0, "use_mask": False}
    torch.save({"format": 1, "config": asdict(net_cfg),
                "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "meta": meta}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record expert episodes")
    c.add_argument("--out", required=True)
    c.add_argument("--episodes", type=int, default=2000)
    c.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 4))
    c.add_argument("--noise", type=float, default=0.1)
    c.add_argument("--unshielded-noise", action="store_true",
                   help="draw exploration noise from all legal actions, not only "
                        "the provably survivable ones (measured: 97 %% of episodes "
                        "then end in death)")
    c.add_argument("--scenario", default="classic")
    c.add_argument("--gamma", type=float, default=0.99)
    c.add_argument("--reward-scale", type=float, default=0.1)
    c.add_argument("--seed", type=int, default=1)

    t = sub.add_parser("train", help="behaviour-clone the network")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--plane-set", default="full")
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--n-step", type=int, default=3)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--margin", type=float, default=0.5,
                   help="softmax temperature of the classification loss")
    t.add_argument("--w-value", type=float, default=1.0)
    t.add_argument("--target-sync", type=int, default=1000)
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--log-every", type=int, default=500)

    args = p.parse_args(argv)

    if args.cmd == "collect":
        cfg = RewardConfig(gamma=args.gamma, reward_scale=args.reward_scale)
        meta = collect(Path(args.out), episodes=args.episodes, workers=args.workers,
                       noise=args.noise, seed=args.seed, scenario=args.scenario,
                       reward_cfg=cfg, shield=not args.unshielded_noise)
        print(json.dumps(meta, indent=1))
        return 0

    net_cfg = QNetConfig(plane_set=args.plane_set, channels=args.channels,
                         blocks=args.blocks)
    meta = train_bc(Path(args.data), Path(args.out), net_cfg=net_cfg, steps=args.steps,
                    batch_size=args.batch_size, lr=args.lr, n_step=args.n_step,
                    gamma=args.gamma, margin=args.margin, w_value=args.w_value,
                    target_sync=args.target_sync, augment=not args.no_augment,
                    device=args.device, seed=args.seed, log_every=args.log_every)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
