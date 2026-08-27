"""The GPU learner: double, dueling, n-step DQN with prioritised replay.

One process owns the online network, the target network, the optimiser and the
replay buffer; the actors only produce episodes.  Everything model-specific
lives here so that :mod:`training.dqn_driver` is pure orchestration.

The pieces, and why each is in:

* **Double Q-learning** -- the ``argmax`` comes from the online network and the
  value from the target network, which removes the systematic overestimation
  that a single ``max`` produces.  In this game overestimation is not academic:
  the actions that are never taken are the fatal ones, so an optimistic ``max``
  over them poisons every state near a bomb.
* **Legal-action masking in the target ``max``**, not only in the behaviour
  policy.  Bootstrapping through an action the engine would reject is pure
  noise, and it is the half people forget.
* **n-step returns** from the episodic buffer (``n = 3``), which moves reward
  information back three times faster along the escape sequences that dominate
  this task.
* **D4 augmentation** -- each sampled transition is mapped through a random one
  of the eight symmetries of the square before the gradient step.  The target is
  computed in the world frame and is invariant under the group, so only the
  input planes and the action index have to be transformed.
* **DQfD margin loss** on expert transitions held in a *second, never
  overwritten* buffer, decayed to zero: the fine-tuned policy can improve on the
  behaviour-cloned teacher but cannot collapse below it early on.  The separate
  buffer is not a detail -- actors fill the main ring at ~3 400 transitions a
  second, so expert data written into it would be gone within the first
  150 seconds of a 20-minute run and the margin loss would spend most of its
  annealing schedule acting on nothing.

Plane decoding happens on the GPU: the buffer stores 615 bit-packed bytes per
state and this module turns a whole minibatch into ``(B, C, 17, 17)`` floats
with a shift and a mask, which costs less than the host-to-device copy would if
the planes were stored expanded.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lib.board import N_ACTIONS
from lib.encode import CELLS, MAX_STEPS, N_BINARY, N_PLANES, PLANE_SETS
from lib.qnet import QNet, QNetConfig
from lib.qnet import save as save_qnet
from lib.symmetry import ACTION_MAP

from .replay import PrioritisedReplay

_SHIFTS = torch.arange(7, -1, -1, dtype=torch.int16)

# Every minibatch has exactly the same shape, so letting cuDNN benchmark its
# algorithms once is free: measured 83 -> 94 gradient steps per second.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def decode_planes(packed: torch.Tensor, step: torch.Tensor, legal: torch.Tensor,
                  channels: torch.Tensor) -> torch.Tensor:
    """``(B, PACKED_BYTES)`` uint8 -> ``(B, C, 17, 17)`` float32, on device.

    Mirrors :func:`lib.encode.unpack_batch`; ``tests/test_encode.py`` asserts the
    two agree bit for bit.
    """
    b = packed.shape[0]
    shifts = _SHIFTS.to(packed.device)
    bits = (packed.to(torch.int16).unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(b, -1)[:, :N_BINARY * CELLS]
    out = torch.empty((b, N_PLANES, 17, 17), dtype=torch.float32, device=packed.device)
    out[:, :N_BINARY] = bits.reshape(b, N_BINARY, 17, 17).to(torch.float32)
    out[:, N_BINARY] = ((legal.to(torch.int16) >> 5) & 1).to(torch.float32)[:, None, None]
    out[:, N_BINARY + 1] = (step.to(torch.float32) / MAX_STEPS)[:, None, None]
    return out.index_select(1, channels)


def unpack_masks(legal: torch.Tensor) -> torch.Tensor:
    """``(B,)`` uint8 bit masks -> ``(B, 6)`` bool."""
    bits = torch.arange(N_ACTIONS, device=legal.device, dtype=torch.int16)
    return ((legal.to(torch.int16).unsqueeze(1) >> bits) & 1).bool()


def transform_batch(x: torch.Tensor, g: int) -> torch.Tensor:
    """Apply one D4 element to a batch of board planes (see :mod:`lib.symmetry`)."""
    flip, rot = divmod(int(g), 4)
    if flip:
        x = torch.flip(x, dims=(-2,))
    if rot:
        x = torch.rot90(x, k=-rot, dims=(-1, -2))
    return x


def _concat(a: dict, b: dict) -> dict:
    """Join two sampled minibatches; the online part always comes first.

    Keeping the two halves contiguous is what lets ``update`` write the priority
    updates back to the buffer each sample actually came from.
    """
    return {k: np.concatenate([a[k], b[k]]) for k in a}


class DQNLearner:
    """Owns the network, the optimiser and the replay buffer."""

    def __init__(self, cfg: dict, device: str = "cuda", seed: int = 0):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        net_cfg = QNetConfig(plane_set=cfg["plane_set"], channels=cfg["channels"],
                             blocks=cfg["blocks"])
        self.net = QNet(net_cfg).to(self.device)
        self.target = QNet(net_cfg).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=1.5e-4)
        self.replay = PrioritisedReplay(cfg["replay_capacity"], n_step=cfg["n_step"],
                                        gamma=cfg["gamma"], alpha=cfg["per_alpha"],
                                        seed=seed)
        self.channels = torch.tensor(PLANE_SETS[cfg["plane_set"]], dtype=torch.long,
                                     device=self.device)
        self.action_map = torch.tensor(np.asarray(ACTION_MAP, dtype=np.int64),
                                       device=self.device)
        self.amp = bool(cfg.get("amp", True)) and self.device.type == "cuda"
        self.expert: PrioritisedReplay | None = None
        self.grad_steps = 0
        #: Deferred priority write-back, see :meth:`_flush_priorities`.
        self._pending: tuple | None = None
        #: Materialise scalar metrics only this often; every ``float()`` on a
        #: GPU tensor is a synchronisation point.
        self.metric_every = int(cfg.get("metric_every", 25))
        self.rng = np.random.default_rng(seed + 7)
        self._last = {}

    # -- data -------------------------------------------------------------
    def add_episode(self, ep: dict, *, expert: bool = False) -> int:
        return self.replay.add_episode(ep, expert=expert)

    def load_expert(self, episodes, capacity: int) -> int:
        """Fill the demonstration buffer, which the ring never overwrites."""
        self.expert = PrioritisedReplay(capacity, n_step=self.cfg["n_step"],
                                        gamma=self.cfg["gamma"],
                                        alpha=self.cfg["per_alpha"],
                                        seed=self.cfg["seed"] + 11)
        added = 0
        for ep in episodes:
            if added + len(ep["reward"]) > capacity:
                break
            added += self.expert.add_episode(ep, expert=True)
        return added

    def expert_share(self, frac: float) -> float:
        """Fraction of each minibatch drawn from the demonstrations."""
        if self.expert is None or len(self.expert) == 0:
            return 0.0
        end = self.cfg.get("expert_anneal_frac", 0.0)
        share = float(self.cfg.get("expert_share", 0.25))
        if end <= 0:
            return share
        return share * max(0.0, 1.0 - frac / end)

    def ready(self) -> bool:
        return len(self.replay) >= self.cfg["learn_start"]

    # -- schedules --------------------------------------------------------
    def lr_at(self, frac: float) -> float:
        lr0, lr1 = self.cfg["lr"], self.cfg["lr_end"]
        cos = 0.5 * (1.0 + np.cos(np.pi * min(max(frac, 0.0), 1.0)))
        return float(lr1 + (lr0 - lr1) * cos)

    def beta_at(self, frac: float) -> float:
        b0, b1 = self.cfg["per_beta"], 1.0
        return float(b0 + (b1 - b0) * min(max(frac, 0.0), 1.0))

    def margin_weight(self, frac: float) -> float:
        end = self.cfg.get("margin_anneal_frac", 0.0)
        if end <= 0:
            return 0.0
        return float(self.cfg["margin_weight"] * max(0.0, 1.0 - frac / end))

    # -- one gradient step ------------------------------------------------
    def update(self, frac: float) -> dict:
        if not self.ready():
            return {}
        cfg = self.cfg
        beta = self.beta_at(frac)
        share = self.expert_share(frac)
        n_expert = round(cfg["batch_size"] * share)
        n_online = cfg["batch_size"] - n_expert
        batch = self.replay.sample(n_online, beta)
        if n_expert:
            batch = _concat(batch, self.expert.sample(n_expert, beta))
        dev = self.device

        packed = torch.from_numpy(batch["packed"]).to(dev, non_blocking=True)
        step = torch.from_numpy(batch["step"].astype(np.int32)).to(dev)
        legal = torch.from_numpy(batch["legal"]).to(dev)
        action = torch.from_numpy(batch["action"].astype(np.int64)).to(dev)
        ret = torch.from_numpy(batch["ret"]).to(dev)
        discount = torch.from_numpy(batch["discount"]).to(dev)
        weight = torch.from_numpy(batch["weight"]).to(dev)
        expert = torch.from_numpy(batch["expert"]).to(dev)
        n_packed = torch.from_numpy(batch["next_packed"]).to(dev, non_blocking=True)
        n_step_i = torch.from_numpy(batch["next_step"].astype(np.int32)).to(dev)
        n_legal = torch.from_numpy(batch["next_legal"]).to(dev)

        x = decode_planes(packed, step, legal, self.channels)
        x_next = decode_planes(n_packed, n_step_i, n_legal, self.channels)
        next_mask = unpack_masks(n_legal)

        # The target is a scalar and the group acts on frames, not on values, so
        # it is computed once in the world frame and reused for the augmented
        # copy of (s, a).
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=self.amp):
            q_online_next = self.net(x_next).float()
            q_target_next = self.target(x_next).float()
        q_online_next = q_online_next.masked_fill(~next_mask, -float("inf"))
        best = q_online_next.argmax(dim=1, keepdim=True)
        boot = q_target_next.gather(1, best).squeeze(1)
        # A state with no legal action cannot occur (WAIT is always legal), but a
        # terminal bootstrap must not leak an inf if it ever did.
        boot = torch.nan_to_num(boot, nan=0.0, posinf=0.0, neginf=0.0)
        target = ret + discount * boot

        if cfg.get("augment", True):
            # The group assignment is drawn on the *host* and the per-group row
            # indices are built with numpy.  Selecting with a GPU boolean mask
            # instead would need ``bool(sel.any())`` -- seven full GPU syncs per
            # gradient step, which measured as the single largest cost in this
            # function.
            gs_host = self.rng.integers(0, 8, size=x.shape[0])
            for g in range(1, 8):
                rows = np.flatnonzero(gs_host == g)
                if rows.size:
                    sel = torch.from_numpy(rows).to(dev)
                    x[sel] = transform_batch(x[sel], g)
            gs = torch.from_numpy(gs_host).to(dev)
            action = self.action_map[gs, action]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            q_all = self.net(x).float()
        q_sa = q_all.gather(1, action.unsqueeze(1)).squeeze(1)

        td = q_sa - target
        loss_each = nn.functional.smooth_l1_loss(q_sa, target, reduction="none",
                                                 beta=cfg.get("huber_delta", 1.0))
        loss = (weight * loss_each).mean()

        margin_w = self.margin_weight(frac)
        margin_loss = None
        if margin_w > 0 and n_expert:
            margin = torch.full_like(q_all, float(cfg["margin"]))
            margin.scatter_(1, action.unsqueeze(1), 0.0)
            violation = (q_all + margin).max(dim=1).values - q_sa
            margin_loss = (violation * expert.float()).sum() / expert.float().sum()
            loss = loss + margin_w * margin_loss

        for group in self.opt.param_groups:
            group["lr"] = self.lr_at(frac)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                   cfg["grad_clip"])
        self.opt.step()
        self.grad_steps += 1

        if self.grad_steps % cfg["target_sync"] == 0:
            self.target.load_state_dict(self.net.state_dict())

        # Priorities are written back one step late, on purpose: reading ``td``
        # now would block the host until the GPU has finished this step, which
        # serialises the two completely.  By the next call the work is done and
        # the copy is free.  A single step of staleness in a priority is
        # irrelevant -- prioritisation is a heuristic over half a million
        # transitions.
        self._flush_priorities()
        self._pending = (batch["index"], n_online, td.detach().abs())

        if self.grad_steps % self.metric_every == 0:
            scalars = {
                "loss": loss.detach(), "q_mean": q_sa.detach().mean(),
                "q_max": q_all.detach().max(), "target_mean": target.mean(),
                "grad_norm": grad_norm,
                "margin_loss": margin_loss.detach() if margin_loss is not None
                else torch.zeros((), device=dev),
            }
            # One synchronisation for the whole metric block instead of six.
            values = {k: float(v) for k, v in scalars.items()}
            self._last = {
                **{f"train/{k}": v for k, v in values.items()},
                "train/td_error_abs_mean": float(td.detach().abs().mean()),
                "train/lr": self.lr_at(frac),
                "train/beta": self.beta_at(frac),
                "train/is_weight_mean": float(batch["weight"].mean()),
                "train/per_priority_mean": float(self.replay.tree.total
                                                 / max(len(self.replay), 1)),
                "train/margin_weight": margin_w,
                "train/replay_size": float(len(self.replay)),
                "train/expert_share": share,
                "train/grad_steps": float(self.grad_steps),
            }
        return self._last

    def _flush_priorities(self) -> None:
        if self._pending is None:
            return
        index, n_online, td = self._pending
        self._pending = None
        td_np = td.cpu().numpy()
        self.replay.update_priorities(index[:n_online], td_np[:n_online])
        if len(index) > n_online:
            self.expert.update_priorities(index[n_online:], td_np[n_online:])

    # -- artifacts --------------------------------------------------------
    def flat_weights(self) -> np.ndarray:
        with torch.no_grad():
            return torch.nn.utils.parameters_to_vector(
                self.net.parameters()).detach().cpu().numpy().astype(np.float32)

    def export_inference(self, path: Path, meta: dict | None = None) -> None:
        cpu = QNet(self.net.cfg)
        cpu.load_state_dict({k: v.detach().cpu() for k, v in self.net.state_dict().items()})
        save_qnet(path, cpu, meta or {})

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "grad_steps": self.grad_steps,
                "config": self.cfg, "net_config": asdict(self.net.cfg)}

    def load_state_dict(self, blob: dict) -> None:
        self.net.load_state_dict(blob["net"])
        self.target.load_state_dict(blob["target"])
        self.opt.load_state_dict(blob["opt"])
        self.grad_steps = int(blob.get("grad_steps", 0))

    def load_pretrained(self, path: Path) -> dict:
        """Initialise from a behaviour-cloning artifact (same architecture)."""
        blob = torch.load(path, map_location=self.device, weights_only=True)
        if blob["config"] != asdict(self.net.cfg):
            raise ValueError(f"pretrained network config {blob['config']} does not "
                             f"match this run's {asdict(self.net.cfg)}")
        self.net.load_state_dict(blob["state_dict"])
        self.target.load_state_dict(self.net.state_dict())
        return blob.get("meta", {})


def write_checkpoint(root: Path, progress: int, learner: DQNLearner, meta: dict,
                     *, with_state: bool = True) -> Path:
    """Atomically materialise one checkpoint directory (``dev/plan.md`` §9.3/9.4)."""
    import os
    import shutil
    import tempfile

    from .checkpoint import git_sha

    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{progress:010d}"
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    try:
        full = {"progress": progress, "git_sha": git_sha(), **meta}
        learner.export_inference(tmp / "model.pt", full)
        if with_state:
            torch.save(learner.state_dict(), tmp / "train_state.pt")
        (tmp / "meta.json").write_text(json.dumps(full, indent=1, default=float))
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final
