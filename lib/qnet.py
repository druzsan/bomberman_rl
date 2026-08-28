"""The deep Q-network: a dueling, fully-convolutional action-value function.

Shipped with the agent (``callbacks.setup`` builds it from the config stored in
``model.pt``), and imported unchanged by the learner, so the architecture can
never drift between training and inference.

Two design choices are worth stating because they are not the obvious ones:

* **No flatten, no dense trunk.** The trunk is convolutional throughout and the
  advantage head is a ``1x1`` convolution producing six values *at every tile*;
  the six that matter are read out at the agent's own tile.  A
  ``flatten -> linear`` head would have to learn separately what "a crate to my
  left" means at each of the 289 positions, which throws away the translation
  equivariance that makes the D4 augmentation and the small sample budget work.
  The read-out is done as ``(A * self_plane).sum(spatial)`` rather than an index
  gather so that the same code path serves a single state and a minibatch, and
  so that the D4 augmentation needs no separate coordinate transform.
* **Dueling.** In this game most of the value of a state is "am I about to die",
  which is action-independent; splitting it off from the advantage measurably
  stabilises the target.  ``Q = V + (A - mean(A))``.

The receptive field after ``blocks`` residual blocks is ``2 * (2 * blocks + 2)
+ 1`` tiles, i.e. 27 at the default 6 blocks -- larger than the 17-tile board
diameter, so every unit in the read-out sees the whole arena.

``torch`` is imported at module import time, so nothing outside the deep agent
may import this module: the other agents vendor ``lib/`` too and must stay
torch-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .board import N_ACTIONS
from .encode import PLANE_INDEX, PLANE_SETS, n_channels


@dataclass(frozen=True)
class QNetConfig:
    """Everything needed to rebuild the network from a checkpoint."""

    plane_set: str = "full"
    channels: int = 64
    blocks: int = 6
    value_hidden: int = 64
    n_actions: int = N_ACTIONS

    @property
    def in_channels(self) -> int:
        return n_channels(self.plane_set)

    @property
    def self_channel(self) -> int:
        """Index of the ``self`` plane *within the selected channel subset*."""
        return PLANE_SETS[self.plane_set].index(PLANE_INDEX["self"])


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        return self.relu(x + self.conv2(y))


class QNet(nn.Module):
    """``(B, C, 17, 17)`` board planes -> ``(B, 6)`` action values."""

    def __init__(self, cfg: QNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or QNetConfig()
        c = self.cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(self.cfg.in_channels, c, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[ResBlock(c) for _ in range(self.cfg.blocks)])
        self.adv = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(c, self.cfg.n_actions, 1))
        self.val = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True))
        self.val_head = nn.Sequential(nn.Linear(c, self.cfg.value_hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.cfg.value_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The self plane doubles as the read-out selector, so no positions have
        # to be carried alongside the planes through the replay buffer.
        selector = x[:, self.cfg.self_channel:self.cfg.self_channel + 1]
        h = self.trunk(self.stem(x))
        a = (self.adv(h) * selector).sum(dim=(2, 3))
        v = self.val_head(self.val(h).mean(dim=(2, 3)))
        return v + a - a.mean(dim=1, keepdim=True)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save(path: Path | str, net: QNet, meta: dict | None = None) -> None:
    """Write the inference artifact: config + weights, never optimiser state."""
    torch.save({"format": 1, "config": asdict(net.cfg),
                "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "meta": meta or {}}, path)


def save_ensemble(path: Path | str, nets: list[QNet], meta: dict | None = None,
                  ) -> None:
    """Write several networks into one artifact.

    ``state_dict`` still holds the first member, so an artifact written this way
    loads correctly under :func:`load` and under any reader that predates this
    function -- it simply sees one network instead of several.  The extra
    members live in ``members``, and every one of them must share the first's
    :class:`QNetConfig`: an ensemble whose members disagree about the
    architecture is a bug, not a configuration.
    """
    if not nets:
        raise ValueError("an ensemble needs at least one member")
    for i, net in enumerate(nets[1:], 1):
        if asdict(net.cfg) != asdict(nets[0].cfg):
            raise ValueError(f"member {i} has a different QNetConfig")
    torch.save({"format": 2, "config": asdict(nets[0].cfg),
                "state_dict": {k: v.cpu() for k, v in nets[0].state_dict().items()},
                "members": [{k: v.cpu() for k, v in n.state_dict().items()}
                            for n in nets],
                "meta": meta or {}}, path)


def load(path: Path | str, device: str = "cpu") -> tuple[QNet, dict]:
    """Rebuild a network from an artifact written by :func:`save`.

    An ensemble artifact loads as its first member; use :func:`load_all` to get
    all of them.
    """
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = QNetConfig(**blob["config"])
    net = QNet(cfg)
    net.load_state_dict(blob["state_dict"])
    net.to(device)
    net.eval()
    return net, blob.get("meta", {})


def load_all(path: Path | str, device: str = "cpu") -> tuple[list[QNet], dict]:
    """Every member of an artifact; a single-network file yields a list of one."""
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = QNetConfig(**blob["config"])
    states = blob.get("members") or [blob["state_dict"]]
    nets = []
    for state in states:
        net = QNet(cfg)
        net.load_state_dict(state)
        net.to(device)
        net.eval()
        nets.append(net)
    return nets, blob.get("meta", {})
