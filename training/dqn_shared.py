"""Shared memory between the GPU learner and the CPU actors (Ape-X plumbing).

Two blocks, both tiny:

* ``ctrl`` -- the run's control vector (``agent_code.dqn_agent.train.CTRL``):
  epsilon, the shaping scale, the safety-mask level, the curriculum stage, the
  stop flag, a weight-version counter and one env-step counter per actor.  The
  driver writes, the actors read, so a schedule change takes effect without
  restarting anything.
* ``weights`` -- the online network's parameters as one flat float32 vector.
  The learner publishes every ``broadcast_every`` gradient steps and bumps the
  version; actors notice the bump and copy the vector into their own CPU copy
  of the network.

The agent package never imports ``multiprocessing``: the actor attaches the
blocks and injects a plain object into ``callbacks.TRAINING_POLICY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

CTRL_SIZE = 64


@dataclass(frozen=True)
class PolicySpec:
    """Everything an actor needs to attach; small enough to pickle."""

    ctrl_name: str
    weights_name: str
    n_params: int
    net_config: dict


class SharedPolicy:
    """Owner side (the driver/learner): creates the blocks and keeps them alive."""

    def __init__(self, n_params: int, net_config: dict):
        self._blocks = [
            shared_memory.SharedMemory(create=True, size=CTRL_SIZE * 8),
            shared_memory.SharedMemory(create=True, size=max(n_params, 1) * 4),
        ]
        self.spec = PolicySpec(self._blocks[0].name, self._blocks[1].name,
                               n_params, net_config)
        self.ctrl = np.ndarray((CTRL_SIZE,), dtype=np.float64, buffer=self._blocks[0].buf)
        self.weights = np.ndarray((n_params,), dtype=np.float32, buffer=self._blocks[1].buf)
        self.ctrl[:] = 0.0
        self.weights[:] = 0.0

    def publish(self, flat: np.ndarray) -> None:
        """Copy new parameters in and bump the version counter."""
        from agent_code.dqn_agent.train import CTRL

        self.weights[:] = flat
        self.ctrl[CTRL.WEIGHT_VERSION] += 1

    def close(self) -> None:
        del self.ctrl, self.weights
        for b in self._blocks:
            b.close()
            b.unlink()


class AttachedPolicy:
    """Actor side: a CPU network kept in sync with the learner's weights.

    This object *is* what ``callbacks.TRAINING_POLICY`` points at, so it defines
    the small interface the shipped agent uses during training: ``net``,
    ``ctrl``, ``worker_id`` and :meth:`maybe_refresh`.
    """

    def __init__(self, spec: PolicySpec, worker_id: int = 0):
        import torch

        from lib.qnet import QNet, QNetConfig

        torch.set_num_threads(1)
        self._torch = torch
        self._blocks = [shared_memory.SharedMemory(name=spec.ctrl_name),
                        shared_memory.SharedMemory(name=spec.weights_name)]
        self.ctrl = np.ndarray((CTRL_SIZE,), dtype=np.float64, buffer=self._blocks[0].buf)
        self.weights = np.ndarray((spec.n_params,), dtype=np.float32,
                                  buffer=self._blocks[1].buf)
        self.worker_id = worker_id
        self.plane_set = spec.net_config["plane_set"]
        self.net = QNet(QNetConfig(**spec.net_config))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.version = -1.0
        self.maybe_refresh()

    def maybe_refresh(self) -> None:
        """Pull the learner's parameters if a newer version was published."""
        from agent_code.dqn_agent.train import CTRL

        version = float(self.ctrl[CTRL.WEIGHT_VERSION])
        if version == self.version:
            return
        self.version = version
        # Copy rather than ``vector_to_parameters``: that rebinds ``p.data`` to a
        # view of the source, which would leave the network aliasing a buffer
        # the learner overwrites underneath it.
        flat = self._torch.from_numpy(self.weights.copy())
        offset = 0
        for p in self.net.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n

    def close(self) -> None:
        del self.ctrl, self.weights
        for b in self._blocks:
            b.close()
