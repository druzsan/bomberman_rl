"""Shared-memory Q-table for Hogwild-style tabular learning.

For a tabular model the actor *is* the learner: there is no gradient step to
centralise, so the cheapest correct design is one shared table that every actor
reads lock-free and updates in place.  Concurrent unsynchronised updates are the
standard Hogwild setting and are benign here -- updates touch a single
``(state, action)`` cell and the table is extremely sparse in any short window.

The agent package never imports ``multiprocessing``: the worker attaches the
blocks and injects plain numpy views into the agent module.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

N_ACTIONS = 6
CTRL_SIZE = 64


@dataclass(frozen=True)
class TableSpec:
    """Everything a worker needs to attach, small enough to pickle."""

    q_name: str
    n_name: str
    ctrl_name: str
    n_states: int
    feature_set: str
    fold: bool


class SharedTables:
    """Owner side: creates the blocks and keeps them alive."""

    def __init__(self, n_states: int, feature_set: str, fold: bool):
        q_bytes = n_states * N_ACTIONS * 4
        n_bytes = n_states * N_ACTIONS * 4
        self._blocks = [
            shared_memory.SharedMemory(create=True, size=q_bytes),
            shared_memory.SharedMemory(create=True, size=n_bytes),
            shared_memory.SharedMemory(create=True, size=CTRL_SIZE * 8),
        ]
        self.spec = TableSpec(self._blocks[0].name, self._blocks[1].name,
                              self._blocks[2].name, n_states, feature_set, fold)
        self.q = np.ndarray((n_states, N_ACTIONS), dtype=np.float32, buffer=self._blocks[0].buf)
        self.n = np.ndarray((n_states, N_ACTIONS), dtype=np.uint32, buffer=self._blocks[1].buf)
        self.ctrl = np.ndarray((CTRL_SIZE,), dtype=np.float64, buffer=self._blocks[2].buf)
        self.q[:] = 0.0
        self.n[:] = 0
        self.ctrl[:] = 0.0

    def close(self) -> None:
        del self.q, self.n, self.ctrl
        for b in self._blocks:
            b.close()
            b.unlink()


class AttachedTables:
    """Worker side: numpy views onto the owner's blocks."""

    def __init__(self, spec: TableSpec, worker_id: int = 0):
        self._blocks = [shared_memory.SharedMemory(name=spec.q_name),
                        shared_memory.SharedMemory(name=spec.n_name),
                        shared_memory.SharedMemory(name=spec.ctrl_name)]
        shape = (spec.n_states, N_ACTIONS)
        self.q = np.ndarray(shape, dtype=np.float32, buffer=self._blocks[0].buf)
        self.n = np.ndarray(shape, dtype=np.uint32, buffer=self._blocks[1].buf)
        self.ctrl = np.ndarray((CTRL_SIZE,), dtype=np.float64, buffer=self._blocks[2].buf)
        self.feature_set = spec.feature_set
        self.fold = spec.fold
        self.worker_id = worker_id

    def close(self) -> None:
        del self.q, self.n, self.ctrl
        for b in self._blocks:
            b.close()
