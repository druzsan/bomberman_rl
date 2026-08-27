"""Prioritised episodic replay for the deep Q-network.

Three properties drive the design:

* **States are stored bit-packed** (:func:`lib.encode.pack`, 615 B) rather than
  as float planes (22 kB).  Half a million transitions then cost ~330 MB
  instead of 11 GB, and unpacking a minibatch is a shift-and-mask on the GPU.
* **Storage is episodic and contiguous.**  Actors ship whole episodes, which are
  written into consecutive ring slots, so the n-step return of a transition is a
  slice of the buffer and ``s_{t+n}`` is simply the slot ``n`` further on.  The
  actor therefore never has to know ``n_step``: changing it is a learner-side
  config change.  Because slots are overwritten oldest-first and an episode's
  later slots are always *newer* than its earlier ones, a transition that is
  still intact always has its whole n-step window intact -- no validity
  bookkeeping is needed beyond "this slot has been written".
* **Prioritisation is proportional** (Schaul et al.), with a numpy sum tree
  descended vectorised over the batch: 20 array operations per sample call
  rather than a Python loop per element.

The buffer holds no torch tensors and never touches the GPU; the learner owns
the transfer.
"""

from __future__ import annotations

import numpy as np

from lib.encode import PACKED_BYTES


class SumTree:
    """Fixed-capacity sum tree over non-negative priorities.

    ``tree[1]`` is the total; leaves live at ``tree[size:size + size]``.
    """

    def __init__(self, capacity: int):
        self.size = 1 << (int(capacity) - 1).bit_length()
        self.tree = np.zeros(2 * self.size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Set the priority of several leaves at once."""
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        # Duplicate leaves would each read the same stale value and the deltas
        # would be applied twice; keep the last write per index.
        indices, keep = np.unique(indices[::-1], return_index=True)
        priorities = priorities[::-1][keep]

        leaves = indices + self.size
        delta = priorities - self.tree[leaves]
        self.tree[leaves] = priorities
        # Every leaf sits at the same depth, so the whole front reaches the root
        # in the same iteration and one scalar test ends the loop.
        nodes = leaves >> 1
        while nodes[0] >= 1:
            np.add.at(self.tree, nodes, delta)
            if nodes[0] == 1:
                break
            nodes = nodes >> 1

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Stratified proportional sampling: ``(leaf indices, probabilities)``."""
        total = self.total
        if total <= 0:
            raise ValueError("sampling from an empty sum tree")
        edges = (np.arange(n) + rng.random(n)) * (total / n)
        idx = np.ones(n, dtype=np.int64)
        while idx[0] < self.size:
            left = idx * 2
            go_right = edges > self.tree[left]
            edges = edges - np.where(go_right, self.tree[left], 0.0)
            idx = left + go_right
        leaves = idx - self.size
        return leaves, self.tree[idx] / total


class PrioritisedReplay:
    """Ring buffer of bit-packed transitions with proportional prioritisation."""

    def __init__(self, capacity: int, *, n_step: int = 3, gamma: float = 0.99,
                 alpha: float = 0.6, eps: float = 1e-3, seed: int = 0):
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self.packed = np.zeros((self.capacity, PACKED_BYTES), dtype=np.uint8)
        self.step = np.zeros(self.capacity, dtype=np.uint16)
        self.action = np.zeros(self.capacity, dtype=np.uint8)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.legal = np.zeros(self.capacity, dtype=np.uint8)
        #: Transitions remaining in the same episode *after* this one.
        self.remaining = np.zeros(self.capacity, dtype=np.uint16)
        #: Absolute write index, ``-1`` for a slot that was never written.
        self.pos = np.full(self.capacity, -1, dtype=np.int64)
        #: Whether the transition is expert data (kept for the DQfD margin loss).
        self.expert = np.zeros(self.capacity, dtype=bool)

        self.tree = SumTree(self.capacity)
        self.written = 0
        self.max_priority = 1.0
        self._discounts = self.gamma ** np.arange(self.n_step, dtype=np.float64)

    def __len__(self) -> int:
        return min(self.written, self.capacity)

    def add_episode(self, ep: dict, *, expert: bool = False) -> int:
        """Append one whole episode; returns the number of transitions added."""
        n = int(ep["reward"].shape[0])
        if n == 0:
            return 0
        if n > self.capacity:
            raise ValueError("episode longer than the replay capacity")
        start = self.written % self.capacity
        idx = (start + np.arange(n)) % self.capacity

        self.packed[idx] = ep["packed"]
        self.step[idx] = ep["step"]
        self.action[idx] = ep["action"]
        self.reward[idx] = ep["reward"]
        self.legal[idx] = ep["legal"]
        self.remaining[idx] = (n - 1 - np.arange(n)).astype(np.uint16)
        self.pos[idx] = self.written + np.arange(n)
        self.expert[idx] = expert

        # New transitions enter at the running maximum so every one of them is
        # replayed at least once before its priority is ever lowered.
        self.tree.update(idx, np.full(n, self.max_priority ** self.alpha))
        self.written += n
        return n

    def sample(self, batch_size: int, beta: float) -> dict:
        """Draw a prioritised minibatch of n-step transitions."""
        idx, prob = self.tree.sample(batch_size, self.rng)
        n = len(self)
        # A leaf with priority exactly zero (never written) can still be hit
        # when a stratum boundary lands on a prefix sum; fall back to a uniform
        # draw from the written region rather than reading an empty slot.
        empty = self.pos[idx] < 0
        if empty.any():
            idx = idx.copy()
            idx[empty] = self.rng.integers(0, n, size=int(empty.sum()))
            prob = prob.copy()
            prob[empty] = 1.0 / n
        weights = (n * np.maximum(prob, 1e-12)) ** (-beta)
        weights = weights / weights.max()

        remaining = self.remaining[idx].astype(np.int64)
        n_used = np.minimum(self.n_step, remaining + 1)
        ret = np.zeros(batch_size, dtype=np.float64)
        for k in range(self.n_step):
            slot = (idx + k) % self.capacity
            ret += np.where(k < n_used, self._discounts[k] * self.reward[slot], 0.0)

        terminal = remaining < self.n_step
        boot = (idx + self.n_step) % self.capacity
        return {
            "index": idx,
            "packed": self.packed[idx],
            "step": self.step[idx],
            "action": self.action[idx],
            "legal": self.legal[idx],
            "expert": self.expert[idx],
            "ret": ret.astype(np.float32),
            "discount": np.where(terminal, 0.0, self.gamma ** self.n_step
                                 ).astype(np.float32),
            "next_packed": self.packed[boot],
            "next_step": self.step[boot],
            "next_legal": self.legal[boot],
            "weight": weights.astype(np.float32),
        }

    def update_priorities(self, indices: np.ndarray, td_error: np.ndarray) -> None:
        p = np.abs(np.asarray(td_error, dtype=np.float64)) + self.eps
        self.max_priority = max(self.max_priority, float(p.max()))
        self.tree.update(indices, p ** self.alpha)
