"""Compact uniform replay memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .state import EncodedState

TransitionKey = tuple[int, int]


@dataclass(slots=True)
class Transition:
    state: EncodedState
    action: int
    reward: float
    next_state: EncodedState | None
    done: bool
    key: TransitionKey


class ReplayMemory:
    """Fixed-capacity ring buffer storing observations as uint8 arrays."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self._items: list[Transition | None] = [None] * capacity
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Transition:
        if not 0 <= index < self._size:
            raise IndexError(index)
        transition = self._items[index]
        if transition is None:
            raise RuntimeError("Replay memory contains an unexpected empty slot")
        return transition

    def add(self, transition: Transition) -> int:
        index = self._next_index
        transition.state = np.ascontiguousarray(transition.state, dtype=np.uint8)
        if transition.next_state is not None:
            transition.next_state = np.ascontiguousarray(
                transition.next_state, dtype=np.uint8
            )
        self._items[index] = transition
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return index

    def patch_terminal(self, index: int, expected_key: TransitionKey) -> None:
        transition = self._items[index]
        if transition is None or transition.key != expected_key:
            raise RuntimeError(
                "Cannot patch terminal replay item: the ring slot no longer "
                f"contains transition {expected_key}"
            )
        transition.done = True

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from {self._size}"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        transitions = [self._items[int(index)] for index in indices]
        if any(transition is None for transition in transitions):
            raise RuntimeError("Replay memory contains an unexpected empty slot")
        return [transition for transition in transitions if transition is not None]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": self._items,
            "next_index": self._next_index,
            "size": self._size,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ReplayMemory:
        memory = cls(int(state["capacity"]))
        memory._items = state["items"]
        memory._next_index = int(state["next_index"])
        memory._size = int(state["size"])
        if len(memory._items) != memory.capacity:
            raise ValueError("Saved replay storage does not match its capacity")
        return memory
