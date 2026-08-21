from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .augmentation import transform_action, transform_mask, transform_spatial


@dataclass(slots=True)
class Transition:
    spatial: NDArray[np.uint8]
    globals: NDArray[np.float32]
    action: int
    reward_components: dict[str, float]
    next_spatial: NDArray[np.uint8]
    next_globals: NDArray[np.float32]
    next_mask: NDArray[np.bool_]
    terminated: bool
    discount: float
    player_id: int
    round_id: int
    events: tuple[str, ...]


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.data: deque[Transition] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.data)

    def append(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> dict[str, NDArray]:
        indices = self.rng.choice(len(self.data), size=batch_size, replace=False)
        records = [self.data[int(index)] for index in indices]
        symmetries = self.rng.integers(0, 8, size=batch_size)
        return {
            "spatial": np.stack(
                [
                    transform_spatial(r.spatial, int(s))
                    for r, s in zip(records, symmetries)
                ]
            ).astype(np.float32),
            "globals": np.stack([r.globals for r in records]),
            "actions": np.asarray(
                [
                    transform_action(r.action, int(s))
                    for r, s in zip(records, symmetries)
                ],
                dtype=np.int64,
            ),
            "rewards": np.asarray(
                [sum(r.reward_components.values()) for r in records], dtype=np.float32
            ),
            "next_spatial": np.stack(
                [
                    transform_spatial(r.next_spatial, int(s))
                    for r, s in zip(records, symmetries)
                ]
            ).astype(np.float32),
            "next_globals": np.stack([r.next_globals for r in records]),
            "next_masks": np.stack(
                [
                    transform_mask(r.next_mask, int(s))
                    for r, s in zip(records, symmetries)
                ]
            ),
            "terminated": np.asarray([r.terminated for r in records], dtype=np.float32),
            "discounts": np.asarray([r.discount for r in records], dtype=np.float32),
        }


@dataclass(slots=True)
class PendingStep:
    spatial: NDArray[np.uint8]
    globals: NDArray[np.float32]
    action: int
    reward_components: dict[str, float]
    next_spatial: NDArray[np.uint8]
    next_globals: NDArray[np.float32]
    next_mask: NDArray[np.bool_]
    terminated: bool
    events: tuple[str, ...]


class NStepAssembler:
    def __init__(self, n_step: int, gamma: float, replay: ReplayBuffer) -> None:
        self.n_step = n_step
        self.gamma = gamma
        self.replay = replay
        self.queues: dict[int, deque[PendingStep]] = {}

    def add(self, player_id: int, round_id: int, step: PendingStep) -> None:
        queue = self.queues.setdefault(player_id, deque())
        queue.append(step)
        if len(queue) >= self.n_step:
            self._emit(player_id, round_id)
        if step.terminated:
            while queue:
                self._emit(player_id, round_id)

    def _emit(self, player_id: int, round_id: int) -> None:
        queue = self.queues[player_id]
        horizon = min(self.n_step, len(queue))
        components: dict[str, float] = {}
        last = queue[0]
        used = 0
        used_items: list[PendingStep] = []
        for index, item in enumerate(queue):
            if index >= horizon:
                break
            for name, value in item.reward_components.items():
                components[name] = (
                    components.get(name, 0.0) + (self.gamma**index) * value
                )
            last = item
            used = index + 1
            used_items.append(item)
            if item.terminated:
                break
        first = queue.popleft()
        self.replay.append(
            Transition(
                spatial=first.spatial,
                globals=first.globals,
                action=first.action,
                reward_components=components,
                next_spatial=last.next_spatial,
                next_globals=last.next_globals,
                next_mask=last.next_mask,
                terminated=last.terminated,
                discount=self.gamma**used,
                player_id=player_id,
                round_id=round_id,
                events=tuple(event for item in used_items for event in item.events),
            )
        )
