from __future__ import annotations

from collections import Counter, deque

import numpy as np

from ..types import GameState

REWARD_WEIGHTS = {
    "COIN_COLLECTED": 1.0,
    "KILLED_OPPONENT": 5.0,
    "INVALID_ACTION": -0.05,
    "BOMB_DROPPED": -0.02,
    "CRATE_DESTROYED": 0.10,
    "COIN_FOUND": 0.10,
    "GOT_KILLED": -2.0,
    "KILLED_SELF": -1.0,
}


def _nearest_coin_distance(game_state: GameState) -> int | None:
    coins = set(game_state["coins"])
    if not coins:
        return None
    field = np.asarray(game_state["field"])
    start = game_state["self"][3]
    blocked = {position for position, _ in game_state["bombs"]}
    blocked.update(player[3] for player in game_state["others"])
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        position, distance = queue.popleft()
        if position in coins:
            return distance
        x, y = position
        for neighbor in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if (
                neighbor not in visited
                and field[neighbor] == 0
                and neighbor not in blocked
            ):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def reward_components(
    events: list[str],
    old_state: GameState | None = None,
    new_state: GameState | None = None,
    coin_progress_reward: float = 0.0,
    time_penalty: float = 0.0,
) -> dict[str, float]:
    counts = Counter(events)
    components = {
        event: weight * counts[event]
        for event, weight in REWARD_WEIGHTS.items()
        if counts[event]
    }
    if time_penalty:
        components["TIME_PENALTY"] = -time_penalty
    if (
        coin_progress_reward
        and old_state is not None
        and new_state is not None
        and "COIN_COLLECTED" not in counts
    ):
        old_distance = _nearest_coin_distance(old_state)
        new_distance = _nearest_coin_distance(new_state)
        if old_distance is not None and new_distance is not None:
            progress = float(np.clip(old_distance - new_distance, -1, 1))
            if progress:
                components["COIN_PROGRESS"] = coin_progress_reward * progress
    return components


def total_reward(components: dict[str, float]) -> float:
    return float(sum(components.values()))
