from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..policy import NetworkPolicy
from .environment import BombermanEnv


def evaluate_policy(
    policy: NetworkPolicy,
    scenario: str,
    players: int,
    rounds: int,
    seed: int,
    bombs_enabled: bool = True,
) -> dict[str, float | int]:
    totals: Counter[str] = Counter()
    scores: list[float] = []
    steps: list[int] = []
    was_training = policy.network.training
    policy.network.eval()
    try:
        for round_offset in range(rounds):
            environment = BombermanEnv(players, scenario, seed + round_offset)
            observations = environment.reset()
            while environment.running:
                player_ids = list(observations)
                selected_actions = policy.act_many(
                    [observations[player_id] for player_id in player_ids],
                    epsilon=0.0,
                    bombs_enabled=bombs_enabled,
                )
                actions = dict(zip(player_ids, selected_actions, strict=True))
                result = environment.step(actions)
                for events in result.events.values():
                    totals.update(events)
                observations = environment.observations() if environment.running else {}
            scores.append(sum(player.score for player in environment.players.values()))
            steps.append(environment.step_count)
    finally:
        policy.network.train(was_training)

    return {
        "mean_score": float(np.mean(scores)),
        "mean_coins": totals["COIN_COLLECTED"] / rounds,
        "mean_kills": totals["KILLED_OPPONENT"] / rounds,
        "mean_suicides": totals["KILLED_SELF"] / rounds,
        "mean_steps": float(np.mean(steps)),
    }


def evaluate_model(
    model_path: Path,
    scenario: str,
    players: int,
    rounds: int,
    seed: int,
    device: str,
    bombs_enabled: bool = True,
) -> dict[str, float | int | str]:
    policy = NetworkPolicy.load(model_path, device=device)
    result: dict[str, float | int | str] = {
        "model": str(model_path),
        "scenario": scenario,
        "players": players,
        "rounds": rounds,
        "seed": seed,
        "bombs_enabled": bombs_enabled,
    }
    result.update(
        evaluate_policy(policy, scenario, players, rounds, seed, bombs_enabled)
    )
    return result
