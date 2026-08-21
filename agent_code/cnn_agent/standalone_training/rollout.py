from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..policy import ACTIONS, NetworkPolicy
from ..state import action_mask, encode_state
from .config import TrainingConfig
from .environment import BombermanEnv
from .replay import NStepAssembler, PendingStep
from .rewards import reward_components


class RolloutCollector:
    def __init__(
        self,
        environment: BombermanEnv,
        policy: NetworkPolicy,
        assembler: NStepAssembler,
        config: TrainingConfig,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.assembler = assembler
        self.config = config
        self.environment_steps = 0

    def collect_round(self, epsilon: float) -> dict[str, float | int]:
        observations = self.environment.reset()
        event_counts: Counter[str] = Counter()
        shaped_return = 0.0
        component_totals: defaultdict[str, float] = defaultdict(float)
        transitions = 0

        while self.environment.running:
            player_ids = list(observations)
            selected_actions = self.policy.act_many(
                [observations[player_id] for player_id in player_ids],
                epsilon,
                self.config.bombs_enabled,
            )
            actions = dict(zip(player_ids, selected_actions, strict=True))
            result = self.environment.step(actions)
            next_observations = (
                self.environment.observations() if self.environment.running else {}
            )
            for player_id, observation in observations.items():
                spatial, globals_ = encode_state(observation)
                terminated = result.terminated[player_id]
                next_observation = next_observations.get(player_id, observation)
                next_spatial, next_globals = encode_state(next_observation)
                next_legal = action_mask(next_observation)
                if not self.config.bombs_enabled:
                    next_legal[ACTIONS.index("BOMB")] = False
                if terminated:
                    next_legal[:] = False
                    next_legal[4] = True
                components = reward_components(
                    result.events[player_id],
                    observation,
                    next_observation,
                    self.config.coin_progress_reward,
                    self.config.time_penalty,
                )
                self.assembler.add(
                    player_id,
                    self.environment.round,
                    PendingStep(
                        spatial=spatial.astype(np.uint8),
                        globals=globals_,
                        action=ACTIONS.index(actions[player_id]),
                        reward_components=components,
                        next_spatial=next_spatial.astype(np.uint8),
                        next_globals=next_globals,
                        next_mask=next_legal,
                        terminated=terminated,
                        events=tuple(result.events[player_id]),
                    ),
                )
                event_counts.update(result.events[player_id])
                shaped_return += sum(components.values())
                for name, value in components.items():
                    component_totals[name] += value
                transitions += 1
            observations = next_observations
            self.environment_steps += 1

        scores = [player.score for player in self.environment.players.values()]
        record: dict[str, float | int] = {
            "steps": self.environment.step_count,
            "transitions": transitions,
            "score": float(sum(scores)),
            "shaped_return": shaped_return,
            "coins": event_counts["COIN_COLLECTED"],
            "kills": event_counts["KILLED_OPPONENT"],
            "suicides": event_counts["KILLED_SELF"],
            "invalid": event_counts["INVALID_ACTION"],
        }
        record.update(
            {
                f"reward_{name.lower()}": value
                for name, value in component_totals.items()
            }
        )
        return record
