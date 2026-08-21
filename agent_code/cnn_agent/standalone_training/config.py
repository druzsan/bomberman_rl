from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class TrainingConfig:
    rounds: int = 1_000
    scenario: Literal["coin-heaven", "classic"] = "coin-heaven"
    players: int = 1
    seed: int = 0
    device: str = "cpu"
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    n_step: int = 5
    batch_size: int = 64
    replay_capacity: int = 100_000
    replay_warmup: int = 2_000
    train_every: int = 4
    target_update_every: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    checkpoint_every: int = 100
    evaluation_every: int = 2_500
    evaluation_rounds: int = 200
    evaluation_seed: int = 10_000
    bombs_enabled: bool = True
    coin_progress_reward: float = 0.0
    time_penalty: float = 0.0
    output_dir: Path = Path("agent_code/cnn_agent/runs/default")

    def validate(self) -> None:
        if self.rounds < 1 or not 1 <= self.players <= 4:
            raise ValueError("rounds must be positive and players must be in [1, 4]")
        if self.scenario == "classic" and self.players < 2:
            raise ValueError("classic training needs at least two players")
        if (
            self.n_step < 1
            or self.batch_size < 1
            or self.replay_capacity < self.batch_size
        ):
            raise ValueError("invalid replay or batch configuration")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.evaluation_every < 1 or self.evaluation_rounds < 1:
            raise ValueError("evaluation intervals and rounds must be positive")
        if self.coin_progress_reward < 0 or self.time_penalty < 0:
            raise ValueError("reward magnitudes must be non-negative")

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TrainingConfig:
        known = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in known}
        filtered["output_dir"] = Path(str(filtered["output_dir"]))
        return cls(**filtered)  # type: ignore[arg-type]
