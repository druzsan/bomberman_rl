"""Central configuration for the brute-force DQN agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .types import Action

ACTIONS: tuple[Action, ...] = (
    "UP",
    "RIGHT",
    "DOWN",
    "LEFT",
    "WAIT",
    "BOMB",
)
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTIONS)}

BOARD_SIZE = 17
INTERIOR_SIZE = BOARD_SIZE - 2
INPUT_CHANNELS = 10
INPUT_SHAPE = (INPUT_CHANNELS, INTERIOR_SIZE, INTERIOR_SIZE)

BOMB_TIMER_SCALE = 5.0  # The encoded value is timer + 1, for timers 0...4.
EXPLOSION_SCALE = 2.0

AGENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("DQN_ARTIFACT_DIR", AGENT_DIR)).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoint.pt"
BEST_CHECKPOINT_PATH = ARTIFACT_DIR / "best_model.pt"
REPLAY_PATH = ARTIFACT_DIR / "replay.pkl"
METRICS_PATH = ARTIFACT_DIR / "training_metrics.jsonl"
CHECKPOINT_VERSION = 1


def _environment_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _environment_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Algorithm settings shared by inference and training."""

    seed: int = 7
    gamma: float = 0.99
    replay_capacity: int = 100_000
    replay_warmup: int = 10_000
    batch_size: int = 32
    learning_rate: float = 0.00025
    rmsprop_alpha: float = 0.95
    rmsprop_epsilon: float = 0.01
    gradient_norm_limit: float = 10.0
    train_frequency: int = 4
    target_copy_frequency: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1_000_000
    evaluation_epsilon: float = 0.05
    checkpoint_interval_rounds: int = 100
    resume_training: bool = False
    save_replay: bool = False


CONFIG = DQNConfig(
    replay_capacity=_environment_int("DQN_REPLAY_CAPACITY", 100_000),
    replay_warmup=_environment_int("DQN_REPLAY_WARMUP", 10_000),
    checkpoint_interval_rounds=_environment_int("DQN_CHECKPOINT_INTERVAL", 100),
    resume_training=_environment_bool("DQN_RESUME", False),
    save_replay=_environment_bool("DQN_SAVE_REPLAY", False),
)


def epsilon_at_step(environment_step: int) -> float:
    """Return linearly annealed training epsilon for an environment step."""
    if environment_step <= 0:
        return CONFIG.epsilon_start
    if environment_step >= CONFIG.epsilon_decay_steps:
        return CONFIG.epsilon_end
    progress = environment_step / CONFIG.epsilon_decay_steps
    return CONFIG.epsilon_start + progress * (CONFIG.epsilon_end - CONFIG.epsilon_start)
