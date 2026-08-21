from __future__ import annotations

from pathlib import Path

import torch

from ..policy import NetworkPolicy
from .config import TrainingConfig
from .environment import BombermanEnv
from .learner import DoubleDQNLearner
from .replay import ReplayBuffer


def save_checkpoint(
    path: Path,
    learner: DoubleDQNLearner,
    config: TrainingConfig,
    round_number: int,
    environment_steps: int,
    transitions_seen: int,
    policy: NetworkPolicy,
    environment: BombermanEnv,
    replay: ReplayBuffer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": learner.online.state_dict(),
            "target": learner.target.state_dict(),
            "optimizer": learner.optimizer.state_dict(),
            "round": round_number,
            "environment_steps": environment_steps,
            "transitions_seen": transitions_seen,
            "updates": learner.updates,
            "config": config.as_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "policy_rng_state": policy.rng.bit_generator.state,
            "environment_rng_state": environment.rng.bit_generator.state,
            "replay_rng_state": replay.rng.bit_generator.state,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    learner: DoubleDQNLearner,
    policy: NetworkPolicy,
    environment: BombermanEnv,
    replay: ReplayBuffer,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=learner.device, weights_only=True)
    learner.online.load_state_dict(checkpoint["model"])
    learner.target.load_state_dict(checkpoint["target"])
    learner.optimizer.load_state_dict(checkpoint["optimizer"])
    learner.updates = int(checkpoint.get("updates", 0))
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "policy_rng_state" in checkpoint:
        policy.rng.bit_generator.state = checkpoint["policy_rng_state"]
    if "environment_rng_state" in checkpoint:
        environment.rng.bit_generator.state = checkpoint["environment_rng_state"]
    if "replay_rng_state" in checkpoint:
        replay.rng.bit_generator.state = checkpoint["replay_rng_state"]
    return checkpoint


def promote_model(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint["model"]}, destination)
