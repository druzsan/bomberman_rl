from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..policy import NetworkPolicy
from .checkpoint import load_checkpoint, promote_model, save_checkpoint
from .config import TrainingConfig
from .environment import BombermanEnv
from .evaluation import evaluate_policy
from .learner import DoubleDQNLearner
from .metrics import MetricsWriter
from .replay import NStepAssembler, ReplayBuffer
from .rollout import RolloutCollector


def _transitions_from_metrics(config: TrainingConfig) -> int:
    metrics_path = config.output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return 0
    total = 0
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += int(json.loads(line).get("transitions", 0))
    return total


def _save_evaluated_snapshot(
    config: TrainingConfig,
    learner: DoubleDQNLearner,
    policy: NetworkPolicy,
    environment: BombermanEnv,
    replay: ReplayBuffer,
    collector: RolloutCollector,
    evaluations: MetricsWriter,
    round_number: int,
    transitions_seen: int,
    epsilon: float,
) -> None:
    result = evaluate_policy(
        policy,
        config.scenario,
        config.players,
        config.evaluation_rounds,
        config.evaluation_seed,
        config.bombs_enabled,
    )
    checkpoint_path = config.output_dir / "checkpoints" / f"round_{round_number:08d}.pt"
    model_path = config.output_dir / "models" / f"round_{round_number:08d}.pt"
    checkpoint_arguments = (
        learner,
        config,
        round_number,
        collector.environment_steps,
        transitions_seen,
        policy,
        environment,
        replay,
    )
    save_checkpoint(checkpoint_path, *checkpoint_arguments)
    save_checkpoint(config.output_dir / "latest.pt", *checkpoint_arguments)
    promote_model(checkpoint_path, model_path)
    promote_model(checkpoint_path, config.output_dir / "model.pt")
    result.update(
        {
            "training_round": round_number,
            "transitions_seen": transitions_seen,
            "updates": learner.updates,
            "epsilon": epsilon,
            "scenario": config.scenario,
            "players": config.players,
            "evaluation_rounds": config.evaluation_rounds,
            "evaluation_seed": config.evaluation_seed,
            "bombs_enabled": config.bombs_enabled,
            "coin_progress_reward": config.coin_progress_reward,
            "time_penalty": config.time_penalty,
            "checkpoint": str(checkpoint_path),
            "model": str(model_path),
        }
    )
    evaluations.write(result)
    print(f"Evaluation: {json.dumps(result, sort_keys=True)}", flush=True)


def train(config: TrainingConfig, resume_from: str | None = None) -> None:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    replay = ReplayBuffer(config.replay_capacity, config.seed)
    learner = DoubleDQNLearner(
        replay, config.learning_rate, config.batch_size, config.device
    )
    policy = NetworkPolicy(learner.online, config.device, config.seed + 1)
    environment = BombermanEnv(config.players, config.scenario, config.seed + 2)
    assembler = NStepAssembler(config.n_step, config.gamma, replay)
    collector = RolloutCollector(environment, policy, assembler, config)
    metrics = MetricsWriter(config.output_dir / "metrics.jsonl")
    evaluations = MetricsWriter(config.output_dir / "evaluations.jsonl")
    transitions_seen = 0
    start_round = 0
    last_checkpoint = config.output_dir / "latest.pt"
    if resume_from is not None:
        checkpoint = load_checkpoint(
            Path(resume_from),
            learner,
            policy,
            environment,
            replay,
        )
        start_round = int(checkpoint["round"])
        transitions_seen = int(
            checkpoint.get("transitions_seen", _transitions_from_metrics(config))
        )
        collector.environment_steps = int(checkpoint.get("environment_steps", 0))
        print(
            f"Resuming after round {start_round} with {transitions_seen} transitions; "
            "replay will warm up again.",
            flush=True,
        )

    last_completed_round = start_round
    try:
        for round_number in range(start_round + 1, config.rounds + 1):
            progress = min(1.0, transitions_seen / max(1, config.epsilon_decay_steps))
            epsilon = config.epsilon_start + progress * (
                config.epsilon_end - config.epsilon_start
            )
            record = collector.collect_round(epsilon)
            new_transitions = int(record["transitions"])
            transitions_seen += new_transitions
            losses: list[float] = []
            if len(replay) >= config.replay_warmup:
                updates = max(1, new_transitions // config.train_every)
                for _ in range(updates):
                    losses.append(learner.update())
                    if learner.updates % config.target_update_every == 0:
                        learner.sync_target()
            record.update(
                {
                    "round": round_number,
                    "epsilon": epsilon,
                    "replay_size": len(replay),
                    "transitions_seen": transitions_seen,
                    "updates": learner.updates,
                    "loss": float(np.mean(losses)) if losses else None,
                }
            )
            metrics.write(record)
            last_completed_round = round_number
            if round_number == start_round + 1 or round_number % 10 == 0:
                print(json.dumps(record, sort_keys=True), flush=True)
            evaluation_due = (
                round_number % config.evaluation_every == 0
                or round_number == config.rounds
            )
            if evaluation_due:
                _save_evaluated_snapshot(
                    config,
                    learner,
                    policy,
                    environment,
                    replay,
                    collector,
                    evaluations,
                    round_number,
                    transitions_seen,
                    epsilon,
                )
            elif round_number % config.checkpoint_every == 0:
                save_checkpoint(
                    last_checkpoint,
                    learner,
                    config,
                    round_number,
                    collector.environment_steps,
                    transitions_seen,
                    policy,
                    environment,
                    replay,
                )
    except KeyboardInterrupt:
        save_checkpoint(
            last_checkpoint,
            learner,
            config,
            last_completed_round,
            collector.environment_steps,
            transitions_seen,
            policy,
            environment,
            replay,
        )
        print(
            f"Interrupted safely; checkpoint saved after round {last_completed_round}."
        )
        return

    print(
        f"Training complete. Evaluated checkpoints: {config.output_dir / 'checkpoints'}"
    )
