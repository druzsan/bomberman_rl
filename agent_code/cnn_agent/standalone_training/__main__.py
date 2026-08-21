from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .checkpoint import promote_model
from .config import TrainingConfig
from .evaluation import evaluate_model
from .trainer import train as run_training

app = typer.Typer(help="Train and validate the standalone factorized-CNN agent.")


@app.command()
def train(
    rounds: Annotated[
        int, typer.Option(min=1, help="Number of complete training rounds.")
    ] = 1_000,
    scenario: Annotated[
        str, typer.Option(help="Training world: coin-heaven or classic.")
    ] = "coin-heaven",
    players: Annotated[
        int, typer.Option(min=1, max=4, help="Shared-policy players per world.")
    ] = 1,
    seed: Annotated[
        int,
        typer.Option(
            help="Seed for simulator, exploration, and network initialization."
        ),
    ] = 0,
    device: Annotated[
        str, typer.Option(help="PyTorch device, for example cpu or cuda.")
    ] = "cpu",
    learning_rate: Annotated[float, typer.Option(min=0.0)] = 2.5e-4,
    gamma: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.99,
    n_step: Annotated[int, typer.Option(min=1)] = 5,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    replay_capacity: Annotated[int, typer.Option(min=1)] = 100_000,
    replay_warmup: Annotated[int, typer.Option(min=0)] = 2_000,
    train_every: Annotated[
        int, typer.Option(min=1, help="One update per this many new transitions.")
    ] = 4,
    target_update_every: Annotated[int, typer.Option(min=1)] = 2_000,
    epsilon_start: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
    epsilon_end: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
    epsilon_decay_steps: Annotated[int, typer.Option(min=1)] = 100_000,
    checkpoint_every: Annotated[int, typer.Option(min=1)] = 100,
    evaluation_every: Annotated[
        int, typer.Option(min=1, help="Evaluate and archive every N training rounds.")
    ] = 2_500,
    evaluation_rounds: Annotated[
        int, typer.Option(min=1, help="Held-out rounds per automatic evaluation.")
    ] = 200,
    evaluation_seed: Annotated[
        int, typer.Option(help="First fixed seed used for automatic evaluation.")
    ] = 10_000,
    disable_bombs: Annotated[
        bool,
        typer.Option(
            "--disable-bombs",
            help="Mask BOMB in behavior, replay targets, and automatic evaluation.",
        ),
    ] = False,
    coin_progress_reward: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Reward magnitude for one BFS step toward the nearest coin.",
        ),
    ] = 0.0,
    time_penalty: Annotated[
        float,
        typer.Option(min=0.0, help="Positive magnitude subtracted every transition."),
    ] = 0.0,
    output_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Run artifacts directory.")
    ] = Path("agent_code/cnn_agent/runs/default"),
) -> None:
    """Run shared-policy Double DQN training."""
    if scenario not in {"coin-heaven", "classic"}:
        raise typer.BadParameter(
            "must be coin-heaven or classic", param_hint="--scenario"
        )
    config = TrainingConfig(
        rounds=rounds,
        scenario=scenario,  # type: ignore[arg-type]
        players=players,
        seed=seed,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
        n_step=n_step,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        replay_warmup=replay_warmup,
        train_every=train_every,
        target_update_every=target_update_every,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        checkpoint_every=checkpoint_every,
        evaluation_every=evaluation_every,
        evaluation_rounds=evaluation_rounds,
        evaluation_seed=evaluation_seed,
        bombs_enabled=not disable_bombs,
        coin_progress_reward=coin_progress_reward,
        time_penalty=time_penalty,
        output_dir=output_dir,
    )
    try:
        run_training(config)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def resume(
    run_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Existing run directory containing latest.pt.",
        ),
    ] = Path("agent_code/cnn_agent/runs/default"),
    additional_rounds: Annotated[
        int, typer.Option(min=1, help="Rounds to add to the checkpoint's round count.")
    ] = 1_000,
    device: Annotated[
        str, typer.Option(help="Override the saved PyTorch device.")
    ] = "cpu",
    batch_size: Annotated[
        int | None,
        typer.Option(min=1, help="Optionally override the saved training batch size."),
    ] = None,
    evaluation_every: Annotated[
        int | None,
        typer.Option(min=1, help="Optionally override the evaluation interval."),
    ] = None,
    evaluation_rounds: Annotated[
        int | None,
        typer.Option(min=1, help="Optionally override held-out rounds per evaluation."),
    ] = None,
    evaluation_seed: Annotated[
        int | None,
        typer.Option(help="Optionally override the first held-out evaluation seed."),
    ] = None,
) -> None:
    """Continue a run from its latest checkpoint."""
    checkpoint_path = run_dir / "latest.pt"
    if not checkpoint_path.is_file():
        raise typer.BadParameter(f"checkpoint not found: {checkpoint_path}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainingConfig.from_dict(checkpoint["config"])
    config.output_dir = run_dir
    config.device = device
    config.rounds = int(checkpoint["round"]) + additional_rounds
    if batch_size is not None:
        config.batch_size = batch_size
    if evaluation_every is not None:
        config.evaluation_every = evaluation_every
    if evaluation_rounds is not None:
        config.evaluation_rounds = evaluation_rounds
    if evaluation_seed is not None:
        config.evaluation_seed = evaluation_seed
    try:
        run_training(config, resume_from=str(checkpoint_path))
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def evaluate(
    model: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Candidate model.pt.")
    ] = Path("agent_code/cnn_agent/runs/default/model.pt"),
    scenario: Annotated[
        str, typer.Option(help="Evaluation world: coin-heaven or classic.")
    ] = "coin-heaven",
    players: Annotated[int, typer.Option(min=1, max=4)] = 1,
    rounds: Annotated[int, typer.Option(min=1)] = 100,
    seed: Annotated[
        int, typer.Option(help="First deterministic evaluation seed.")
    ] = 10_000,
    device: Annotated[str, typer.Option()] = "cpu",
    disable_bombs: Annotated[
        bool,
        typer.Option(
            "--disable-bombs", help="Mask BOMB during this curriculum evaluation."
        ),
    ] = False,
) -> None:
    """Evaluate a frozen model greedily without learning or exploration."""
    if scenario not in {"coin-heaven", "classic"}:
        raise typer.BadParameter(
            "must be coin-heaven or classic", param_hint="--scenario"
        )
    result = evaluate_model(
        model, scenario, players, rounds, seed, device, not disable_bombs
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def promote(
    model: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Validated run model.pt."),
    ] = Path("agent_code/cnn_agent/runs/default/model.pt"),
    destination: Annotated[
        Path, typer.Option(dir_okay=False, help="Tournament callback model path.")
    ] = Path("agent_code/cnn_agent/model.pt"),
) -> None:
    """Copy a validated candidate into the tournament callback package."""
    promote_model(model, destination)
    typer.echo(f"Promoted {model} to {destination}")


@app.command()
def smoke(seed: Annotated[int, typer.Option(help="Simulator seed.")] = 0) -> None:
    """Run one deterministic simulator round without optimization."""
    from ..model import QNetwork
    from ..policy import NetworkPolicy
    from .environment import BombermanEnv

    environment = BombermanEnv(players=1, scenario="coin-heaven", seed=seed)
    policy = NetworkPolicy(QNetwork(), seed=seed)
    observations = environment.reset()
    while environment.running:
        actions = {
            player_id: policy.act(state, epsilon=1.0)
            for player_id, state in observations.items()
        }
        environment.step(actions)
        observations = environment.observations() if environment.running else {}
    typer.echo(f"completed round in {environment.step_count} steps")


if __name__ == "__main__":
    app()
