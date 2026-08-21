"""Inference callbacks for the brute-force DQN agent."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .config import (
    ACTIONS,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    CHECKPOINT_VERSION,
    CONFIG,
    INPUT_SHAPE,
    epsilon_at_step,
)
from .model import DQN
from .persistence import load_torch_checkpoint
from .state import encode_state
from .types import Action, AgentContext, GameState


def _limit_torch_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits changing this after no parallel work has started.
        pass


def _validate_checkpoint(checkpoint: dict[str, object]) -> None:
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported DQN checkpoint version: "
            f"{checkpoint.get('version')!r} != {CHECKPOINT_VERSION}"
        )
    if tuple(checkpoint.get("actions", ())) != ACTIONS:
        raise ValueError("Checkpoint action order does not match this agent")
    if tuple(checkpoint.get("input_shape", ())) != INPUT_SHAPE:
        raise ValueError("Checkpoint input shape does not match this agent")


def setup(self: AgentContext) -> None:
    """Initialize the CPU model and optionally load a local checkpoint."""
    _limit_torch_threads()
    torch.manual_seed(CONFIG.seed)
    self.device = torch.device("cpu")
    self.rng = np.random.default_rng(CONFIG.seed)
    self.model = DQN(len(ACTIONS)).to(self.device)
    self.loaded_checkpoint = None
    self.environment_step = 0
    self.optimizer_step = 0
    self.completed_rounds = 0

    checkpoint_path = None
    if self.train and CONFIG.resume_training:
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"DQN_RESUME is enabled but {CHECKPOINT_PATH.name} is missing"
            )
        checkpoint_path = CHECKPOINT_PATH
    elif not self.train:
        if BEST_CHECKPOINT_PATH.is_file():
            checkpoint_path = BEST_CHECKPOINT_PATH
        elif CHECKPOINT_PATH.is_file():
            checkpoint_path = CHECKPOINT_PATH
        else:
            raise FileNotFoundError(
                "No trained DQN checkpoint found. Expected "
                f"{BEST_CHECKPOINT_PATH.name} or {CHECKPOINT_PATH.name}."
            )

    if checkpoint_path is not None:
        checkpoint = load_torch_checkpoint(checkpoint_path)
        _validate_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint["model_state"])
        self.loaded_checkpoint = checkpoint
        self.environment_step = int(checkpoint.get("environment_step", 0))
        self.optimizer_step = int(checkpoint.get("optimizer_step", 0))
        self.completed_rounds = int(checkpoint.get("completed_rounds", 0))
        self.logger.info("Loaded DQN checkpoint from %s", checkpoint_path.name)
    else:
        self.logger.info("Initialized a new DQN model from seed %d", CONFIG.seed)

    self.epsilon = (
        epsilon_at_step(self.environment_step)
        if self.train
        else CONFIG.evaluation_epsilon
    )
    self.model.train(self.train)
    self.round_action_times = []
    self.round_q_values = []


def act(self: AgentContext, game_state: GameState) -> Action:
    """Choose an unmasked epsilon-greedy action from the six-action space."""
    started = perf_counter()
    encoded_state = encode_state(game_state)
    if self.rng.random() < self.epsilon:
        action_index = int(self.rng.integers(len(ACTIONS)))
    else:
        observation = torch.from_numpy(encoded_state).unsqueeze(0)
        observation = observation.to(self.device)
        with torch.inference_mode():
            q_values = self.model(observation)
        action_index = int(q_values.argmax(dim=1).item())
        self.round_q_values.append(float(q_values.max().item()))

    self.round_action_times.append(perf_counter() - started)
    return ACTIONS[action_index]
