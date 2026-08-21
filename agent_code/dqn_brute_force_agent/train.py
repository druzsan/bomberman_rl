"""Training callbacks for uniform-replay deep Q-learning."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional

import events as e

from .config import (
    ACTION_TO_INDEX,
    ACTIONS,
    CHECKPOINT_PATH,
    CHECKPOINT_VERSION,
    CONFIG,
    INPUT_SHAPE,
    METRICS_PATH,
    REPLAY_PATH,
    epsilon_at_step,
)
from .model import DQN, bellman_targets
from .persistence import (
    append_json_line,
    atomic_pickle_save,
    atomic_torch_save,
    load_pickle,
)
from .replay import ReplayMemory, Transition, TransitionKey
from .state import encode_state
from .types import Action, AgentContext, GameState


def setup_training(self: AgentContext) -> None:
    """Initialize replay, target network, optimizer, and round metrics."""
    self.target_model = DQN(len(ACTIONS)).to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()
    self.optimizer = torch.optim.RMSprop(
        self.model.parameters(),
        lr=CONFIG.learning_rate,
        alpha=CONFIG.rmsprop_alpha,
        eps=CONFIG.rmsprop_epsilon,
    )
    self.replay = ReplayMemory(CONFIG.replay_capacity)
    self.last_transition_key = None
    self.last_transition_index = None

    checkpoint = self.loaded_checkpoint
    if checkpoint is not None:
        if "target_model_state" in checkpoint:
            self.target_model.load_state_dict(checkpoint["target_model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "numpy_rng_state" in checkpoint:
            self.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if CONFIG.save_replay and REPLAY_PATH.is_file():
            replay_state = load_pickle(REPLAY_PATH)
            self.replay = ReplayMemory.from_state_dict(replay_state)
            if self.replay.capacity != CONFIG.replay_capacity:
                raise ValueError(
                    "Saved replay capacity differs from DQN_REPLAY_CAPACITY"
                )
            self.logger.info("Restored %d replay transitions", len(self.replay))

    _reset_round_metrics(self)


def reward_from_events(events: list[str]) -> float:
    """Return only score rewards supplied by the unmodified environment."""
    rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
    }
    return sum(rewards.get(event, 0.0) for event in events)


def _transition_key(game_state: GameState) -> TransitionKey:
    return game_state["round"], game_state["step"]


def _record_transition(
    self: AgentContext,
    old_game_state: GameState,
    action: Action,
    new_game_state: GameState | None,
    events: list[str],
    *,
    done: bool,
) -> None:
    key = _transition_key(old_game_state)
    if self.last_transition_key == key:
        raise RuntimeError(f"Transition {key} was presented more than once")

    reward = reward_from_events(events)
    transition = Transition(
        state=encode_state(old_game_state),
        action=ACTION_TO_INDEX[action],
        reward=reward,
        next_state=None if new_game_state is None else encode_state(new_game_state),
        done=done,
        key=key,
    )
    self.last_transition_index = self.replay.add(transition)
    self.last_transition_key = key
    self.environment_step += 1
    self.epsilon = epsilon_at_step(self.environment_step)
    self.round_return += reward
    self.round_steps += 1
    self.round_events.update(events)

    if (
        len(self.replay) >= max(CONFIG.replay_warmup, CONFIG.batch_size)
        and self.environment_step % CONFIG.train_frequency == 0
    ):
        _optimize(self)


def _optimize(self: AgentContext) -> None:
    transitions = self.replay.sample(CONFIG.batch_size, self.rng)
    states = torch.from_numpy(np.stack([item.state for item in transitions])).to(
        self.device
    )
    actions = torch.tensor(
        [item.action for item in transitions], dtype=torch.long, device=self.device
    )
    rewards = torch.tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=self.device
    )
    dones = torch.tensor(
        [item.done for item in transitions], dtype=torch.bool, device=self.device
    )

    predicted = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_values = torch.zeros(
        CONFIG.batch_size, dtype=torch.float32, device=self.device
    )
    nonterminal_indices = [
        index
        for index, item in enumerate(transitions)
        if not item.done and item.next_state is not None
    ]
    if any(not item.done and item.next_state is None for item in transitions):
        raise RuntimeError("A non-terminal replay transition has no successor state")
    if nonterminal_indices:
        next_states = torch.from_numpy(
            np.stack([transitions[index].next_state for index in nonterminal_indices])
        ).to(self.device)
        with torch.no_grad():
            next_values[nonterminal_indices] = (
                self.target_model(next_states).max(1).values
            )

    targets = bellman_targets(rewards, next_values, dones, CONFIG.gamma)
    loss = functional.smooth_l1_loss(predicted, targets)
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(), CONFIG.gradient_norm_limit
    )
    self.optimizer.step()

    self.optimizer_step += 1
    self.round_losses.append(float(loss.item()))
    self.round_gradient_norms.append(float(gradient_norm))
    if self.optimizer_step % CONFIG.target_copy_frequency == 0:
        self.target_model.load_state_dict(self.model.state_dict())


def game_events_occurred(
    self: AgentContext,
    old_game_state: GameState,
    self_action: Action,
    new_game_state: GameState,
    events: list[str],
) -> None:
    """Store one non-terminal transition after a surviving step."""
    _record_transition(
        self,
        old_game_state,
        self_action,
        new_game_state,
        events,
        done=False,
    )


def end_of_round(
    self: AgentContext,
    last_game_state: GameState | None,
    last_action: Action | None,
    events: list[str],
) -> None:
    """Record a fatal step or mark a survivor's recorded step terminal."""
    if last_game_state is not None and last_action is not None:
        key = _transition_key(last_game_state)
        if self.last_transition_key == key:
            if self.last_transition_index is None:
                raise RuntimeError("A transition key exists without a replay index")
            self.replay.patch_terminal(self.last_transition_index, key)
        else:
            # Dead agents do not receive game_events_occurred for the fatal step.
            _record_transition(
                self,
                last_game_state,
                last_action,
                None,
                events,
                done=True,
            )
    elif last_game_state is not None or last_action is not None:
        raise RuntimeError("Incomplete final transition supplied by the engine")

    self.completed_rounds += 1
    _write_round_metrics(self, last_game_state)
    if self.completed_rounds % CONFIG.checkpoint_interval_rounds == 0:
        _save_training_state(self)
    _reset_round_metrics(self)


def _checkpoint_payload(self: AgentContext) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "actions": ACTIONS,
        "input_shape": INPUT_SHAPE,
        "config": asdict(CONFIG),
        "model_state": self.model.state_dict(),
        "target_model_state": self.target_model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "environment_step": self.environment_step,
        "optimizer_step": self.optimizer_step,
        "completed_rounds": self.completed_rounds,
        "epsilon": self.epsilon,
        "numpy_rng_state": self.rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }


def _save_training_state(self: AgentContext) -> None:
    atomic_torch_save(_checkpoint_payload(self), CHECKPOINT_PATH)
    if CONFIG.save_replay:
        atomic_pickle_save(self.replay.state_dict(), REPLAY_PATH)
    self.logger.info(
        "Saved DQN checkpoint after %d rounds and %d transitions",
        self.completed_rounds,
        self.environment_step,
    )


def _reset_round_metrics(self: AgentContext) -> None:
    self.round_return = 0.0
    self.round_steps = 0
    self.round_events = Counter()
    self.round_losses = []
    self.round_gradient_norms = []
    self.round_action_times = []
    self.round_q_values = []
    self.round_started_at = perf_counter()


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _write_round_metrics(self: AgentContext, last_game_state: GameState | None) -> None:
    metrics = {
        "round": self.completed_rounds,
        "engine_round": None if last_game_state is None else last_game_state["round"],
        "environment_step": self.environment_step,
        "optimizer_step": self.optimizer_step,
        "epsilon": self.epsilon,
        "episode_return": self.round_return,
        "score": self.round_return,
        "lifetime_steps": self.round_steps,
        "coins": self.round_events[e.COIN_COLLECTED],
        "kills": self.round_events[e.KILLED_OPPONENT],
        "deaths": self.round_events[e.GOT_KILLED],
        "suicides": self.round_events[e.KILLED_SELF],
        "invalid_actions": self.round_events[e.INVALID_ACTION],
        "bombs": self.round_events[e.BOMB_DROPPED],
        "mean_loss": _mean(self.round_losses),
        "mean_q": _mean(self.round_q_values),
        "max_q": max(self.round_q_values, default=None),
        "mean_gradient_norm": _mean(self.round_gradient_norms),
        "mean_action_seconds": _mean(self.round_action_times),
        "max_action_seconds": max(self.round_action_times, default=None),
        "round_wall_seconds": perf_counter() - self.round_started_at,
    }
    append_json_line(METRICS_PATH, metrics)
