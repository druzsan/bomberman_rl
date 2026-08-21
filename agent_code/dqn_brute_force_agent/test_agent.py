"""Focused unit tests for the self-contained DQN agent."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import events as e

from . import callbacks, train
from .config import ACTION_TO_INDEX, ACTIONS, CONFIG, INPUT_SHAPE, epsilon_at_step
from .model import DQN, bellman_targets
from .persistence import atomic_torch_save, load_torch_checkpoint
from .replay import ReplayMemory, Transition
from .state import encode_state


def game_state(*, round_number: int = 1, step: int = 1) -> dict:
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    field[2, 3] = -1
    field[4, 5] = 1
    explosion_map = np.zeros_like(field, dtype=float)
    explosion_map[9, 10] = 1
    return {
        "round": round_number,
        "step": step,
        "field": field,
        "self": ("learner", 0, True, (1, 1)),
        "others": [("opponent", 0, True, (15, 15))],
        "bombs": [((7, 8), 0)],
        "coins": [(5, 6)],
        "user_input": None,
        "explosion_map": explosion_map,
    }


def training_context() -> SimpleNamespace:
    context = SimpleNamespace(train=True, logger=logging.getLogger("dqn-test"))
    callbacks.setup(context)
    train.setup_training(context)
    return context


class StateEncodingTest(unittest.TestCase):
    def test_encoder_crops_border_and_maps_coordinates(self) -> None:
        state = game_state()
        original_field = state["field"].copy()
        encoded = encode_state(state)

        self.assertEqual(encoded.shape, INPUT_SHAPE)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(encoded[0, 2, 1], 1)  # engine field[2, 3]
        self.assertEqual(encoded[1, 4, 3], 1)  # engine field[4, 5]
        self.assertEqual(encoded[2, 5, 4], 1)  # coin (5, 6)
        self.assertEqual(encoded[3, 0, 0], 1)  # self (1, 1)
        self.assertEqual(encoded[4, 14, 14], 1)  # opponent (15, 15)
        self.assertEqual(encoded[5, 7, 6], 1)  # bomb (7, 8)
        self.assertEqual(encoded[6, 7, 6], 1)  # timer zero encoded as one
        self.assertEqual(encoded[7, 9, 8], 1)  # explosion at (9, 10)
        self.assertEqual(encoded[8, 0, 0], 1)
        self.assertEqual(encoded[9, 14, 14], 1)
        np.testing.assert_array_equal(state["field"], original_field)

    def test_encoder_rejects_changed_border(self) -> None:
        state = game_state()
        state["field"][0, 3] = 0
        with self.assertRaisesRegex(ValueError, "border"):
            encode_state(state)

    def test_encoder_rejects_entity_on_border(self) -> None:
        state = game_state()
        state["coins"] = [(0, 1)]
        with self.assertRaisesRegex(ValueError, "border"):
            encode_state(state)


class ModelAndScheduleTest(unittest.TestCase):
    def test_action_mapping_round_trip(self) -> None:
        self.assertEqual(len(ACTIONS), 6)
        for index, action in enumerate(ACTIONS):
            self.assertEqual(ACTION_TO_INDEX[action], index)

    def test_network_output(self) -> None:
        network = DQN(len(ACTIONS))
        observations = torch.zeros((3, *INPUT_SHAPE), dtype=torch.uint8)
        result = network(observations)
        self.assertEqual(tuple(result.shape), (3, 6))
        self.assertTrue(torch.isfinite(result).all())

    def test_bellman_targets_mask_terminal_states(self) -> None:
        rewards = torch.tensor([1.0, 5.0])
        next_values = torch.tensor([10.0, 10.0])
        dones = torch.tensor([False, True])
        targets = bellman_targets(rewards, next_values, dones, gamma=0.9)
        torch.testing.assert_close(targets, torch.tensor([10.0, 5.0]))

    def test_epsilon_schedule_endpoints(self) -> None:
        self.assertEqual(epsilon_at_step(0), CONFIG.epsilon_start)
        self.assertAlmostEqual(epsilon_at_step(CONFIG.epsilon_decay_steps // 2), 0.55)
        self.assertEqual(
            epsilon_at_step(CONFIG.epsilon_decay_steps * 2), CONFIG.epsilon_end
        )


class ReplayTest(unittest.TestCase):
    def _transition(self, step: int) -> Transition:
        state = np.zeros(INPUT_SHAPE, dtype=np.uint8)
        return Transition(state, 0, float(step), state.copy(), False, (1, step))

    def test_capacity_sampling_and_terminal_patch(self) -> None:
        replay = ReplayMemory(3)
        indices = [replay.add(self._transition(step)) for step in range(1, 5)]
        self.assertEqual(indices, [0, 1, 2, 0])
        self.assertEqual(len(replay), 3)
        sampled = replay.sample(3, np.random.default_rng(1))
        self.assertEqual({item.key for item in sampled}, {(1, 2), (1, 3), (1, 4)})
        replay.patch_terminal(0, (1, 4))
        self.assertTrue(replay[0].done)

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        model = DQN(len(ACTIONS))
        observation = torch.zeros((1, *INPUT_SHAPE), dtype=torch.uint8)
        expected = model(observation).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch_save({"model_state": model.state_dict(), "step": 12}, path)
            checkpoint = load_torch_checkpoint(path)
        restored = DQN(len(ACTIONS))
        restored.load_state_dict(checkpoint["model_state"])
        torch.testing.assert_close(restored(observation), expected)
        self.assertEqual(checkpoint["step"], 12)


class CallbackLifecycleTest(unittest.TestCase):
    def test_survivor_final_transition_is_not_duplicated(self) -> None:
        context = training_context()
        old_state = game_state(step=1)
        new_state = game_state(step=1)
        new_state["self"] = ("learner", 0, True, (1, 2))
        train.game_events_occurred(context, old_state, "DOWN", new_state, [])

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train, "METRICS_PATH", Path(directory) / "metrics.jsonl"),
        ):
            train.end_of_round(context, old_state, "DOWN", [e.SURVIVED_ROUND])

        self.assertEqual(len(context.replay), 1)
        self.assertTrue(context.replay[0].done)
        self.assertIsNotNone(context.replay[0].next_state)
        self.assertEqual(context.environment_step, 1)

    def test_fatal_transition_is_added_at_end_of_round(self) -> None:
        context = training_context()
        last_state = game_state(step=1)
        events = [e.KILLED_SELF, e.GOT_KILLED]

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(train, "METRICS_PATH", Path(directory) / "metrics.jsonl"),
        ):
            train.end_of_round(context, last_state, "BOMB", events)

        self.assertEqual(len(context.replay), 1)
        self.assertTrue(context.replay[0].done)
        self.assertIsNone(context.replay[0].next_state)
        self.assertEqual(context.environment_step, 1)
        self.assertEqual(context.round_steps, 0)  # Metrics reset for the next round.


if __name__ == "__main__":
    unittest.main()
