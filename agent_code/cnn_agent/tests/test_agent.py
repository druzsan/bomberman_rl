from __future__ import annotations

import unittest

import numpy as np
import torch

from agent_code.cnn_agent.model import QNetwork
from agent_code.cnn_agent.policy import ACTIONS, NetworkPolicy
from agent_code.cnn_agent.standalone_training.augmentation import (
    transform_action,
    transform_mask,
    transform_spatial,
)
from agent_code.cnn_agent.standalone_training.environment import BombermanEnv
from agent_code.cnn_agent.standalone_training.rewards import reward_components
from agent_code.cnn_agent.state import (
    BOMB_TIMER_START,
    COINS,
    EXPLOSIONS,
    OPPONENT_CAN_BOMB,
    SELF,
    action_mask,
    encode_state,
)


def sample_state() -> dict:
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2, 3] = -1
    return {
        "round": 1,
        "step": 200,
        "field": field,
        "self": ("me", 0, True, (1, 15)),
        "others": [("them", 0, True, (15, 1))],
        "bombs": [((2, 15), 0), ((14, 1), 4)],
        "coins": [(3, 14)],
        "user_input": None,
        "explosion_map": np.pad(np.eye(15), 1),
    }


class EncoderTests(unittest.TestCase):
    def test_coordinates_border_and_timer_zero(self) -> None:
        planes, globals_ = encode_state(sample_state())
        self.assertEqual(planes.shape, (12, 15, 15))
        self.assertEqual(planes[SELF, 14, 0], 1)
        self.assertEqual(planes[COINS, 13, 2], 1)
        self.assertEqual(planes[OPPONENT_CAN_BOMB, 0, 14], 1)
        self.assertEqual(planes[BOMB_TIMER_START, 14, 1], 1)
        self.assertEqual(planes[BOMB_TIMER_START + 4, 0, 13], 1)
        self.assertEqual(planes[EXPLOSIONS, 0, 0], 1)
        np.testing.assert_allclose(globals_, [1.0, 0.5])

    def test_action_mask_blocks_border_bomb_and_opponent(self) -> None:
        state = sample_state()
        state["self"] = ("me", 0, True, (1, 1))
        state["others"] = [("them", 0, True, (2, 1))]
        state["bombs"] = [((1, 2), 3)]
        np.testing.assert_array_equal(
            action_mask(state), [False, False, False, False, True, True]
        )

    def test_batched_policy_returns_one_legal_action_per_state(self) -> None:
        states = [sample_state(), sample_state()]
        policy = NetworkPolicy(QNetwork(), seed=7)
        actions = policy.act_many(states, epsilon=0.0)
        self.assertEqual(len(actions), len(states))
        for state, action in zip(states, actions, strict=True):
            self.assertTrue(action_mask(state)[ACTIONS.index(action)])

    def test_curriculum_mask_removes_bomb_from_greedy_policy(self) -> None:
        network = QNetwork()
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.head[-1].bias[ACTIONS.index("BOMB")] = 10
        policy = NetworkPolicy(network, seed=7)
        self.assertEqual(policy.act(sample_state(), bombs_enabled=True), "BOMB")
        self.assertNotEqual(policy.act(sample_state(), bombs_enabled=False), "BOMB")


class RewardTests(unittest.TestCase):
    def test_coin_progress_is_symmetric_and_collection_has_no_jump(self) -> None:
        old_state = sample_state()
        old_state["self"] = ("me", 0, False, (1, 1))
        old_state["coins"] = [(3, 1)]
        old_state["bombs"] = []
        new_state = dict(old_state)
        new_state["self"] = ("me", 0, False, (2, 1))

        closer = reward_components(["MOVED_RIGHT"], old_state, new_state, 0.05, 0.001)
        farther = reward_components(["MOVED_LEFT"], new_state, old_state, 0.05, 0.001)
        collected = reward_components(
            ["COIN_COLLECTED"], old_state, new_state, 0.05, 0.001
        )
        self.assertAlmostEqual(closer["COIN_PROGRESS"], 0.05)
        self.assertAlmostEqual(farther["COIN_PROGRESS"], -0.05)
        self.assertEqual(closer["TIME_PENALTY"], -0.001)
        self.assertNotIn("COIN_PROGRESS", collected)


class SymmetryTests(unittest.TestCase):
    def test_action_and_plane_transforms_agree(self) -> None:
        for symmetry in range(8):
            for action, (row, col) in enumerate(((1, 2), (2, 3), (3, 2), (2, 1))):
                plane = np.zeros((1, 5, 5), dtype=np.uint8)
                plane[0, row, col] = 1
                transformed = transform_spatial(plane, symmetry)[0]
                center = np.asarray([2, 2])
                location = np.argwhere(transformed == 1)[0]
                delta = tuple(location - center)
                expected = ((-1, 0), (0, 1), (1, 0), (0, -1)).index(delta)
                self.assertEqual(transform_action(action, symmetry), expected)

    def test_mask_is_a_permutation(self) -> None:
        mask = np.asarray([True, False, True, False, True, False])
        for symmetry in range(8):
            transformed = transform_mask(mask, symmetry)
            self.assertEqual(int(transformed.sum()), int(mask.sum()))
            self.assertTrue(transformed[4])


class EnvironmentTests(unittest.TestCase):
    def test_same_seed_produces_same_initial_state(self) -> None:
        first = BombermanEnv(4, "classic", 17).reset()
        second = BombermanEnv(4, "classic", 17).reset()
        for player_id in first:
            np.testing.assert_array_equal(
                first[player_id]["field"], second[player_id]["field"]
            )
            self.assertEqual(first[player_id]["self"], second[player_id]["self"])
            self.assertEqual(first[player_id]["coins"], second[player_id]["coins"])

    def test_observations_exist_before_any_mutation(self) -> None:
        environment = BombermanEnv(4, "classic", 4)
        observations = environment.reset()
        self.assertTrue(all(state["step"] == 1 for state in observations.values()))
        actions = {player_id: ACTIONS[4] for player_id in observations}
        result = environment.step(actions)
        self.assertFalse(result.round_over)
        self.assertEqual(environment.step_count, 2)
        self.assertTrue(all(events == ["WAITED"] for events in result.events.values()))


if __name__ == "__main__":
    unittest.main()
