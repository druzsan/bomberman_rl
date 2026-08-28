"""Tests for the multi-network artifact and the averaging that reads it.

An ensemble is easy to get silently wrong: load one member and average nothing,
load the same member twice and average it with itself, or break every artifact
written before the format changed.  Each of those looks exactly like "the
ensemble did not help".
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lib.board import N_ACTIONS
from lib.encode import n_channels
from lib.qnet import QNet, QNetConfig, load, load_all, save, save_ensemble


def net(seed: int) -> QNet:
    torch.manual_seed(seed)
    n = QNet(QNetConfig(plane_set="full", channels=8, blocks=1))
    n.eval()
    return n


def planes() -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(
        rng.random((1, n_channels("full"), 17, 17)).astype(np.float32))


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ens-"))

    def test_a_single_network_file_loads_as_an_ensemble_of_one(self):
        p = self.tmp / "one.pt"
        save(p, net(1), {"tta": True})
        nets, meta = load_all(p)
        self.assertEqual(len(nets), 1)
        self.assertTrue(meta["tta"])

    def test_an_ensemble_artifact_still_loads_under_the_old_reader(self):
        """`load` predates this format and must keep working on the new files."""
        p = self.tmp / "two.pt"
        a, b = net(1), net(2)
        save_ensemble(p, [a, b], {"tta": False})
        one, _ = load(p)
        x = planes()
        with torch.inference_mode():
            np.testing.assert_allclose(one(x).numpy(), a(x).numpy(), atol=1e-6)

    def test_members_come_back_distinct_and_in_order(self):
        p = self.tmp / "two.pt"
        a, b = net(1), net(2)
        save_ensemble(p, [a, b])
        nets, _ = load_all(p)
        x = planes()
        with torch.inference_mode():
            qa, qb = a(x).numpy(), b(x).numpy()
            got = [n(x).numpy() for n in nets]
        self.assertFalse(np.allclose(qa, qb), "the fixture members are identical")
        np.testing.assert_allclose(got[0], qa, atol=1e-6)
        np.testing.assert_allclose(got[1], qb, atol=1e-6)

    def test_mismatched_architectures_are_refused(self):
        wide = QNet(QNetConfig(plane_set="full", channels=16, blocks=1))
        with self.assertRaises(ValueError):
            save_ensemble(self.tmp / "bad.pt", [net(1), wide])

    def test_an_empty_ensemble_is_refused(self):
        with self.assertRaises(ValueError):
            save_ensemble(self.tmp / "bad.pt", [])


class AveragingTest(unittest.TestCase):
    """`callbacks.q_values` must average members, and compose that with TTA."""

    def context(self, nets, tta: bool):
        from types import SimpleNamespace

        return SimpleNamespace(nets=nets, net=nets[0], torch=torch, tta=tta,
                               plane_set="full")

    def state(self):
        rng = np.random.default_rng(3)
        field = np.zeros((17, 17), dtype=np.int64)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        field[rng.random((17, 17)) < 0.2] = 1
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        field[8, 8] = 0
        return {"round": 1, "step": 7, "field": field,
                "self": ("me", 0, True, (8, 8)), "others": [],
                "bombs": [], "coins": [(3, 3)], "user_input": None,
                "explosion_map": np.zeros((17, 17))}

    def q(self, nets, tta):
        import agent_code.dqn_agent.callbacks as cb

        gs = self.state()
        view = cb.StateView(gs)
        return cb.q_values(self.context(nets, tta), view, gs)

    def test_two_members_average_their_single_network_answers(self):
        a, b = net(1), net(2)
        got = self.q([a, b], tta=False)
        want = (self.q([a], tta=False) + self.q([b], tta=False)) / 2
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_the_average_composes_with_test_time_augmentation(self):
        a, b = net(1), net(2)
        got = self.q([a, b], tta=True)
        want = (self.q([a], tta=True) + self.q([b], tta=True)) / 2
        np.testing.assert_allclose(got, want, atol=1e-5)

    def test_one_member_is_bit_for_bit_the_old_behaviour(self):
        """Adding the ensemble must not perturb the shipped single-model path."""
        a = net(1)
        for tta in (False, True):
            got = self.q([a], tta=tta)
            self.assertEqual(got.shape, (N_ACTIONS,))
            self.assertTrue(np.isfinite(got).all())

    def test_averaging_actually_changes_the_answer(self):
        """A test that passes when the members are secretly the same is useless."""
        a, b = net(1), net(2)
        self.assertFalse(np.allclose(self.q([a], tta=True), self.q([b], tta=True)))
        both = self.q([a, b], tta=True)
        self.assertFalse(np.allclose(both, self.q([a], tta=True)))


if __name__ == "__main__":
    unittest.main()
