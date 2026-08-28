"""Conformance tests for the board-plane encoding and the replay buffer.

Three things can silently break the deep model and none of them shows up as an
exception:

* the packed representation and the direct one disagree, so the network is
  trained on states it never sees at inference time;
* the GPU decoder and the numpy decoder disagree, so the loss is computed on
  different planes than the actor recorded;
* the D4 augmentation transforms the planes but not the actions, which trains
  the network on systematically wrong labels and still converges -- to
  something useless.

Each of those gets a test here.
"""

from __future__ import annotations

import unittest

import numpy as np

from lib import encode, symmetry
from lib.board import ACTIONS, N_ACTIONS
from lib.features import analyse
from training.replay import PrioritisedReplay, SumTree

BOARD = encode.BOARD


def make_state(seed: int = 0, step: int = 17) -> dict:
    """A hand-built state with every plane non-empty."""
    rng = np.random.default_rng(seed)
    field = np.zeros((BOARD, BOARD), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(2, BOARD - 1, 2):
        for y in range(2, BOARD - 1, 2):
            field[x, y] = -1
    free = [(x, y) for x in range(1, BOARD - 1) for y in range(1, BOARD - 1)
            if field[x, y] == 0]
    crates = rng.choice(len(free), size=40, replace=False)
    for i in crates:
        field[free[i]] = 1

    open_tiles = [(x, y) for x, y in free if field[x, y] == 0]
    picks = rng.choice(len(open_tiles), size=9, replace=False)
    coins = [open_tiles[i] for i in picks[:3]]
    self_pos = open_tiles[picks[3]]
    others = [("o1", 2, True, open_tiles[picks[4]]),
              ("o2", 0, False, open_tiles[picks[5]]),
              ("o3", 1, True, open_tiles[picks[6]])]
    bombs = [(open_tiles[picks[7]], 3), (open_tiles[picks[8]], 0)]
    explosion = np.zeros((BOARD, BOARD))
    explosion[open_tiles[picks[2]]] = 1.0
    return {
        "round": 1, "step": step, "field": field,
        "self": ("me", 4, True, self_pos), "others": others, "bombs": bombs,
        "coins": coins, "user_input": None, "explosion_map": explosion,
    }


class EncodeTestCase(unittest.TestCase):
    def test_planes_match_the_state(self):
        state = make_state(1)
        p = encode.planes(state)
        self.assertEqual(p.shape, (encode.N_PLANES, BOARD, BOARD))
        idx = encode.PLANE_INDEX
        np.testing.assert_array_equal(p[idx["wall"]], (state["field"] == -1))
        np.testing.assert_array_equal(p[idx["crate"]], (state["field"] == 1))
        self.assertEqual(p[idx["coin"]].sum(), len(state["coins"]))
        self.assertEqual(p[idx["self"]].sum(), 1)
        self.assertEqual(p[idx["self"]][state["self"][3]], 1.0)
        self.assertEqual(p[idx["others"]].sum(), 3)
        self.assertEqual(p[idx["others_bomb"]].sum(), 2)
        self.assertEqual(p[idx["bomb_t3"]][state["bombs"][0][0]], 1.0)
        self.assertEqual(p[idx["bomb_t0"]][state["bombs"][1][0]], 1.0)
        self.assertTrue((p[idx["step"]] == state["step"] / encode.MAX_STEPS).all())
        self.assertTrue((p[idx["bombs_left"]] == 1.0).all())

    def test_lethal_planes_agree_with_the_danger_model(self):
        state = make_state(2)
        info = analyse(state)
        p = encode.planes(state, info.lethal)
        for tau in range(6):
            want = (info.lethal & np.uint8(1 << tau)) != 0
            np.testing.assert_array_equal(p[encode.PLANE_INDEX[f"lethal_tau{tau}"]],
                                          want.astype(np.float32))

    def test_pack_round_trip(self):
        states = [make_state(s, step=3 * s) for s in range(8)]
        packed = np.stack([encode.pack(s) for s in states])
        steps = np.array([s["step"] for s in states], dtype=np.uint16)
        left = np.ones(len(states), dtype=np.uint8)
        got = encode.unpack_batch(packed, steps, left)
        want = np.stack([encode.planes(s) for s in states])
        np.testing.assert_array_equal(got, want)

    def test_plane_set_ablation_is_a_channel_subset(self):
        state = make_state(3)
        full = encode.planes(state, plane_set="full")
        cut = encode.planes(state, plane_set="nodanger")
        self.assertEqual(cut.shape[0], encode.n_channels("nodanger"))
        self.assertEqual(cut.shape[0], encode.N_PLANES - 6)
        np.testing.assert_array_equal(cut, full[list(encode.PLANE_SETS["nodanger"])])

    def test_planes_are_d4_equivariant(self):
        """Transforming the state and transforming the planes must agree.

        This is the property the minibatch augmentation relies on; if it fails,
        the network is trained on labels that belong to a different board.
        """
        state = make_state(4)
        base = encode.planes(state)
        for g in range(symmetry.N_G):
            moved = symmetry.transform_state(state, g)
            got = encode.planes(moved)
            want = symmetry.transform_plane(base, g)
            np.testing.assert_array_equal(got, want, err_msg=f"group element {g}")

    def test_mask_round_trip(self):
        for value in range(1 << N_ACTIONS):
            bits = encode.mask_to_bits(value, N_ACTIONS)
            self.assertEqual(encode.bits_to_mask(bits), value)


class TorchDecoderTestCase(unittest.TestCase):
    """The GPU decoder must reproduce the numpy one bit for bit."""

    def test_decode_matches_numpy(self):
        import torch

        from training.dqn_learner import decode_planes, unpack_masks

        states = [make_state(s, step=7 * s) for s in range(6)]
        infos = [analyse(s) for s in states]
        packed = np.stack([encode.pack(s, i.lethal) for s, i in zip(states, infos)])
        steps = np.array([s["step"] for s in states], dtype=np.uint16)
        legal = np.array([encode.bits_to_mask(i.legal) for i in infos], dtype=np.uint8)

        channels = torch.tensor(encode.PLANE_SETS["full"], dtype=torch.long)
        got = decode_planes(torch.from_numpy(packed),
                            torch.from_numpy(steps.astype(np.int32)),
                            torch.from_numpy(legal), channels).numpy()
        want = encode.unpack_batch(packed, steps, (legal >> 5) & 1)
        np.testing.assert_array_equal(got, want)

        masks = unpack_masks(torch.from_numpy(legal)).numpy()
        for row, info in zip(masks, infos):
            np.testing.assert_array_equal(row, info.legal)

    def test_action_map_and_plane_transform_agree(self):
        """``transform_batch`` and ``ACTION_MAP`` must describe the same rotation.

        Concretely: moving UP in the transformed frame has to land on the tile
        that moving ``ACTION_MAP[g, UP]`` lands on in the original frame.
        """
        import torch

        from training.dqn_learner import transform_batch

        state = make_state(5)
        planes = torch.from_numpy(encode.planes(state)).unsqueeze(0)
        self_c = encode.PLANE_INDEX["self"]
        for g in range(symmetry.N_G):
            moved = transform_batch(planes, g)[0].numpy()
            want = encode.planes(symmetry.transform_state(state, g))
            np.testing.assert_array_equal(moved, want, err_msg=f"group element {g}")
            # the read-out selector still marks exactly one tile
            self.assertEqual(moved[self_c].sum(), 1.0)

        # The property the augmentation actually depends on: playing the mapped
        # action on the mapped board must land on the mapped tile.  Getting this
        # wrong trains the network on labels from a different board and still
        # converges -- to something useless.
        from lib.board import DELTAS

        pos = state["self"][3]
        shape = state["field"].shape
        for g in range(symmetry.N_G):
            moved_pos = symmetry.transform_coord(pos[0], pos[1], shape, g)
            for a, name in enumerate(ACTIONS[:4]):
                landed = (pos[0] + DELTAS[a][0], pos[1] + DELTAS[a][1])
                want = symmetry.transform_coord(landed[0], landed[1], shape, g)
                b = int(symmetry.ACTION_MAP[g, a])
                got = (moved_pos[0] + DELTAS[b][0], moved_pos[1] + DELTAS[b][1])
                self.assertEqual(got, want, f"{name} under group element {g}")


class TestTimeAugmentationTestCase(unittest.TestCase):
    """The D4 average must not depend on which image of the board it is given.

    ``q_values`` averages the network over all eight symmetry images and maps
    each frame's actions back to the world frame.  If the action mapping is
    inverted or composed the wrong way round the average still runs and still
    produces plausible numbers -- it simply mixes the value of moving left with
    the value of moving up.  Feeding it a transformed board and requiring the
    permuted answer catches exactly that.
    """

    def test_averaged_values_are_equivariant(self):
        import torch

        from agent_code.dqn_agent import callbacks
        from lib import symmetry
        from lib.qnet import QNet, QNetConfig

        torch.manual_seed(0)
        net = QNet(QNetConfig(channels=8, blocks=1)).eval()
        ctx = type("Ctx", (), {})()
        ctx.torch = torch
        ctx.net = net
        ctx.nets = [net]
        ctx.plane_set = "full"
        ctx.tta = True

        state = make_state(6)
        base = callbacks.q_values(ctx, callbacks.StateView(state), state)
        for g in range(symmetry.N_G):
            moved = symmetry.transform_state(state, g)
            got = callbacks.q_values(ctx, callbacks.StateView(moved), moved)
            want = base[symmetry.ACTION_MAP[symmetry.INVERSE[g]]]
            np.testing.assert_allclose(got, want, atol=1e-5,
                                       err_msg=f"group element {g}")

    def test_an_ensemble_average_is_equivariant_too(self):
        """The property has to survive the second averaging axis.

        Each member is D4-equivariant on its own, so their mean is as well --
        but only if the per-member action mapping happens *before* the mean and
        not after, which is the kind of thing that still produces plausible
        numbers when it is wrong.
        """
        import torch

        from agent_code.dqn_agent import callbacks
        from lib import symmetry
        from lib.qnet import QNet, QNetConfig

        nets = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            nets.append(QNet(QNetConfig(channels=8, blocks=1)).eval())
        ctx = type("Ctx", (), {})()
        ctx.torch, ctx.net, ctx.nets = torch, nets[0], nets
        ctx.plane_set, ctx.tta = "full", True

        state = make_state(6)
        base = callbacks.q_values(ctx, callbacks.StateView(state), state)
        for g in range(symmetry.N_G):
            moved = symmetry.transform_state(state, g)
            got = callbacks.q_values(ctx, callbacks.StateView(moved), moved)
            np.testing.assert_allclose(
                got, base[symmetry.ACTION_MAP[symmetry.INVERSE[g]]], atol=1e-5,
                err_msg=f"group element {g}")


class ReplayTestCase(unittest.TestCase):
    def episode(self, n: int, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        return {
            "packed": rng.integers(0, 256, size=(n, encode.PACKED_BYTES),
                                   dtype=np.uint8),
            "step": np.arange(n, dtype=np.uint16),
            "action": rng.integers(0, N_ACTIONS, size=n, dtype=np.uint8),
            "reward": rng.normal(size=n).astype(np.float32),
            "legal": np.full(n, 0b111111, dtype=np.uint8),
        }

    def test_sum_tree_totals_and_sampling(self):
        tree = SumTree(64)
        rng = np.random.default_rng(0)
        p = rng.random(64) + 0.1
        tree.update(np.arange(64), p)
        self.assertAlmostEqual(tree.total, p.sum(), places=6)
        tree.update(np.array([3, 3, 7]), np.array([0.5, 2.0, 1.0]))
        p[3], p[7] = 2.0, 1.0
        self.assertAlmostEqual(tree.total, p.sum(), places=6)
        leaves, prob = tree.sample(2000, rng)
        self.assertTrue((leaves >= 0).all() and (leaves < 64).all())
        # Empirical frequency must track the priority distribution.
        counts = np.bincount(leaves, minlength=64) / 2000
        np.testing.assert_allclose(counts, p / p.sum(), atol=0.02)
        np.testing.assert_allclose(prob.sum() > 0, True)

    def test_n_step_returns_and_terminal_handling(self):
        gamma, n = 0.9, 3
        buf = PrioritisedReplay(1024, n_step=n, gamma=gamma, alpha=0.0, seed=0)
        ep = self.episode(10, seed=1)
        buf.add_episode(ep)
        self.assertEqual(len(buf), 10)
        batch = buf.sample(512, beta=1.0)
        for i, ret, disc in zip(batch["index"], batch["ret"], batch["discount"]):
            i = int(i)
            remaining = 10 - 1 - i
            used = min(n, remaining + 1)
            want = sum(gamma ** k * ep["reward"][i + k] for k in range(used))
            self.assertAlmostEqual(float(ret), float(want), places=4)
            self.assertAlmostEqual(float(disc), 0.0 if remaining < n else gamma ** n,
                                   places=6)

    def test_episodes_do_not_bleed_into_each_other(self):
        buf = PrioritisedReplay(64, n_step=3, gamma=1.0, alpha=0.0, seed=0)
        first = self.episode(5, seed=2)
        first["reward"][:] = 1.0
        second = self.episode(5, seed=3)
        second["reward"][:] = 100.0
        buf.add_episode(first)
        buf.add_episode(second)
        batch = buf.sample(256, beta=1.0)
        for i, ret in zip(batch["index"], batch["ret"]):
            expected_max = 3.0 if int(i) < 5 else 300.0
            self.assertLessEqual(float(ret), expected_max + 1e-4)

    def test_priorities_change_the_sampling_distribution(self):
        buf = PrioritisedReplay(256, n_step=1, gamma=1.0, alpha=1.0, seed=0)
        buf.add_episode(self.episode(20, seed=4))
        buf.update_priorities(np.arange(20), np.full(20, 1e-6))
        buf.update_priorities(np.array([7]), np.array([100.0]))
        batch = buf.sample(500, beta=1.0)
        self.assertGreater((batch["index"] == 7).mean(), 0.9)

    def test_wraparound_keeps_the_buffer_consistent(self):
        buf = PrioritisedReplay(32, n_step=2, gamma=1.0, alpha=0.0, seed=0)
        for k in range(10):
            buf.add_episode(self.episode(7, seed=10 + k))
        self.assertEqual(len(buf), 32)
        batch = buf.sample(128, beta=1.0)
        self.assertTrue((buf.pos[batch["index"]] >= 0).all())
        self.assertEqual(batch["packed"].shape, (128, encode.PACKED_BYTES))


if __name__ == "__main__":
    unittest.main()
