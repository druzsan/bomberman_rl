"""D4 equivariance of the group action and of the feature extractor.

Folding the Q-table over the eight symmetries of the square is only valid if
the features transform consistently with the actions.  This test generates
random states, applies each group element to the whole ``game_state``, and
compares the feature blocks.

Direction *masks* must be exactly equivariant.  The scalar direction features
take the lowest set bit of their mask, so mirrored states may legitimately
report a different -- but equally shortest -- direction when several tie; the
test asserts the reported direction always lies inside the transformed mask.
"""

from __future__ import annotations

import unittest

import numpy as np

from lib import symmetry as sym
from lib.board import CRATE, FREE, WALL
from lib.features import FEATURE_SETS, FeatureSpec, analyse
from lib.symmetry import BITS4_MAP, DIR_MAP, N_G


def random_state(rng: np.random.Generator, size: int = 11):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = WALL
    for x in range(size):
        for y in range(size):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = WALL
    free = field == FREE
    field[free & (rng.random(field.shape) < rng.uniform(0.0, 0.6))] = CRATE

    free_xy = [tuple(int(v) for v in c) for c in np.argwhere(field == FREE)]
    rng.shuffle(free_xy)
    pos = free_xy.pop()
    others = [(f"o{i}", 0, bool(rng.integers(2)), free_xy.pop())
              for i in range(int(rng.integers(0, 4)))]
    bombs = [(free_xy.pop(), int(rng.integers(0, 5))) for _ in range(int(rng.integers(0, 4)))]
    coins = [free_xy.pop() for _ in range(int(rng.integers(0, 5)))]

    explosion_map = np.zeros(field.shape)
    for _ in range(int(rng.integers(0, 3))):
        c = free_xy.pop()
        explosion_map[c] = 1.0

    return {
        "round": 1, "step": int(rng.integers(1, 400)), "field": field,
        "self": ("me", 0, bool(rng.integers(2)), pos), "others": others,
        "bombs": bombs, "coins": coins, "user_input": None,
        "explosion_map": explosion_map,
    }


class GroupTest(unittest.TestCase):
    def test_group_axioms(self):
        for g in range(N_G):
            ginv = int(sym.INVERSE[g])
            self.assertEqual(int(sym.COMPOSE[ginv, g]), 0)
            for h in range(N_G):
                comp = int(sym.COMPOSE[g, h])
                for d in range(4):
                    self.assertEqual(int(DIR_MAP[comp, d]),
                                     int(DIR_MAP[g, DIR_MAP[h, d]]))

    def test_plane_and_coord_agree(self):
        rng = np.random.default_rng(11)
        a = rng.integers(0, 1000, size=(9, 9))
        for g in range(N_G):
            b = sym.transform_plane(a, g)
            for x in range(9):
                for y in range(9):
                    tx, ty = sym.transform_coord(x, y, (9, 9), g)
                    self.assertEqual(int(b[tx, ty]), int(a[x, y]))

    def test_bit_mask_map(self):
        for g in range(N_G):
            for m in range(16):
                want = 0
                for d in range(4):
                    if m & (1 << d):
                        want |= 1 << int(DIR_MAP[g, d])
                self.assertEqual(int(BITS4_MAP[g, m]), want)


class FeatureEquivarianceTest(unittest.TestCase):
    SCALAR_BLOCKS = ("danger_now", "bomb_ready", "bomb_here_value", "opp_dist", "in_dead_end")

    def test_features_are_equivariant(self):
        rng = np.random.default_rng(5)
        for trial in range(120):
            state = random_state(rng)
            base = analyse(state)
            for g in range(N_G):
                got = analyse(sym.transform_state(state, g))
                msg = f"trial={trial} g={g}"
                for b in self.SCALAR_BLOCKS:
                    self.assertEqual(got.blocks[b], base.blocks[b], f"{b} {msg}")
                self.assertEqual(got.blocks["walkable"],
                                 int(BITS4_MAP[g, base.blocks["walkable"]]), f"walkable {msg}")
                self.assertEqual(got.target_mask, int(BITS4_MAP[g, base.target_mask]),
                                 f"target_mask {msg}")
                self.assertEqual(got.opp_mask, int(BITS4_MAP[g, base.opp_mask]),
                                 f"opp_mask {msg}")
                # escape_mask has a fifth bit for WAIT, which is fixed by D4.
                self.assertEqual(got.escape_mask & 0b1111,
                                 int(BITS4_MAP[g, base.escape_mask & 0b1111]),
                                 f"escape_mask {msg}")
                self.assertEqual(got.escape_mask >> 4, base.escape_mask >> 4, f"wait bit {msg}")
                self.assertEqual(got.obj_dist, base.obj_dist, f"obj_dist {msg}")
                self.assertEqual(got.opp_dist, base.opp_dist, f"opp_dist {msg}")
                self.assertEqual(got.escape_possible, base.escape_possible, f"escape {msg}")
                self._assert_in_mask(got.blocks["target_dir"], got.target_mask, 5, msg)
                self._assert_in_mask(got.blocks["opp_dir"], got.opp_mask, 4, msg)

    def _assert_in_mask(self, value, mask, sentinel, msg):
        if value < 4:
            self.assertTrue(mask & (1 << value), f"dir {value} not in mask {mask} {msg}")

    def test_canonical_form_is_stable_under_the_group(self):
        """Every image of a tuple must canonicalise to the same entry."""
        rng = np.random.default_rng(7)
        for name in FEATURE_SETS:
            spec = FeatureSpec(name)
            for _ in range(200):
                values = tuple(int(rng.integers(c)) for c in spec.cards)
                canon, g0 = spec.canonical(values)
                for g in range(N_G):
                    other = spec.transform(values, g)
                    canon2, _ = spec.canonical(other)
                    self.assertEqual(canon, canon2, f"{name} {values} g={g}")
                self.assertEqual(spec.transform(values, g0), canon)


if __name__ == "__main__":
    unittest.main()
