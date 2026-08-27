"""Tests for the S5b forward search (:mod:`lib.search`).

The search is only worth shipping if it is *better* than the flat ``argmax``,
and the cheapest way to be worse is to be subtly wrong -- a discount applied at
the wrong depth, a reward counted twice, a leaf evaluated as though it were
legal.  So these tests use a **fixed, hand-written evaluator** rather than the
trained network: with the leaf values known, the backed-up value of every root
action is a number that can be written down in advance, and any disagreement is
a bug in the search rather than an opinion of a network.

The last class is the one that matters in a game: with a value function that is
merely *indifferent*, the search must still refuse an action that walks into a
blast, because it sees the death itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from lib.board import ACTIONS
from lib.search import (
    COIN_COLLECTED,
    KILLED_SELF,
    MOVED,
    WAITED,
    SearchConfig,
    legal_actions,
    search,
    step_reward,
)
from lib.sim import Sim, SimAgent


def empty_board() -> np.ndarray:
    a = np.zeros((17, 17), dtype=np.int8)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
    return a


def solo(x: int = 8, y: int = 8, **kw) -> Sim:
    """One agent on an open board, plus a crate far out of blast range.

    The crate is not decoration.  ``time_to_stop`` ends a round the moment one
    agent is left with no crates, no collectable coins and nothing in flight, so
    a genuinely empty board makes every action terminal on the first step and
    the search has nothing to search.  That is correct engine behaviour and it
    is not the case under test.
    """
    arena = empty_board()
    arena[15, 15] = 1
    return Sim(arena=arena, agents=[SimAgent(x, y, **kw)], active=[0])


def cfg(**kw) -> SearchConfig:
    """A config for arithmetic tests: the leaf cap must not interfere.

    ``max_leaves`` defaults to 48 for the tournament's latency budget, and an
    open 17x17 board blows past that at depth 3 -- which is the cap working, not
    a bug.  Tests that are checking the *backup* raise it out of the way and
    :class:`LeafCapTest` checks the cap on its own.
    """
    kw.setdefault("max_leaves", 100_000)
    # The potential correction is real and is tested on its own below; here it
    # would put a state-dependent term into arithmetic meant to be exact.
    kw.setdefault("use_potential", False)
    return SearchConfig(**kw)


def constant(value: float):
    """An evaluator with no opinion: every action of every leaf is worth `value`."""
    def evaluate(states):
        return np.full((len(states), len(ACTIONS)), value, dtype=np.float64)
    return evaluate


def by_position(table: dict[tuple[int, int], float], default: float = 0.0):
    """An evaluator that scores a leaf purely by where we stand in it."""
    def evaluate(states):
        out = np.zeros((len(states), len(ACTIONS)))
        for i, gs in enumerate(states):
            out[i] = table.get(gs["self"][3], default)
        return out
    return evaluate


class LegalActionTest(unittest.TestCase):
    def test_open_board(self):
        self.assertEqual(set(legal_actions(solo(), 0)),
                         {"UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"})

    def test_a_wall_removes_a_direction(self):
        self.assertNotIn("LEFT", legal_actions(solo(1, 8), 0))

    def test_no_bomb_when_one_is_already_out(self):
        self.assertNotIn("BOMB", legal_actions(solo(bombs_left=False), 0))

    def test_wait_is_always_available(self):
        s = solo(1, 1)
        s.arena[1, 2] = 1
        s.arena[2, 1] = 1
        self.assertEqual(set(legal_actions(s, 0)), {"WAIT", "BOMB"})


class BackupArithmeticTest(unittest.TestCase):
    """With a known evaluator the answer is arithmetic, not judgement.

    Every step now carries a small per-action cost (a move -0.02, a wait -0.05,
    priced as training priced them), so these check *differences* between root
    actions rather than absolute totals.  A difference is what an ``argmax``
    reads anyway, and it cancels the step costs the test is not about.
    """

    def test_moves_in_open_space_are_all_worth_the_same(self):
        r = search(solo(), 0, constant(2.0), cfg(depth=3, gamma=0.95))
        moves = [r.values[ACTIONS.index(a)] for a in ("UP", "DOWN", "LEFT", "RIGHT")]
        np.testing.assert_allclose(moves, moves[0])

    def test_waiting_costs_more_than_moving(self):
        r = search(solo(), 0, constant(2.0), cfg(depth=1, gamma=0.95))
        self.assertAlmostEqual(r.values[ACTIONS.index("UP")]
                               - r.values[ACTIONS.index("WAIT")],
                               (MOVED - WAITED) * 0.1)

    def test_the_leaf_is_discounted_by_the_depth_reached(self):
        """Isolated by differencing two evaluators: everything else cancels."""
        for depth in (1, 2, 3):
            conf = cfg(depth=depth, gamma=0.9)
            hot = search(solo(), 0, constant(1.0), conf).values
            cold = search(solo(), 0, constant(0.0), conf).values
            finite = np.isfinite(hot) & np.isfinite(cold)
            np.testing.assert_allclose(hot[finite] - cold[finite], 0.9 ** depth)

    def test_a_coin_on_the_path_is_counted_once_and_undiscounted_at_t0(self):
        s = solo()
        s.coins = [(9, 8, True)]
        conf = cfg(depth=1, gamma=0.5, reward_scale=0.1)
        r = search(s, 0, constant(0.0), conf)
        # LEFT and RIGHT are symmetric here, so their step costs cancel exactly.
        self.assertAlmostEqual(r.values[ACTIONS.index("RIGHT")]
                               - r.values[ACTIONS.index("LEFT")],
                               COIN_COLLECTED * 0.1)

    def test_a_coin_one_step_further_is_discounted_once(self):
        s = solo()
        s.coins = [(10, 8, True)]
        conf = cfg(depth=2, gamma=0.5, reward_scale=0.1)
        r = search(s, 0, constant(0.0), conf)
        self.assertAlmostEqual(r.values[ACTIONS.index("RIGHT")]
                               - r.values[ACTIONS.index("LEFT")],
                               0.5 * COIN_COLLECTED * 0.1)

    def test_illegal_actions_stay_at_minus_infinity(self):
        r = search(solo(1, 1), 0, constant(1.0), cfg(depth=2))
        self.assertEqual(r.values[ACTIONS.index("LEFT")], -np.inf)
        self.assertEqual(r.values[ACTIONS.index("UP")], -np.inf)
        self.assertTrue(np.isfinite(r.values[ACTIONS.index("RIGHT")]))

    def test_the_search_maximises_over_our_own_actions(self):
        """A reward reachable only by a specific three-step path must be found."""
        conf = cfg(depth=3, gamma=1.0)
        r = search(solo(), 0, by_position({(11, 8): 10.0}), conf)
        best = int(np.argmax(np.where(np.isfinite(r.values), r.values, -np.inf)))
        self.assertEqual(best, ACTIONS.index("RIGHT"))
        self.assertGreater(r.values[ACTIONS.index("RIGHT")]
                           - r.values[ACTIONS.index("LEFT")], 9.0)


class PotentialTest(unittest.TestCase):
    """The shaping term the backup has to carry, and why.

    The network's ``Q`` is the value under the *shaped* reward.  Backing up
    unshaped rewards to it leaves a ``gamma^depth * Phi(leaf)`` term that does
    not cancel between siblings -- measured at a mean of 0.100 across the
    siblings of one position, against a decision margin of 0.05.
    """

    def board_with_a_coin(self) -> Sim:
        arena = empty_board()
        arena[15, 15] = 1
        s = Sim(arena=arena, agents=[SimAgent(8, 8)], active=[0])
        s.coins = [(12, 8, True)]
        return s

    def test_it_changes_the_answer_on_a_position_where_phi_differs(self):
        s = self.board_with_a_coin()
        flat = search(s, 0, constant(0.0),
                      SearchConfig(depth=3, max_leaves=10_000, use_potential=False))
        shaped = search(s, 0, constant(0.0),
                        SearchConfig(depth=3, max_leaves=10_000, use_potential=True))
        # The coin is four steps away, out of reach of a depth-3 tree, so
        # without Phi nothing distinguishes moving towards it from moving away.
        self.assertAlmostEqual(flat.values[ACTIONS.index("RIGHT")],
                               flat.values[ACTIONS.index("LEFT")],
                               msg="without Phi an unreachable coin is invisible")
        self.assertGreater(shaped.values[ACTIONS.index("RIGHT")],
                           shaped.values[ACTIONS.index("LEFT")],
                           "with Phi, closing on the coin is worth more")

    def test_it_is_a_pure_addition_of_gamma_depth_times_phi(self):
        from lib.rewards import RewardConfig, potential
        from lib.search import leaf_potentials

        s = self.board_with_a_coin()
        cfg = SearchConfig(depth=2, gamma=0.9, max_leaves=10_000)
        states = []

        def spy(batch):
            states.extend(batch)
            return np.zeros((len(batch), len(ACTIONS)))

        search(s, 0, spy, cfg)
        rcfg = RewardConfig(gamma=cfg.gamma, reward_scale=cfg.reward_scale)
        want = np.array([potential(__import__("lib.features", fromlist=["a"])
                                   .analyse(g), rcfg) * cfg.reward_scale
                         for g in states])
        np.testing.assert_allclose(leaf_potentials(states, cfg), want)

    def test_a_terminal_leaf_carries_no_potential(self):
        """Phi of a terminal state is 0 -- that is what makes it telescope."""
        arena = empty_board()
        arena[15, 15] = 1
        s = Sim(arena=arena, agents=[SimAgent(1, 1)], active=[0])
        from lib.sim import SimBomb
        s.bombs = [SimBomb(1, 1, 0, timer=0)]
        r = search(s, 0, constant(5.0), SearchConfig(depth=1, use_potential=True))
        # Every action dies this step, so no leaf term of any kind is attached
        # and only the death price and the action's own step cost remain.
        finite = r.values[np.isfinite(r.values)]
        np.testing.assert_allclose(finite, KILLED_SELF * 0.1, atol=0.006)


class StepRewardTest(unittest.TestCase):
    def test_a_kill_and_a_coin_are_told_apart(self):
        conf = SearchConfig(reward_scale=1.0)
        before = solo()
        after = before.copy()
        after.agents[0].score += 5
        after.killed = [(1, 0)]
        self.assertAlmostEqual(step_reward(before, after, 0, "WAIT", conf),
                               15.0 + WAITED)

        after2 = before.copy()
        after2.agents[0].score += 1
        self.assertAlmostEqual(step_reward(before, after2, 0, "WAIT", conf),
                               3.0 + WAITED)

    def test_a_suicide_is_priced_below_being_killed(self):
        conf = SearchConfig(reward_scale=1.0)
        before = solo()
        mine = before.copy()
        mine.killed = [(0, 0)]
        theirs = before.copy()
        theirs.killed = [(0, 1)]
        self.assertAlmostEqual(step_reward(before, mine, 0, "WAIT", conf),
                               KILLED_SELF + WAITED)
        self.assertLess(step_reward(before, mine, 0, "WAIT", conf),
                        step_reward(before, theirs, 0, "WAIT", conf))


class SafetyTest(unittest.TestCase):
    """The point of the whole exercise: refuse deaths the search can see.

    The evaluator here is deliberately indifferent -- every leaf is worth the
    same -- so nothing but the death penalty on the path can separate the
    actions.  A flat ``argmax`` over the same evaluator cannot tell them apart
    at all.
    """

    def corridor(self) -> Sim:
        """A dead-end corridor with our own bomb about to go off behind us."""
        arena = empty_board()
        for x in range(1, 16):
            for y in range(1, 16):
                if y != 8:
                    arena[x, y] = -1
        s = Sim(arena=arena, agents=[SimAgent(6, 8, bombs_left=False)], active=[0])
        from lib.sim import SimBomb
        s.bombs = [SimBomb(4, 8, 0, timer=1)]
        return s

    def test_it_walks_away_from_a_bomb_rather_than_into_it(self):
        r = search(self.corridor(), 0, constant(1.0), cfg(depth=3))
        right = r.values[ACTIONS.index("RIGHT")]
        left = r.values[ACTIONS.index("LEFT")]
        self.assertGreater(right, left, "LEFT walks into the blast")
        self.assertEqual(int(np.argmax(np.where(np.isfinite(r.values),
                                                r.values, -np.inf))),
                         ACTIONS.index("RIGHT"))

    def test_a_flat_argmax_on_the_same_evaluator_cannot_tell(self):
        """Establishes that the previous test measures the search, not the values."""
        q = constant(1.0)([self.corridor().to_game_state(0)])[0]
        self.assertEqual(len(set(q.tolist())), 1)

    def test_bombing_into_a_dead_end_is_seen_as_lethal(self):
        arena = empty_board()
        for x in range(1, 16):
            for y in range(1, 16):
                if y != 8 or x > 4:
                    arena[x, y] = -1
        s = Sim(arena=arena, agents=[SimAgent(3, 8)], active=[0])
        r = search(s, 0, constant(1.0), cfg(depth=5))
        self.assertLess(r.values[ACTIONS.index("BOMB")],
                        r.values[ACTIONS.index("WAIT")])


class BookkeepingTest(unittest.TestCase):
    def test_it_reports_what_it_expanded(self):
        r = search(solo(), 0, constant(0.0), cfg(depth=3))
        self.assertGreater(r.nodes, 100)
        self.assertGreater(r.leaves, 0)
        self.assertFalse(r.truncated)

    def test_truncation_is_reported_not_hidden(self):
        r = search(solo(), 0, constant(0.0), SearchConfig(depth=4, max_leaves=20))
        self.assertTrue(r.truncated)
        self.assertTrue(np.isfinite(r.values).any(), "still returns usable values")

    def test_the_cap_is_never_exceeded(self):
        """A latency bound that only usually holds is not a latency bound."""
        for cap in (1, 6, 20, 48):
            r = search(solo(), 0, constant(0.0),
                       SearchConfig(depth=4, max_leaves=cap))
            # One ply is the floor: at most six leaves, never fewer than a search.
            self.assertLessEqual(r.leaves, max(cap, 6), f"cap {cap}")

    def test_truncation_falls_back_to_a_complete_shallower_level(self):
        """Half an expanded level would compare root actions at mixed depths."""
        seen = []

        def spy(states):
            seen.append(len(states))
            return np.zeros((len(states), len(ACTIONS)))

        shallow = search(solo(), 0, spy, SearchConfig(depth=2, max_leaves=100_000))
        n_depth2 = seen[-1]
        capped = search(solo(), 0, spy, SearchConfig(depth=3, max_leaves=n_depth2))
        self.assertTrue(capped.truncated)
        self.assertEqual(seen[-1], n_depth2)
        self.assertEqual(capped.leaves, shallow.leaves)

    def test_deduplication_keeps_the_tree_smaller_than_the_worst_case(self):
        r = search(solo(), 0, constant(0.0), cfg(depth=3))
        self.assertLess(r.nodes, 6 ** 3 + 6 ** 2 + 6)

    def test_the_root_position_is_not_mutated(self):
        s = solo()
        before = s.state()
        search(s, 0, constant(0.0), cfg(depth=3))
        self.assertEqual(s.state(), before)


if __name__ == "__main__":
    unittest.main()
