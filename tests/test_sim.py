"""Conformance test for :mod:`lib.sim` against the real engine.

`lib.sim` exists so a search can plan in the game's own dynamics.  A world model
that is *nearly* right is worse than none at all -- it produces confident values
for positions that cannot occur -- so the claim it makes has to be tested
directly rather than argued for.

The method is a replay conformance test.  A recorded game fixes the board, the
agents' actions and, crucially, the per-step permutation ``world.rng`` drew, so
the engine's ``do_step`` becomes a deterministic function of recorded data.
:class:`ReplayWorld` re-runs it; this test steps :class:`lib.sim.Sim` alongside
and asserts the two digests are equal after *every* step of *every* round.  A
single mismatch fails, and the failure names the step and the field.

Recordings come from ``fixtures/`` if they are there and are otherwise played on
the spot, so the test is self-contained.  The agents are chosen to exercise the
mechanics that are easy to get wrong: `rule_based_agent` bombs and escapes,
`coin_collector_agent` walks over coins, and both die often enough that
mid-round eliminations are covered.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lib.sim import Sim, blast_coords
from training.engine import make_args, quiet_logging

REPO = Path(__file__).resolve().parent.parent
AGENTS = ["rule_based_agent", "coin_collector_agent",
          "rule_based_agent", "random_agent"]
ROUNDS = 3


def record(tmp: Path, scenario: str, seed: int) -> list[Path]:
    """Play `ROUNDS` rounds with replay recording on and return the replay files."""
    from environment import BombeRLeWorld

    args = make_args(scenario=scenario, log_dir=str(tmp / "logs"))
    args.save_replay = True
    world = BombeRLeWorld(args, [(a, False) for a in AGENTS])
    world.rng = np.random.default_rng(seed)
    before = set((REPO / "replays").glob("*.pt"))
    for _ in range(ROUNDS):
        world.new_round()
        while world.running:
            world.do_step()
        if world.running:
            world.end_round()
    world.end()
    new = sorted(set((REPO / "replays").glob("*.pt")) - before)
    moved = []
    for path in new:
        dest = tmp / path.name
        shutil.move(str(path), dest)
        moved.append(dest)
    return moved


class ConformanceTest(unittest.TestCase):
    """One recorded game, stepped in the engine and in the model, digest by digest."""

    @classmethod
    def setUpClass(cls):
        quiet_logging()
        cls.tmp = Path(tempfile.mkdtemp(prefix="simconf-"))
        cls.replays = record(cls.tmp, "classic", seed=20260827)
        cls.replays += record(cls.tmp, "loot-crate", seed=11)
        assert cls.replays, "no replays were recorded"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def replay_world(self, path: Path):
        from replay import ReplayWorld

        args = make_args(log_dir=str(self.tmp / "logs"))
        args.replay = str(path)
        return ReplayWorld(SimpleNamespace(**vars(args)))

    def test_every_step_of_every_replay_matches(self):
        total_steps = 0
        for path in self.replays:
            with self.subTest(replay=path.name):
                world = self.replay_world(path)
                world.new_round()
                sim = Sim.from_world(world)
                self.assertEqual(sim.state(), Sim.from_world(world).state())
                while world.running:
                    step = world.step + 1
                    perm = world.loaded_replay["permutations"][world.step]
                    # Only *active* agents get an entry appended per step, so a
                    # dead agent's action list stops at the step it died.
                    live = {id(a) for a in world.active_agents}
                    actions = {i: world.loaded_replay["actions"][a.name][world.step]
                               for i, a in enumerate(world.agents) if id(a) in live}
                    world.do_step()
                    sim.step(actions, perm)
                    total_steps += 1
                    self.assertEqual(
                        sim.state(), Sim.from_world(world).state(),
                        f"{path.name}: diverged at step {step}\n"
                        + self.explain(sim, Sim.from_world(world)))
                self.assertGreater(world.step, 5)
        self.assertGreater(total_steps, 100, "the replays were too short to prove much")

    @staticmethod
    def explain(got: Sim, want: Sim) -> str:
        """Name the field that differs, so a failure is a lead and not a hex dump."""
        lines = []
        if not np.array_equal(got.arena, want.arena):
            diff = np.argwhere(got.arena != want.arena)
            lines.append(f"  arena differs at {[tuple(map(int, c)) for c in diff][:8]}")
        for i, (a, b) in enumerate(zip(got.agents, want.agents)):
            if (a.x, a.y, a.bombs_left, a.dead, a.score) != \
                    (b.x, b.y, b.bombs_left, b.dead, b.score):
                lines.append(f"  agent {i} {b.name}: sim={a} engine={b}")
        if got.active != want.active:
            lines.append(f"  active: sim={got.active} engine={want.active}")
        if sorted(map(str, got.bombs)) != sorted(map(str, want.bombs)):
            lines.append(f"  bombs: sim={got.bombs} engine={want.bombs}")
        if sorted(map(str, got.explosions)) != sorted(map(str, want.explosions)):
            lines.append(f"  explosions: sim={got.explosions} engine={want.explosions}")
        if got.coins != want.coins:
            bad = [i for i, (a, b) in enumerate(zip(got.coins, want.coins)) if a != b]
            lines.append(f"  coins differ at {bad[:8]}")
        return "\n".join(lines) or "  (state() differs but no field does -- check state())"


class BlastGeometryTest(unittest.TestCase):
    """§2.4 of the plan: blasts stop at walls and pass *through* crates."""

    def arena(self) -> np.ndarray:
        a = np.zeros((17, 17), dtype=np.int8)
        a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = -1
        return a

    def test_open_blast_is_a_full_cross(self):
        coords = set(blast_coords(self.arena(), 8, 8))
        want = {(8, 8)} | {(8 + d, 8) for d in (-3, -2, -1, 1, 2, 3)} \
                        | {(8, 8 + d) for d in (-3, -2, -1, 1, 2, 3)}
        self.assertEqual(coords, want)

    def test_crates_do_not_block(self):
        a = self.arena()
        a[9, 8] = a[10, 8] = 1
        self.assertIn((11, 8), blast_coords(a, 8, 8))

    def test_walls_block_and_are_not_included(self):
        a = self.arena()
        a[10, 8] = -1
        coords = set(blast_coords(a, 8, 8))
        self.assertIn((9, 8), coords)
        self.assertNotIn((10, 8), coords)
        self.assertNotIn((11, 8), coords)

    def test_matches_the_engine_on_the_stock_arena(self):
        from items import Bomb

        rng = np.random.default_rng(3)
        a = self.arena()
        a[np.ix_(range(2, 16, 2), range(2, 16, 2))] = -1
        a[(rng.random(a.shape) < 0.75) & (a == 0)] = 1
        for x in range(1, 16):
            for y in range(1, 16):
                if a[x, y] == -1:
                    continue
                bomb = Bomb.__new__(Bomb)
                bomb.x, bomb.y, bomb.power = x, y, 3
                self.assertEqual(list(blast_coords(a, x, y)),
                                 list(bomb.get_blast_coords(a)), f"at {(x, y)}")


class ExplosionLifetimeTest(unittest.TestCase):
    """An explosion is lethal for exactly two steps and returns the bomb on the third."""

    def sim(self) -> Sim:
        from lib.sim import SimAgent

        arena = np.zeros((17, 17), dtype=np.int8)
        arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
        return Sim(arena=arena, agents=[SimAgent(1, 1), SimAgent(15, 15)],
                   active=[0, 1])

    def test_bomb_takes_four_steps_and_kills_for_two(self):
        s = self.sim()
        s.step({0: "BOMB", 1: "WAIT"})
        self.assertEqual([b.timer for b in s.bombs], [3])
        self.assertFalse(s.agents[0].bombs_left)
        for expected in (2, 1, 0):
            s.step({0: "WAIT", 1: "WAIT"})
            self.assertEqual([b.timer for b in s.bombs], [expected])
        self.assertFalse(s.danger_map()[1, 1], "a bomb at timer 0 has not gone off yet")
        # Five steps from the drop, not four: `update_bombs` decrements in the
        # same step the bomb is placed, and detonates on the step it *finds*
        # the timer at zero.
        s.step({0: "WAIT", 1: "WAIT"})
        self.assertEqual(s.bombs, [])
        self.assertTrue(s.danger_map()[1, 1])

    def test_the_owner_dies_standing_on_its_own_bomb(self):
        s = self.sim()
        s.step({0: "BOMB", 1: "WAIT"})
        for _ in range(4):
            s.step({0: "WAIT", 1: "WAIT"})
        self.assertTrue(s.agents[0].dead)
        self.assertEqual(s.killed, [(0, 0)])
        self.assertEqual(s.agents[0].score, 0, "a suicide pays nobody")
        self.assertEqual(s.active, [1])

    def test_a_kill_pays_five_and_the_victim_nothing(self):
        from lib.sim import SimAgent

        s = self.sim()
        s.agents[1] = SimAgent(1, 2)
        s.step({0: "BOMB", 1: "WAIT"})
        for _ in range(4):
            s.step({0: "WAIT", 1: "WAIT"})
        self.assertTrue(s.agents[1].dead)
        self.assertTrue(s.agents[0].dead, "the owner is standing in its own blast")
        self.assertEqual(s.agents[0].score, 5)
        self.assertEqual(s.agents[1].score, 0)

    def test_the_bomb_comes_back_two_steps_after_the_blast(self):
        s = self.sim()
        s.step({0: "BOMB"})
        for _ in range(3):
            s.step({0: "LEFT"})          # blocked by the wall; stays put, survives?
        s.agents[0].x, s.agents[0].y = 5, 5      # teleport clear of the blast
        s.step({0: "WAIT"})                      # detonation
        self.assertFalse(s.agents[0].bombs_left)
        self.assertTrue(s.danger_map()[1, 1])
        s.step({0: "WAIT"})
        self.assertFalse(s.agents[0].bombs_left, "still lethal, bomb not yet returned")
        s.step({0: "WAIT"})
        self.assertTrue(s.agents[0].bombs_left)
        self.assertFalse(s.danger_map()[1, 1])


class OrderingTest(unittest.TestCase):
    """Within a step, who moves first decides whether a move is legal at all."""

    def sim(self):
        from lib.sim import SimAgent

        arena = np.zeros((17, 17), dtype=np.int8)
        arena[0, :] = arena[-1, :] = arena[:, 0] = arena[:, -1] = -1
        return Sim(arena=arena, agents=[SimAgent(5, 5), SimAgent(7, 5)],
                   active=[0, 1])

    def test_both_agents_cannot_take_the_same_tile(self):
        s = self.sim()
        s.step({0: "RIGHT", 1: "LEFT"}, perm=[0, 1])
        self.assertEqual((s.agents[0].x, s.agents[0].y), (6, 5))
        self.assertEqual((s.agents[1].x, s.agents[1].y), (7, 5), "second mover blocked")

    def test_the_permutation_decides_which_one_wins(self):
        s = self.sim()
        s.step({0: "RIGHT", 1: "LEFT"}, perm=[1, 0])
        self.assertEqual((s.agents[1].x, s.agents[1].y), (6, 5))
        self.assertEqual((s.agents[0].x, s.agents[0].y), (5, 5))

    def test_a_bomb_dropped_earlier_blocks_a_move_later(self):
        s = self.sim()
        s.agents[1].x = 6
        s.step({0: "RIGHT", 1: "RIGHT"}, perm=[1, 0])
        self.assertEqual((s.agents[1].x, s.agents[1].y), (7, 5))
        s2 = self.sim()
        s2.agents[1].x = 6
        s2.step({0: "WAIT", 1: "BOMB"}, perm=[1, 0])
        s2.step({0: "RIGHT", 1: "RIGHT"}, perm=[1, 0])
        self.assertEqual((s2.agents[0].x, s2.agents[0].y), (5, 5),
                         "the bomb at (6,5) blocks the move")


class CopyTest(unittest.TestCase):
    """Search makes one copy per node; a shared array would corrupt siblings."""

    def test_copy_is_deep(self):
        from lib.sim import SimAgent

        arena = np.zeros((17, 17), dtype=np.int8)
        arena[3, 3] = 1
        s = Sim(arena=arena, agents=[SimAgent(1, 1)], active=[0],
                coins=[(3, 3, False)])
        c = s.copy()
        before = c.state()
        s.step({0: "BOMB"})
        for _ in range(5):
            s.step({0: "WAIT"})
        self.assertEqual(c.state(), before)
        self.assertNotEqual(s.state(), before)


if __name__ == "__main__":
    unittest.main()
