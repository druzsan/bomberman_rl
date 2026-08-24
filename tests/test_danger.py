"""Conformance tests for the threat model against the real engine.

``lib.danger`` encodes two rules the engine only states implicitly: blasts pass
through crates, and every blast tile kills for two consecutive steps.  Rather
than assert those from memory, the timing test replays the engine's own
``update_explosions`` / ``update_bombs`` / ``evaluate_explosions`` sequence with
the agent teleported along a chosen path and checks that the bit field predicts
exactly the steps in which the engine kills it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import settings as s
from environment import BombeRLeWorld
from items import Bomb, Explosion
from lib import danger
from lib.board import CRATE, FREE, WALL

WORLD_ARGS = SimpleNamespace(
    no_gui=True, fps=15, turn_based=False, update_interval=0.1, save_replay=False,
    replay=None, make_video=False, continue_without_training=True, log_dir="logs",
    save_stats=False, match_name=None, seed=1, silence_errors=False, scenario="empty",
)


def _world():
    world = BombeRLeWorld(WORLD_ARGS, [("peaceful_agent", False)])
    world.new_round()
    return world


def _empty_field(cols=9, rows=9):
    field = np.zeros((cols, rows), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = WALL
    for x in range(cols):
        for y in range(rows):
            if (x + 1) * (y + 1) % 2 == 1:
                field[x, y] = WALL
    return field


class BlastGeometryTest(unittest.TestCase):
    def test_crates_do_not_block_blasts(self):
        field = _empty_field()
        field[3, 1] = CRATE
        field[1, 3] = CRATE
        coords = set(danger.blast_coords(field, 1, 1))
        self.assertEqual(coords, {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)})

    def test_walls_block_blasts(self):
        field = _empty_field()
        field[2, 1] = WALL
        coords = set(danger.blast_coords(field, 1, 1))
        self.assertEqual(coords, {(1, 1), (1, 2), (1, 3), (1, 4)})

    def test_matches_engine_implementation(self):
        rng = np.random.default_rng(0)
        world = _world()
        owner = world.active_agents[0]
        for _ in range(50):
            field = _empty_field(17, 17)
            free = field == FREE
            field[free & (rng.random(field.shape) < 0.4)] = CRATE
            xs, ys = np.nonzero(field != WALL)
            i = rng.integers(len(xs))
            bomb = Bomb((int(xs[i]), int(ys[i])), owner, s.BOMB_TIMER, s.BOMB_POWER, None)
            self.assertEqual(
                sorted(bomb.get_blast_coords(field)),
                sorted(danger.blast_coords(field, int(xs[i]), int(ys[i]))),
            )


class LethalTimingTest(unittest.TestCase):
    """The bit field must predict engine deaths exactly, per tile and per step."""

    def _engine_death_step(self, field, bombs, explosions, path):
        """Play ``len(path)`` steps of the real world update sequence.

        ``path[i]`` is where the agent stands at the end of step ``i`` (i.e.
        ``tau = i``).  Returns the first ``tau`` at which the engine kills it.
        """
        world = _world()
        world.arena = np.array(field)
        world.coins = []
        agent = world.active_agents[0]
        world.bombs = [Bomb(pos, agent, timer, s.BOMB_POWER, None) for pos, timer in bombs]
        world.explosions = [
            Explosion(danger.blast_coords(field, *pos), [], agent, s.EXPLOSION_TIMER)
            for pos in explosions
        ]
        for tau, (px, py) in enumerate(path):
            agent.x, agent.y = px, py
            world.step += 1
            world.collect_coins()
            world.update_explosions()
            world.update_bombs()
            world.evaluate_explosions()
            if agent.dead:
                return tau
        return None

    def _observed_explosion_map(self, field, explosions):
        emap = np.zeros(field.shape)
        for pos in explosions:
            for x, y in danger.blast_coords(field, *pos):
                emap[x, y] = max(emap[x, y], s.EXPLOSION_TIMER - 1)
        return emap

    def test_single_bomb_all_tiles_all_taus(self):
        field = _empty_field(9, 9)
        for timer in range(s.BOMB_TIMER + 1):
            bombs = [((1, 1), timer)]
            lethal = danger.lethal_bits(field, bombs, None)
            for tau in range(danger.MAX_TAU + 1):
                for x, y in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (5, 5)]:
                    predicted = bool(lethal[x, y] & (1 << tau))
                    # Stand on the tile only at ``tau`` and somewhere provably
                    # safe before and after, so a death pins down that step.
                    path = [(7, 7)] * tau + [(x, y)]
                    died = self._engine_death_step(field, bombs, [], path)
                    self.assertEqual(predicted, died == tau,
                                     f"timer={timer} tile={(x, y)} tau={tau} died={died}")

    def test_lingering_explosion(self):
        field = _empty_field(9, 9)
        explosions = [(1, 1)]
        lethal = danger.lethal_bits(field, [], self._observed_explosion_map(field, explosions))
        for tau in range(3):
            for x, y in [(1, 1), (1, 3), (5, 5)]:
                predicted = bool(lethal[x, y] & (1 << tau))
                path = [(7, 7)] * tau + [(x, y)]
                died = self._engine_death_step(field, [], explosions, path)
                self.assertEqual(predicted, died == tau, f"tile={(x, y)} tau={tau}")

    def test_fresh_bomb_bits(self):
        """A bomb dropped now kills four steps later, and the step after."""
        field = _empty_field(9, 9)
        lethal = danger.lethal_bits(field, [], None, extra_bombs=((1, 1),))
        world = _world()
        world.arena = np.array(field)
        world.coins = []
        agent = world.active_agents[0]
        agent.x, agent.y = 1, 1
        world.perform_agent_action(agent, "BOMB")
        deaths = []
        for tau in range(danger.MAX_TAU + 1):
            world.step += 1
            world.update_explosions()
            world.update_bombs()
            world.evaluate_explosions()
            deaths.append(agent.dead)
            if agent.dead:
                # Revive so the *next* step can kill it too: that second step is
                # exactly the lingering-blast rule under test.
                agent.dead = False
                world.active_agents.append(agent)
        predicted = [bool(lethal[1, 1] & (1 << t)) for t in range(danger.MAX_TAU + 1)]
        self.assertEqual(predicted, deaths)


class EscapeSearchTest(unittest.TestCase):
    """The time-expanded BFS must agree with brute-force enumeration."""

    def _brute_force(self, lethal, blocked, blocked_now, start, horizon):
        best = np.full(5, -1, dtype=np.int16)
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

        def walk(pos, tau, first):
            if lethal[pos] & (1 << tau):
                return
            if (lethal[pos] >> tau) == 0:
                if best[first] < 0 or tau < best[first]:
                    best[first] = tau
                return
            if tau >= horizon:
                return
            for dx, dy in moves:
                nxt = (pos[0] + dx, pos[1] + dy)
                if (dx or dy) and blocked[nxt]:
                    continue
                walk(nxt, tau + 1, first)

        for k, (dx, dy) in enumerate(moves):
            nxt = (start[0] + dx, start[1] + dy)
            if (dx or dy) and (blocked[nxt] or blocked_now[nxt]):
                continue
            walk(nxt, 0, k)
        return best

    def test_matches_brute_force(self):
        rng = np.random.default_rng(3)
        for trial in range(40):
            field = _empty_field(9, 9)
            free = field == FREE
            field[free & (rng.random(field.shape) < 0.25)] = CRATE
            free_xy = np.argwhere(field == FREE)
            start = tuple(int(v) for v in free_xy[rng.integers(len(free_xy))])
            n_bombs = int(rng.integers(1, 4))
            bombs = []
            for _ in range(n_bombs):
                p = free_xy[rng.integers(len(free_xy))]
                bombs.append(((int(p[0]), int(p[1])), int(rng.integers(0, s.BOMB_TIMER + 1))))
            lethal = danger.lethal_bits(field, bombs, None)
            blocked = danger.blocked_mask(field, bombs)
            blocked_now = np.zeros(field.shape, dtype=bool)
            got = danger.escape_taus(lethal, blocked, start, blocked_now=blocked_now)
            want = self._brute_force(lethal, blocked, blocked_now, start, danger.MAX_TAU)
            np.testing.assert_array_equal(got, want, err_msg=f"trial {trial} start {start}")


if __name__ == "__main__":
    unittest.main()
