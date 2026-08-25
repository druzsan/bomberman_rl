"""End-to-end checks on the training callbacks and the run pipeline.

The transition test exists because of a bug that was invisible from the outside:
the engine stamps ``new_game_state`` with the step that just finished, so the
two states of one transition share a step number.  Any cache keyed on it hands
back the old analysis as the new one, and every TD update degenerates into
``Q(s,a) <- r + gamma * max_a Q(s,a)``.  That still ranks actions by immediate
reward, so a coin-collection stage converges and looks healthy -- while the
agent can never learn to escape its own bomb.  The assertions below pin the
symptom (a bomb must produce a MOVED_INTO_DANGER event and a distinct next
state) rather than the implementation.
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

import agent_code.q_tabular_agent.callbacks as cb
import agent_code.q_tabular_agent.train as tr
from lib.rewards import ESCAPED_DANGER, MOVED_INTO_DANGER
from training.engine import make_world, quiet_logging
from training.shared import CTRL_SIZE, N_ACTIONS


class LocalTables:
    """Stand-in for the driver's shared memory."""

    def __init__(self, n_states: int, feature_set: str):
        self.q = np.zeros((n_states, N_ACTIONS), dtype=np.float32)
        self.n = np.zeros((n_states, N_ACTIONS), dtype=np.uint32)
        self.ctrl = np.zeros(CTRL_SIZE, dtype=np.float64)
        self.ctrl[tr.CTRL.EPSILON] = 0.3
        self.ctrl[tr.CTRL.SHAPING] = 1.0
        self.ctrl[tr.CTRL.ALPHA] = 0.1
        self.ctrl[tr.CTRL.GAMMA] = 0.95
        self.ctrl[tr.CTRL.N_STEP] = 1
        self.ctrl[tr.CTRL.LEARN] = 1.0
        self.ctrl[tr.CTRL.ALLOW_BOMB] = 1.0
        self.feature_set = feature_set
        self.fold = True
        self.worker_id = 0


class TransitionTest(unittest.TestCase):
    """Run real rounds with learning on and inspect what the callbacks saw."""

    @classmethod
    def setUpClass(cls):
        quiet_logging()
        from lib.features import FeatureSpec

        cls.tables = LocalTables(FeatureSpec("TQ-M-solo").n_states, "TQ-M-solo")
        cb.TRAINING_TABLES = cls.tables
        cls.seen_events: Counter[str] = Counter()
        cls.pairs: list[tuple[int, int]] = []
        cls.distinct = 0
        cls.with_next = 0
        cls.total = 0

        orig_store = tr._store
        orig_custom = tr.custom_events

        def spy(self, old_state, old_info, action, new_info, events, *, terminal, key):
            cls.total += 1
            if new_info is not None:
                cls.with_next += 1
                cls.distinct += int(new_info is not old_info)
            cls.seen_events.update(events)
            return orig_store(self, old_state, old_info, action, new_info, events,
                              terminal=terminal, key=key)

        def spy_custom(*args, **kwargs):
            out = orig_custom(*args, **kwargs)
            cls.seen_events.update(out)
            return out

        tr._store = spy
        tr.custom_events = spy_custom
        try:
            world = make_world([("q_tabular_agent", True)], scenario="loot-crate",
                               continue_without_training=False)
            for r in range(6):
                world.rng = np.random.default_rng(4000 + r)
                world.new_round()
                while world.running:
                    world.do_step()
        finally:
            tr._store = orig_store
            tr.custom_events = orig_custom
            cb.TRAINING_TABLES = None

    def test_new_state_is_not_the_old_state(self):
        self.assertGreater(self.total, 50)
        self.assertEqual(self.distinct, self.with_next,
                         "new_game_state was analysed as the old state")

    def test_bombing_is_seen_as_entering_danger(self):
        self.assertGreater(self.seen_events["BOMB_DROPPED"], 0)
        self.assertGreater(self.seen_events[MOVED_INTO_DANGER], 0,
                           "dropping a bomb must register as entering danger")

    def test_danger_transitions_are_classified(self):
        """Unit-check the three danger events on synthetic state pairs.

        The integration run above is too short and too random to guarantee a
        successful escape, but the classification itself must be exact.
        """
        from lib.rewards import STAYED_IN_DANGER, RewardConfig, custom_events

        cfg = RewardConfig()
        safe = _fake_info(danger_tau=-1)
        danger = _fake_info(danger_tau=3)
        self.assertIn(MOVED_INTO_DANGER, custom_events(safe, danger, [], cfg))
        self.assertIn(ESCAPED_DANGER, custom_events(danger, safe, [], cfg))
        self.assertIn(STAYED_IN_DANGER,
                      custom_events(danger, _fake_info(danger_tau=2), [], cfg))
        self.assertEqual(custom_events(safe, safe, [], cfg), [])

    def test_updates_were_applied(self):
        self.assertGreater(int((self.tables.n > 0).sum()), 10)
        self.assertTrue(np.isfinite(self.tables.q).all())


def _fake_info(*, danger_tau: int):
    """Minimal StateInfo stand-in for the danger-classification unit test."""
    from lib.features import StateInfo

    return StateInfo(
        pos=(1, 1), bombs_left=True, step=1, n_others=0,
        field=np.zeros((3, 3), dtype=int), passable=np.ones((3, 3), dtype=bool),
        lethal=np.zeros((3, 3), dtype=np.uint8), blocked=np.zeros((3, 3), dtype=bool),
        occupied=np.zeros((3, 3), dtype=bool), danger_tau=danger_tau,
        escape=np.zeros(5, dtype=np.int16), escape_possible=True, safe_here=danger_tau < 0,
        survivable=np.ones(5, dtype=bool), objective=0, obj_dist=-1, target_dir=5,
        opp_dist=-1, opp_dir_raw=4, threat=None, target_mask=0, opp_mask=0, escape_mask=0,
        bomb_crates=0, bomb_hits=0, bomb_safe=True, bomb_safe_strict=True,
        bomb_safe_slack=True, in_dead_end=False, legal=np.ones(6, dtype=bool),
    )


class StepNumberTest(unittest.TestCase):
    def test_engine_reuses_the_step_number_across_a_transition(self):
        """Pin the engine behaviour the cache fix is built on."""
        quiet_logging()
        seen: list[tuple[int, int]] = []
        import agent_code.q_tabular_agent.train as train_module

        orig = train_module.game_events_occurred

        def spy(self, old_game_state, self_action, new_game_state, events):
            seen.append((old_game_state["step"], new_game_state["step"]))
            return orig(self, old_game_state, self_action, new_game_state, events)

        from lib.features import FeatureSpec

        cb.TRAINING_TABLES = LocalTables(FeatureSpec("TQ-M-solo").n_states, "TQ-M-solo")
        train_module.game_events_occurred = spy
        try:
            world = make_world([("q_tabular_agent", True)], scenario="coin-heaven",
                               continue_without_training=False)
            world.rng = np.random.default_rng(1)
            world.new_round()
            for _ in range(10):
                if not world.running:
                    break
                world.do_step()
        finally:
            train_module.game_events_occurred = orig
            cb.TRAINING_TABLES = None
        self.assertTrue(seen)
        self.assertTrue(all(a == b for a, b in seen),
                        f"expected identical step numbers, got {seen[:5]}")


if __name__ == "__main__":
    unittest.main()
