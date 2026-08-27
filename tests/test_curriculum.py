"""Tests for the two curriculum knobs added for E15 (plan §15 open questions 7, 8).

Both are small, both sit inside a two-hour training run, and both fail silently
if they are wrong: a margin schedule that never changes shape produces the same
collapse it was meant to remove, and a ``scenario_mix`` that is ignored produces
a run that looks exactly like the baseline it was meant to differ from.
"""

from __future__ import annotations

import math
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from training.dqn_actor import Actor
from training.dqn_learner import DQNLearner


class MarginScheduleTest(unittest.TestCase):
    """``margin_anneal="cosine"`` must be gentler at the endpoint, not merely different."""

    END = 0.20

    def weight(self, frac: float, **cfg) -> float:
        learner = SimpleNamespace(
            cfg=dict(margin_weight=1.0, margin_anneal_frac=self.END, **cfg))
        return DQNLearner.margin_weight(learner, frac)

    def test_linear_is_unchanged_by_default(self):
        for frac in (0.0, 0.05, 0.1, 0.15, 0.2, 0.5):
            self.assertAlmostEqual(
                self.weight(frac), max(0.0, 1.0 - frac / self.END), places=9)

    def test_both_schedules_start_at_one_and_end_at_zero(self):
        for shape in ("linear", "cosine"):
            self.assertAlmostEqual(self.weight(0.0, margin_anneal=shape), 1.0)
            self.assertAlmostEqual(self.weight(self.END, margin_anneal=shape), 0.0)
            self.assertAlmostEqual(self.weight(0.9, margin_anneal=shape), 0.0)

    def test_cosine_is_monotone(self):
        prev = math.inf
        for i in range(101):
            w = self.weight(self.END * i / 100.0, margin_anneal="cosine")
            self.assertLessEqual(w, prev + 1e-12)
            prev = w

    def test_cosine_is_far_smaller_near_the_endpoint(self):
        """The whole point: the weight is already gone before it is switched off.

        E13's collapse is at the step the weight reaches zero, so what matters is
        how much is left just before that step, not that both reach zero.
        """
        near = 0.975 * self.END
        self.assertLess(self.weight(near, margin_anneal="cosine"),
                        self.weight(near) / 10.0)

    def test_floor_is_a_fraction_of_the_initial_weight_and_persists(self):
        w = self.weight(0.9, margin_anneal="cosine", margin_floor=0.05)
        self.assertAlmostEqual(w, 0.05)
        self.assertGreater(self.weight(0.1, margin_anneal="cosine", margin_floor=0.05),
                           self.weight(0.1, margin_anneal="cosine"))

    def test_no_margin_window_still_means_no_margin(self):
        learner = SimpleNamespace(cfg={"margin_weight": 1.0, "margin_anneal_frac": 0.0,
                                       "margin_anneal": "cosine", "margin_floor": 0.05})
        self.assertEqual(DQNLearner.margin_weight(learner, 0.0), 0.0)


class ScenarioMixTest(unittest.TestCase):
    """``_sample_episode`` decides which board every training episode is played on."""

    def actor(self, seed: int = 0) -> SimpleNamespace:
        """A stand-in with just enough of an ``Actor`` to sample an episode.

        Building a real one would spawn worlds and logger handlers; the two
        methods under test touch nothing but ``self.rng``.
        """
        stub = SimpleNamespace(rng=np.random.default_rng(seed))
        stub._sample_opponents = lambda stage: Actor._sample_opponents(stub, stage)
        return stub

    def sample(self, stage: dict, n: int = 4000, seed: int = 0):
        actor = self.actor(seed)
        return Counter(Actor._sample_episode(actor, stage) for _ in range(n))

    def base_stage(self, **extra) -> dict:
        return dict(scenario="classic",
                    opponent_mix=[(0.5, ["rule_based_agent"] * 3), (0.5, [])],
                    **extra)

    def test_without_a_mix_every_episode_uses_the_stage_scenario(self):
        counts = self.sample(self.base_stage())
        self.assertEqual({s for s, _ in counts}, {"classic"})
        self.assertEqual({o for _, o in counts},
                         {("rule_based_agent",) * 3, ()})

    def test_share_is_respected(self):
        stage = self.base_stage(scenario_mix=[(0.9, "classic", None),
                                              (0.1, "coin-heaven", [])])
        counts = self.sample(stage)
        n = sum(counts.values())
        heaven = sum(c for (s, _), c in counts.items() if s == "coin-heaven")
        self.assertAlmostEqual(heaven / n, 0.10, delta=0.02)

    def test_the_coin_heaven_share_is_solo(self):
        stage = self.base_stage(scenario_mix=[(0.9, "classic", None),
                                              (0.1, "coin-heaven", [])])
        for (scenario, opponents) in self.sample(stage):
            if scenario == "coin-heaven":
                self.assertEqual(opponents, ())

    def test_none_opponents_defers_to_the_opponent_mix(self):
        """This is what keeps the league (which rewrites opponent_mix) working."""
        stage = self.base_stage(scenario_mix=[(1.0, "classic", None)])
        counts = self.sample(stage)
        classic = {o: c for (s, o), c in counts.items() if s == "classic"}
        self.assertEqual(set(classic), {("rule_based_agent",) * 3, ()})
        n = sum(classic.values())
        self.assertAlmostEqual(classic[()] / n, 0.5, delta=0.03)

    def test_weights_need_not_be_normalised(self):
        stage = self.base_stage(scenario_mix=[(9, "classic", None),
                                              (1, "coin-heaven", [])])
        counts = self.sample(stage)
        n = sum(counts.values())
        heaven = sum(c for (s, _), c in counts.items() if s == "coin-heaven")
        self.assertAlmostEqual(heaven / n, 0.10, delta=0.02)

    def test_returned_opponents_are_hashable_world_cache_keys(self):
        stage = self.base_stage(scenario_mix=[(0.9, "classic", None),
                                              (0.1, "coin-heaven", [])])
        for scenario, opponents in self.sample(stage, n=200):
            self.assertIsInstance(scenario, str)
            self.assertIsInstance(opponents, tuple)


class ConfigDiffTest(unittest.TestCase):
    """Each E15 arm must differ from the shipped run in exactly one place."""

    def load(self, path: str) -> dict:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_cfg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.config()

    def test_margin_config_reaches_the_learner_schedule(self):
        """Close the loop: the config file, not a hand-built dict, changes the curve.

        The shipped run's ``margin_weight`` reaches zero at 1.60 M of 8 M steps
        and E13's collapsed checkpoint is at 1 600 360.  What has to be smaller
        is the weight *just before* that step.
        """
        base = self.load("training/configs/s3_gamma95.py")
        margin = self.load("training/configs/s3_margin.py")
        near = 0.99 * base["margin_anneal_frac"]
        w = lambda cfg: DQNLearner.margin_weight(SimpleNamespace(cfg=cfg), near)
        self.assertGreater(w(base), 0.005)
        self.assertLess(w(margin), w(base) / 20.0)
        self.assertAlmostEqual(w(base), w(margin), places=1)  # both are near zero
        for key in set(base) | set(margin):
            if key not in ("margin_anneal", "margin_floor"):
                self.assertEqual(repr(base[key]), repr(margin[key]), key)

    def test_coinheaven_differs_from_gamma95_only_in_the_scenario_mix(self):
        base = self.load("training/configs/s3_gamma95.py")
        coin = self.load("training/configs/s3_coinheaven.py")
        self.assertIsNone(base["stages"][-1].get("scenario_mix"))
        self.assertEqual(coin["stages"][-1]["scenario_mix"],
                         [(0.9, "classic", None), (0.1, "coin-heaven", [])])
        strip = lambda cfg: [{k: v for k, v in s.items() if k != "scenario_mix"}
                             for s in cfg["stages"]]
        self.assertEqual(strip(base), strip(coin))
        for key in set(base) | set(coin):
            if key != "stages":
                self.assertEqual(repr(base[key]), repr(coin[key]), key)


if __name__ == "__main__":
    unittest.main()
