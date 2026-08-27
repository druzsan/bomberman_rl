"""Curriculum bookkeeping shared by the training drivers.

A run is a list of stages, each with its own scenario, opponent mixture and
epsilon ramp; progress is counted in environment steps (``dev/plan.md`` §9.1
R1).  Everything that maps "how many steps have we done" onto "what should the
actors be doing right now" lives here, so the tabular and the deep driver cannot
drift apart on the one thing they genuinely share.
"""

from __future__ import annotations

import numpy as np


class Curriculum:
    """Stage lookup and the epsilon / safety-mask / shaping schedules."""

    def __init__(self, stages: list[dict], cfg: dict):
        self.stages = stages
        self.cfg = cfg
        self.ends = np.cumsum([s["steps"] for s in stages])
        self.total_steps = int(self.ends[-1])

    def locate(self, progress: int) -> tuple[int, float]:
        """``(stage index, fraction through that stage in [0, 1])``."""
        index = min(int(np.searchsorted(self.ends, progress, side="right")),
                    len(self.stages) - 1)
        start = 0 if index == 0 else int(self.ends[index - 1])
        frac = (progress - start) / max(self.stages[index]["steps"], 1)
        return index, float(np.clip(frac, 0.0, 1.0))

    def epsilon(self, index: int, frac: float) -> float:
        eps0, eps1 = self.stages[index]["eps"]
        return float(eps0 + (eps1 - eps0) * frac)

    def mask_level(self, progress: int) -> int:
        """Safety-curriculum level at this point of the run.

        Escaping a bomb needs four consecutive correct moves, so an untrained
        greedy policy dies within ~20 of the 400 steps and never sees the payoff
        that makes bombing worthwhile.  Shielding the policy early (level 2)
        buys full-length episodes to learn from; annealing the shield off
        (level 1, then 0) then lets the agent learn what the shield was hiding,
        so the submitted policy does not depend on it.

        Configured as ``[(start_fraction, level), ...]``.  ``mask_scope`` picks
        what the fraction is relative to: ``"run"`` shields the beginning of the
        whole curriculum, ``"stage"`` shields the beginning of *each* stage,
        which matters because a new opponent mix makes old safety knowledge
        partly wrong again.
        """
        schedule = self.cfg.get("mask_schedule")
        if not schedule:
            return int(self.cfg.get("mask_lethal", 0))
        if self.cfg.get("mask_scope", "run") == "stage":
            _, frac = self.locate(progress)
        else:
            frac = progress / max(self.total_steps, 1)
        level = schedule[0][1]
        for start, value in schedule:
            if frac >= start:
                level = value
        return int(level)

    def shaping(self, index: int, frac: float) -> float:
        """Reward-shaping scale; annealed to 0 over the tail of the last stage.

        Shaping is a training aid, and a policy that has been weaned off it is
        one whose value function is calibrated against the score the tournament
        actually pays.
        """
        anneal = self.cfg.get("shaping_anneal_frac", 0.0)
        if index == len(self.stages) - 1 and anneal > 0 and frac > 1 - anneal:
            return float(np.clip((1.0 - frac) / anneal, 0.0, 1.0))
        return 1.0
