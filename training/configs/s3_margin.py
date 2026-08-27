"""E15-B -- a gentler DQfD margin anneal, plan §15 open question 7.

E13 §A established the failure and its mechanism.  Three runs with a margin loss
each produced **exactly one** catastrophic checkpoint, and in each case it is the
checkpoint at which the margin weight reaches zero:

    main2 (10 M)   weight hits 0 at 1.99-2.01 M   step 2 000 779   1.66 / 99 %
    gamma95 (8 M)  weight hits 0 at 1.58-1.61 M   step 1 600 360   1.90 / 100 %
    nodanger (4 M) weight hits 0 at 0.79-0.81 M   step   800 056   2.24 / 95 %

The arm with no margin loss at all has no collapse anywhere in its curve.  The
margin term holds ``Q`` down for every action the expert did not take, so while
it is on the greedy ``argmax`` is effectively pinned to the expert's choice;
removing it un-pins those values faster than temporal-difference learning can
re-calibrate them.

The fix is to remove it slowly enough that TD keeps up.  ``margin_anneal =
"cosine"`` leaves the weight at 0.0015 of its initial value at 97.5 % of the
window where the linear schedule is still at 0.025 -- 16x smaller -- and its
derivative goes to zero at the endpoint instead of jumping.  ``margin_floor``
stays at 0: a residual margin would also work, but it biases the final policy
towards a teacher that suicides 17 % of the time, forever, which is the opposite
of what the whole anneal exists for.

This is a *hygiene* experiment, not a scoring one.  Success is "no checkpoint in
the curve at 90 %+ suicide", with the final score statistically indistinguishable
from `gamma95`'s.  If the score moves at all it is a bonus and needs its own
held-out confirmation; if the collapse survives a cosine anneal, the mechanism is
not the one E13 proposes and that is worth knowing too.

Run it as::

    uv run python -m training.dqn_driver --config training/configs/s3_margin.py \\
        --tag margin --actors 16 --eval-workers 5 --steps 8000000 \\
        --override pretrained='"runs/bc/g95/model.pt"' \\
        --override expert_episodes='"data/expert-g95"' \\
        --override checkpoint_every=400000 --override eval_every=400000 \\
        --override keep_last=14

The collapse sits at ``margin_anneal_frac * steps``, so with 8 M steps and 0.20
the checkpoint to look at is 1.6 M -- exactly where `gamma95` scored 1.90.
"""

from __future__ import annotations

# Absolute, not relative: the driver loads a config by file path with
# ``spec_from_file_location``, so the module has no parent package.
from training.configs.s3_gamma95 import config as base_config


def config() -> dict:
    cfg = base_config()
    cfg["margin_anneal"] = "cosine"
    cfg["margin_floor"] = 0.0
    return cfg
