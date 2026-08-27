"""E15-C -- a ``coin-heaven`` share in the curriculum, plan §15 open question 8.

The shipped deep agent fails gate Task 1 at **40.08 of 50 coins** on
`coin-heaven`, where the tabular agent takes all 50 in 125 steps.  It survives
every round and never suicides; it simply does not finish.  The cause is
distribution, not capability: its entire curriculum is `classic`, `coin-heaven`
has no crates at all, and a policy tuned to interleave crate-bombing with coin
collection has never seen a board where the only thing to do is sprint.  The
tabular agent is immune because its ``target_dir`` feature is a hand-written BFS
to the nearest coin, and is therefore scenario-independent by construction.

**The reason this was not simply done at the end of the last session** is E07:
a curriculum stage that *removes an action* biases the value function against
that action in every later stage, and it cost this project 38 coins on Task 2
once already.  The open question is whether a *scenario* share behaves better
than an *action-restricted* stage did.  The argument that it should: nothing is
disabled here.  All six actions stay legal in `coin-heaven`, ``BOMB`` is merely
useless there, and the share is permanent rather than a phase the policy leaves
behind -- so there is no point in training at which the action distribution
shifts underneath the value function.  That is an argument, not a measurement,
which is why it is an experiment.

The share is 10 % and **solo**, because solo is what Task 1 measures and what the
skill actually is.  It is expressed through ``scenario_mix`` rather than a second
stage, deliberately: a stage is a phase the curriculum passes through, a mix is a
distribution it never leaves.  ``opponent_mix`` is untouched, so the self-play
league (which rewrites it in place) is unaffected.

Two numbers decide it, and the second is the one that matters:

* **Task 1 coins**, which should go 40 -> high 40s.  If it does not, a scenario
  share is not enough and the honest answer stays "this is the cost of learning
  features instead of designing them".
* **Task 4 score against 3 x rule_based_agent**, which must *not* move.  10 % of
  episodes spent on a board with no crates and no opponents is 10 % fewer
  `classic` episodes; if that costs tournament strength, the trade is a bad one,
  because the tournament is `classic` and Task 1 is a diagnostic.

Run it as::

    uv run python -m training.dqn_driver --config training/configs/s3_coinheaven.py \\
        --tag coinheaven --actors 16 --eval-workers 5 --steps 8000000 \\
        --override pretrained='"runs/bc/g95/model.pt"' \\
        --override expert_episodes='"data/expert-g95"' \\
        --override checkpoint_every=400000 --override eval_every=400000 \\
        --override keep_last=14
"""

from __future__ import annotations

# Absolute, not relative: the driver loads a config by file path with
# ``spec_from_file_location``, so the module has no parent package.
from training.configs.s3_gamma95 import config as base_config

COIN_HEAVEN_SHARE = 0.10


def config() -> dict:
    cfg = base_config()
    stage = dict(cfg["stages"][-1])
    stage["scenario_mix"] = [
        (1.0 - COIN_HEAVEN_SHARE, "classic", None),
        (COIN_HEAVEN_SHARE, "coin-heaven", []),
    ]
    cfg["stages"] = cfg["stages"][:-1] + [stage]
    return cfg
