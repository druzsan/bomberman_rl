"""E15-A -- kill-weighted reward at fixed ``gamma``, plan §15 open question 6.

E12 predicted the deep agent farms crates because its horizon is too long, and
the *value-scale* half of that prediction was confirmed exactly: ``q_mean`` fell
2.4 -> 0.43 when ``gamma`` went 0.99 -> 0.95.  The *behavioural* half was not.
Kills moved only 0.42 -> 0.48 against the tabular agent's 0.68, and coins rose
too, so most of the +0.4 points is easier credit assignment rather than any new
appetite for risk.

This is the separating experiment.  ``gamma`` is held at 0.95 -- identical to the
shipped model -- and the *only* change is the price of a kill:

    KILLED_OPPONENT   15.0 -> 30.0   (1.5 -> 3.0 after reward_scale = 0.1)

Everything conditioned on *approaching* a kill is left alone
(``BOMB_NEAR_OPPONENT`` 2.0, ``TRAPPED_OPPONENT`` 5.0), because moving those too
would confound "the agent wants kills more" with "the agent is guided towards
kills more", and the question is about appetite.

Read the outcome on two numbers, not one:

* **kills/round up, score flat or down** -- 0.5 kills/round is what a
  survival-first policy gets, the shaped optimum is already where the engine's
  optimum is, and there is nothing more on the table.  Ship nothing.
* **kills/round up *and* score up** -- the shaping was underpricing the engine's
  own ``REWARD_KILL = 5``, and there is another point to take.

The engine pays 5 for a kill and 1 for a coin.  At ``reward_scale = 0.1`` the
default event rewards price them 1.5 and 0.3, i.e. a ratio of 5:1 -- faithful.
Doubling the kill deliberately *overprices* it against the true objective, which
is the point: it is a probe for appetite, not a candidate for shipping as-is.
Score is measured on the real engine reward either way, so the probe cannot
flatter itself.

Run it as::

    uv run python -m training.dqn_driver --config training/configs/s3_kill.py \\
        --tag kill --actors 16 --eval-workers 5 --steps 8000000 \\
        --override pretrained='"runs/bc/g95/model.pt"' \\
        --override expert_episodes='"data/expert-g95"' \\
        --override checkpoint_every=400000 --override eval_every=400000 \\
        --override keep_last=14
"""

from __future__ import annotations

from lib.rewards import DEFAULT_EVENT_REWARDS, KILLED_OPPONENT, RewardConfig

# Absolute, not relative: the driver loads a config by file path with
# ``spec_from_file_location``, so the module has no parent package.
from training.configs.s3_gamma95 import GAMMA
from training.configs.s3_gamma95 import config as base_config

KILL_REWARD = 30.0


def config() -> dict:
    cfg = base_config()
    rewards = dict(DEFAULT_EVENT_REWARDS)
    rewards[KILLED_OPPONENT] = KILL_REWARD
    cfg["reward"] = RewardConfig(gamma=GAMMA, reward_scale=0.1,
                                 event_rewards=rewards)
    return cfg
