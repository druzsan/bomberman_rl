"""S3 with ``gamma = 0.95`` -- the tabular agent's discount factor.

Hypothesis, from the first two deep runs. The deep agent and the tabular agent
end up within half a point of each other for opposite reasons: measured over 400
gate rounds, the deep agent takes **3.35 coins and 45 crates but only 0.42
kills**, the tabular agent **2.73 coins and 35 crates with 0.68 kills**. Since a
kill is worth five points and a coin one, the whole difference is that the deep
agent farms and the tabular agent hunts.

The discount factor explains why, and the numbers are measurable rather than
rhetorical. With ``gamma = 0.99`` the deep agent's own mean Q is ~2.4, so dying
costs it ``V + 2.0 ~ 4.4`` against a kill worth ``+1.5``: contesting an opponent
is only worth it below a **34 %** chance of dying. At ``gamma = 0.95`` the same
shaped reward rate gives ``V ~ 0.4``, dying costs ~2.4, and the same kill is
worth taking up to a **62 %** risk. The tabular agent is not braver by
temperament; it is brave because its horizon is shorter.

Setting it to 0.95 also makes S1 and S3 differ in *only* the function class --
same features available, same reward function, same curriculum, same discount --
which is the controlled comparison the report wants.

``gamma`` appears in three places that must agree, and the third is the trap:
the learner's bootstrap, the n-step return **and the potential-based shaping
term** ``gamma * phi(s') - phi(s)``.  If the shaping's gamma differs from the
learner's, the shaping is no longer potential-based and it changes the optimal
policy (Ng et al.).  A demonstration dataset therefore has to be *re-recorded*
for a different discount -- its rewards already contain the shaping term:

    uv run python -m training.bc collect --out data/expert-g95 --episodes 8000 \\
        --workers 24 --gamma 0.95 --seed 7
    uv run python -m training.bc train --data data/expert-g95 --out runs/bc/g95 \\
        --gamma 0.95 --steps 40000
    uv run python -m training.dqn_driver --config training/configs/s3_gamma95.py \\
        --tag gamma95 --override pretrained='"runs/bc/g95/model.pt"' \\
        --override expert_episodes='"data/expert-g95"'
"""

from __future__ import annotations

from lib.rewards import RewardConfig

# Absolute, not relative: the driver loads a config by file path with
# ``spec_from_file_location``, so the module has no parent package.
from training.configs.s3 import config as base_config

GAMMA = 0.95


def config() -> dict:
    cfg = base_config()
    cfg["gamma"] = GAMMA
    cfg["reward"] = RewardConfig(gamma=GAMMA, reward_scale=0.1)
    return cfg
