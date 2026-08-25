"""Task 1 -- navigation only (`coin-heaven`, solo, BOMB removed).

Kept as its own run rather than as the first stage of the curriculum: warm
starting Task 2 from it costs 38 coins (E07).
"""

from __future__ import annotations

from lib.rewards import RewardConfig

GAMMA = 0.95


def config() -> dict:
    return dict(
        agent="q_tabular_agent",
        feature_set="TQ-M",
        fold=True,
        gamma=GAMMA,
        n_step=3,
        alpha=0.1,
        alpha_end=0.01,
        alpha_power=0.6,
        export_average_beta=0.05,
        algo=0,
        mask_schedule=[(0.0, 0)],
        mask_scope="stage",
        reward=RewardConfig(gamma=GAMMA),
        stages=[
            dict(name="task1-coins", steps=600_000, scenario="coin-heaven",
                 opponent_mix=[(1.0, [])], allow_bomb=False, eps=(1.0, 0.05),
                 eval_suite={"scenario": "coin-heaven", "opponents": [], "rounds": 60}),
        ],
        actors=20,
        eval_workers=4,
        eval_rounds=60,
        log_every=50_000,
        checkpoint_every=100_000,
        eval_every=100_000,
        shaping_anneal_frac=0.0,
        seed=1,
    )
