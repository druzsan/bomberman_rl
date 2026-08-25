"""Task-2 only (loot-crate, solo): the stage that decides tournament strength.

Isolated so hyperparameters can be A/B-ed in about a minute per arm -- a solo
stage runs at ~25k env steps/s on 20 actors.
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
        alpha_end=0.01,       # geometric decay of the step size over the run
        alpha_power=0.6,
        export_average_beta=0.05,   # Polyak average of the shipped table
        algo=0,
        # Safety curriculum: (fraction of the run, mask level).
        # 2 = shield the whole policy, 1 = shield exploration only, 0 = off.
        mask_schedule=[(0.0, 2), (0.5, 1), (0.8, 0)],
        reward=RewardConfig(gamma=GAMMA),
        stages=[
            dict(name="task2-crates", steps=2_000_000, scenario="loot-crate",
                 opponent_mix=[(1.0, [])], allow_bomb=True, eps=(0.6, 0.05),
                 eval_suite={"scenario": "loot-crate", "opponents": [], "rounds": 40}),
        ],
        actors=20,
        eval_workers=5,
        eval_rounds=60,
        log_every=50_000,
        checkpoint_every=250_000,
        eval_every=250_000,
        shaping_anneal_frac=0.0,
        seed=1,
    )
