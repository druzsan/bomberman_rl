"""Task 1 then Task 2, to isolate what the bomb-free navigation stage does.

Same total budget as ``s1_task2`` so the arms are comparable.
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
        n_step=1,
        alpha=0.1,
        alpha_end=0.01,
        alpha_power=0.6,
        export_average_beta=0.05,
        algo=0,
        mask_schedule=[(0.0, 2), (0.5, 1), (0.8, 0)],
        mask_scope="stage",
        reward=RewardConfig(gamma=GAMMA),
        stages=[
            dict(name="task1-coins", steps=400_000, scenario="coin-heaven",
                 opponent_mix=[(1.0, [])], allow_bomb=False, eps=(1.0, 0.15),
                 eval_suite={"scenario": "coin-heaven", "opponents": [], "rounds": 40}),
            dict(name="task2-crates", steps=3_600_000, scenario="loot-crate",
                 opponent_mix=[(1.0, [])], allow_bomb=True, eps=(0.6, 0.10),
                 eval_suite={"scenario": "loot-crate", "opponents": [], "rounds": 40}),
        ],
        actors=20,
        eval_workers=3,
        eval_rounds=40,
        log_every=50_000,
        checkpoint_every=250_000,
        eval_every=250_000,
        shaping_anneal_frac=0.0,
        seed=1,
    )
