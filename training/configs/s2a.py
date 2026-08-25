"""S2a -- linear Q-function approximation on the same features.

The second model. Identical state analysis, identical curriculum and identical
harness as the tabular agent, so the comparison is a controlled one: only the
function class changes, from ~600 independent table entries to ~290 weights.

Two deliberate differences, both forced by the function class:

* **Expected SARSA instead of Q-learning.** Off-policy bootstrapping with
  function approximation is the "deadly triad"; with shared weights one
  overestimated action drags every state with it. Reported as a finding rather
  than discovered at 3 a.m.
* **A larger step size.** Each update touches ~17 active features instead of one
  table cell, and the step is normalised by ||phi||^2.
"""

from __future__ import annotations

from lib.rewards import RewardConfig

GAMMA = 0.95


def stages() -> list[dict]:
    return [
        dict(name="task2-crates", steps=2_000_000, scenario="loot-crate",
             opponent_mix=[(1.0, [])], allow_bomb=True, eps=(0.6, 0.10),
             eval_suite={"scenario": "loot-crate", "opponents": [], "rounds": 40}),
        dict(name="task3-weak", steps=2_000_000, scenario="classic",
             opponent_mix=[(0.4, ["peaceful_agent"] * 3),
                           (0.6, ["coin_collector_agent"] * 3)],
             eps=(0.4, 0.08),
             eval_suite={"scenario": "classic",
                         "opponents": ["coin_collector_agent"] * 3, "rounds": 60}),
        dict(name="task4-full", steps=12_000_000, scenario="classic",
             opponent_mix=[(0.30, ["rule_based_agent"] * 3),
                           (0.10, ["coin_collector_agent"] * 3),
                           (0.35, ["bfs_expert"] * 3),
                           (0.25, ["q_linear_agent"] * 3)],
             eps=(0.3, 0.03)),
    ]


def config() -> dict:
    return dict(
        agent="q_linear_agent",
        feature_set="TQ-M",
        fold=True,
        gamma=GAMMA,
        n_step=3,             # inverts vs n=1 once exploration is shielded (E04.4)
        alpha=0.25,
        alpha_end=0.02,
        alpha_power=0.6,
        export_average_beta=0.05,
        algo=1,               # Expected SARSA
        mask_schedule=[(0.0, 2), (0.5, 1), (0.75, 0)],
        mask_scope="stage",
        reward=RewardConfig(gamma=GAMMA),
        stages=stages(),
        actors=20,
        eval_workers=6,
        eval_rounds=100,
        log_every=50_000,
        checkpoint_every=250_000,
        eval_every=250_000,
        shaping_anneal_frac=0.0,
        seed=1,
    )
