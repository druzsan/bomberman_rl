"""S1b -- the tuned tabular curriculum.

Differences from :mod:`training.configs.s1`, each backed by a measurement in
``dev/experiments/``:

* survivability features and the safety shield model opponents as obstacles
  that *move* (E02/E05), which is what fixed `bfs_expert`'s suicide rate;
* the safety shield is scheduled **per stage**, not per run: a new opponent mix
  makes previously-safe habits wrong again, so each stage gets a shielded start
  and an unshielded finish;
* three-step returns instead of one (E04.4): 48.6 vs 10.9 coins on Task 2;
* the bomb-free navigation stage is **gone**. It converges to 50/50 coins in
  260 k steps, but warm-starting Task 2 from it caps Task 2 at 10.9 coins where
  training Task 2 directly reaches 48.95 (E07): the navigation stage inflates
  the value of every move action while BOMB, which is disabled there, stays at
  its initial 0 and is never chosen greedily afterwards. Task 1 is still run and
  reported, as its own gate, from ``s1_task1.py``.
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
        dict(name="task4-full", steps=8_000_000, scenario="classic",
             opponent_mix=[(0.50, ["rule_based_agent"] * 3),
                           (0.20, ["coin_collector_agent"] * 3),
                           (0.15, ["bfs_expert"] * 3),
                           (0.15, ["q_tabular_agent"] * 3)],
             eps=(0.3, 0.03)),
    ]


def config() -> dict:
    return dict(
        agent="q_tabular_agent",
        feature_set="TQ-M",
        fold=True,
        gamma=GAMMA,
        n_step=3,             # inverts vs n=1 once exploration is shielded (E04.4)
        alpha=0.1,
        alpha_end=0.01,
        alpha_power=0.6,
        export_average_beta=0.05,
        algo=0,
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
