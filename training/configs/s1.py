"""S1 -- tabular Q-learning, full four-stage curriculum.

Configs are python modules rather than YAML so they can compute derived values
and carry a real ``RewardConfig`` object; ``--override key=json`` patches any
top-level key from the command line and the patch is recorded in
``config.json``.
"""

from __future__ import annotations

from lib.rewards import RewardConfig

GAMMA = 0.95


def stages() -> list[dict]:
    return [
        # 1. Navigation only. BOMB is removed from the action set, so the agent
        #    cannot die and the whole budget goes into "walk to the coin".
        dict(name="task1-coins", steps=400_000, scenario="coin-heaven",
             opponent_mix=[(1.0, [])], allow_bomb=False, eps=(1.0, 0.15),
             eval_suite={"scenario": "coin-heaven", "opponents": [], "rounds": 40}),
        # 2. Crates and bomb safety -- the stage that decides tournament strength.
        dict(name="task2-crates", steps=2_000_000, scenario="loot-crate",
             opponent_mix=[(1.0, [])], allow_bomb=True, eps=(0.5, 0.10),
             eval_suite={"scenario": "loot-crate", "opponents": [], "rounds": 40}),
        # 3. Weak opponents: learn that other agents exist and can be hunted.
        dict(name="task3-weak", steps=2_000_000, scenario="classic",
             opponent_mix=[(0.5, ["peaceful_agent"] * 3),
                           (0.5, ["coin_collector_agent"] * 3)],
             eps=(0.3, 0.08),
             eval_suite={"scenario": "classic",
                         "opponents": ["coin_collector_agent"] * 3, "rounds": 60}),
        # 4. The real thing. A mixture, not just rule_based_agent: that opponent
        #    suicides in half its rounds, which is very exploitable and very
        #    non-transferable.
        dict(name="task4-full", steps=6_000_000, scenario="classic",
             opponent_mix=[(0.50, ["rule_based_agent"] * 3),
                           (0.20, ["coin_collector_agent"] * 3),
                           (0.15, ["bfs_expert"] * 3),
                           (0.15, ["q_tabular_agent"] * 3)],
             eps=(0.2, 0.03)),
    ]


def config() -> dict:
    return dict(
        agent="q_tabular_agent",
        feature_set="TQ-M",
        fold=True,
        gamma=GAMMA,
        n_step=1,             # n-step returns without importance correction blame
                              # epsilon-random deaths on the escape action (E04)
        alpha=0.1,            # 0 would select Robbins-Monro 1/(1+N)**alpha_power
        alpha_end=0.01,       # geometric decay of the step size over the run
        alpha_power=0.6,
        export_average_beta=0.05,   # Polyak average of the shipped table
        algo=0,               # 0 Q-learning, 1 Expected SARSA
        # Safety curriculum, see Driver.mask_level. The shield is fully off for
        # the last 20 % of training, so the submitted policy stands on its own.
        mask_schedule=[(0.0, 2), (0.5, 1), (0.8, 0)],
        reward=RewardConfig(gamma=GAMMA),
        stages=stages(),
        actors=20,
        eval_workers=6,
        eval_rounds=100,
        log_every=50_000,
        checkpoint_every=250_000,
        eval_every=250_000,
        # Annealing the shaping to zero is a separate experiment (E06); the
        # potential term cannot change the optimal policy anyway.
        shaping_anneal_frac=0.0,
        seed=1,
    )
