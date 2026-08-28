"""S3 -- the deep Q-network, the plan's headline model.

Rainbow-lite on 19 board planes: double, dueling, 3-step returns, prioritised
replay, D4 augmentation, legal-action masking in the behaviour policy *and* in
the target ``max``.

Deliberate deviations from ``dev/plan.md`` §S3, each with a reason:

* **``reward_scale = 0.1``.**  The event rewards were designed for the tabular
  agent, where the scale of a Q value is irrelevant; a Huber loss with
  ``delta = 1`` is not scale-free, and at the raw scale (a suicide is -20)
  every update outside a narrow band degenerates to L1.  Scaling the whole
  reward function by a constant leaves the optimal policy untouched and puts
  the Q values in the range Huber was chosen for.
* **``gamma = 0.99`` in both the learner and the reward config.**  The
  potential-based shaping term is ``gamma * phi(s') - phi(s)``; if its gamma
  disagrees with the learner's, the shaping stops being potential-based and
  starts changing the optimal policy.
* **A cosine margin anneal** rather than the linear one this config shipped
  with; see the comment on ``margin_anneal`` below, and E15-B.  The runs that
  produced the submission predate it, and their own ``config.json`` records
  what they actually used.
* **One stage, not a warm-up ladder.**  E07 measured that a curriculum stage
  which removes an action biases the value function against that action in
  every later stage.  With behaviour cloning doing the bootstrapping, a warm-up
  stage buys nothing that the expert data has not already given us.
* **A permanent 10 % solo share** in the opponent mixture, so the escape and
  crate-bombing skills cannot be forgotten once the league gets crowded.

Run it as:

    uv run python -m training.bc collect --out data/expert --episodes 3000
    uv run python -m training.bc train --data data/expert --out runs/bc/main
    uv run python -m training.dqn_driver --config training/configs/s3.py \\
        --tag main --override pretrained='"runs/bc/main/model.pt"' \\
        --override expert_episodes='"data/expert"'
"""

from __future__ import annotations

from lib.rewards import RewardConfig

GAMMA = 0.99


def stages() -> list[dict]:
    return [
        dict(name="task4-full", steps=4_000_000, scenario="classic",
             opponent_mix=[(0.30, ["rule_based_agent"] * 3),
                           (0.30, ["bfs_expert"] * 3),
                           (0.10, ["coin_collector_agent"] * 3),
                           (0.20, ["q_tabular_agent"] * 3),
                           (0.10, [])],
             eps=(0.30, 0.02),
             eval_suite={"scenario": "classic",
                         "opponents": ["bfs_expert"] * 3, "rounds": 60}),
    ]


def config() -> dict:
    return dict(
        agent="dqn_agent",
        # -- model
        plane_set="full",
        channels=64,
        blocks=6,
        # -- learning
        gamma=GAMMA,
        n_step=3,
        lr=1e-4,
        lr_end=1e-5,
        batch_size=256,
        huber_delta=1.0,
        grad_clip=10.0,
        target_sync=2_500,
        augment=True,
        amp=True,
        device="cuda",
        # -- replay
        replay_capacity=524_288,
        learn_start=25_000,
        per_alpha=0.6,
        per_beta=0.4,
        # -- DQfD: expert transitions stay resident, the margin decays away
        pretrained=None,
        expert_episodes=None,
        expert_capacity=1_000_000,
        expert_share=0.25,
        expert_anneal_frac=0.5,
        margin=0.5,
        margin_weight=1.0,
        margin_anneal_frac=0.20,
        # E15-B.  With the linear schedule, four runs out of four produced
        # exactly one 95-100 % suicide checkpoint, at precisely the step the
        # weight reaches zero; the run without a margin loss had none anywhere.
        # Cosine reaches zero at the same step with 38x less weight just before
        # it and no derivative jump, and the collapse disappears from the whole
        # curve (minimum 4.30 against 1.36-2.24) at no cost to the final score
        # (6.10 vs 6.08 over the last 4 M steps, p = 0.65).
        margin_anneal="cosine",
        margin_floor=0.0,
        # -- actors
        actors=20,
        eps_ladder=True,
        train_ratio=0.125,
        grad_chunk=8,
        broadcast_every=200,
        queue_size=256,
        # -- safety curriculum (S1's lesson: shield exploration, then wean off)
        mask_schedule=[(0.0, 1), (0.6, 0)],
        mask_scope="run",
        ship_mask=False,
        # -- rewards
        reward=RewardConfig(gamma=GAMMA, reward_scale=0.1),
        shaping_anneal_frac=0.0,
        # -- self-play league (off until the policy is worth practising against)
        league_after=0,
        league_size=6,
        league_share=0.25,
        league_min_score=4.0,
        # -- bookkeeping
        stages=stages(),
        eval_workers=6,
        eval_rounds=100,
        log_every=25_000,
        checkpoint_every=200_000,
        eval_every=200_000,
        keep_last=8,
        save_train_state=True,
        seed=1,
    )
