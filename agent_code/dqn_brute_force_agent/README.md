# Brute-force DQN agent

This agent learns the unmodified four-player `classic` scenario with a
convolutional deep Q-network. It deliberately has no pathfinding, action mask,
blast simulation, scripted escape behavior, demonstrations, or curriculum.

## Observation and objective

The fixed stone border is validated and cropped. The remaining observation is
a `(10, 15, 15)` `uint8` pseudo-image containing internal walls, crates,
revealed coins, the learner, opponents, bomb presence/timers, current
explosions, and player bomb availability. Engine `(x, y)` coordinates map to
image `[y - 1, x - 1]`.

Only native score is rewarded: `+1` for collecting a coin and `+5` for killing
an opponent. All six actions remain available to the network, including
actions that the engine may reject.

The learner uses uniform replay, epsilon-greedy exploration, a target network,
Huber loss, RMSprop, and gradient clipping. Defaults are defined in
`config.py`; `PLAN.md` records the rationale and planned ablations.

## Training

Start a fresh run against three rule-based opponents:

```bash
uv run --extra torch python main.py play \
  --no-gui \
  --my-agent dqn_brute_force_agent \
  --train 1 \
  --scenario classic \
  --n-rounds 1000 \
  --save-stats results/dqn_brute_force_train.json
```

Training is fresh by default even when `checkpoint.pt` exists. Supported
environment overrides are:

| Variable | Purpose | Default |
| --- | --- | ---: |
| `DQN_RESUME` | restore model, optimizer, counters, and RNG state | `0` |
| `DQN_SAVE_REPLAY` | save/restore the separate replay artifact | `0` |
| `DQN_REPLAY_CAPACITY` | maximum replay transitions | `100000` |
| `DQN_REPLAY_WARMUP` | transitions before optimization | `10000` |
| `DQN_CHECKPOINT_INTERVAL` | rounds between checkpoints | `100` |
| `DQN_ARTIFACT_DIR` | alternate checkpoint/metrics directory | agent folder |

For an exact resume, the original capacity must be retained and both resume
and replay persistence must have been enabled. Replay persistence is off by
default because its artifact can be hundreds of megabytes.

A short integration run can override the large-run settings:

```bash
DQN_REPLAY_CAPACITY=128 \
DQN_REPLAY_WARMUP=32 \
DQN_CHECKPOINT_INTERVAL=1 \
uv run --extra torch python main.py play \
  --no-gui --my-agent dqn_brute_force_agent --train 1 --n-rounds 2 --seed 1
```

Training metrics are appended to `training_metrics.jsonl`. `checkpoint.pt` is
written atomically. Because the engine has no final shutdown callback, choose
a round count divisible by the checkpoint interval or set the interval to one
for short runs.

## Evaluation and model selection

Evaluation loads `best_model.pt` when present, otherwise `checkpoint.pt`, and
fails clearly when neither exists:

```bash
uv run --extra torch python main.py play \
  --no-gui --my-agent dqn_brute_force_agent --n-rounds 100 --seed 1001 \
  --save-stats results/dqn_brute_force_eval.json
```

After an external evaluation establishes that the latest checkpoint is the
best candidate, promote it explicitly:

```bash
uv run --extra torch python -m \
  agent_code.dqn_brute_force_agent.promote_checkpoint
```

## Tests

All agent-specific tests live inside this directory to keep repository changes
within the requested scope:

```bash
uv run --extra torch python -m unittest \
  agent_code.dqn_brute_force_agent.test_agent
uv run ruff check agent_code/dqn_brute_force_agent
```

For submission, include the Python modules and selected checkpoint. Exclude
logs, `training_metrics.jsonl`, `replay.pkl`, temporary files, and
optimizer-only training artifacts.
