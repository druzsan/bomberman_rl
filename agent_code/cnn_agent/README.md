# Factorized CNN agent

This is the first reference implementation selected from `dev/plan.md`:

- allocentric 15x15 playable field;
- 12 factorized spatial planes plus `can_bomb` and `step / 400`;
- small residual CNN with a flattened spatial head;
- legal-action-masked, five-step Double DQN;
- replay-time dihedral augmentation;
- standalone callback-free simulator and shared-policy collection;
- R2 event rewards, without navigation or safety potentials.

The tournament callback imports only `model.py`, `state.py`, and `policy.py`.
Training does not use the framework training callbacks.

# Training runbook

Run every command below from the repository root. Keep every seed and
configuration in its own run directory; reusing a directory can mix metrics or
replace snapshots with the same round number.

## 1. Install and verify the GPU

```bash
uv sync --group dev --extra torch
```

```bash
uv run python -c "import torch; print('torch:', torch.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Do not start with `--device cuda` unless this prints `CUDA available: True`.
The CLI reference is available with:

```bash
uv run --group dev python -m agent_code.cnn_agent.standalone_training --help
uv run --group dev python -m agent_code.cnn_agent.standalone_training train --help
uv run --group dev python -m agent_code.cnn_agent.standalone_training resume --help
```

## 2. Validate the implementation

```bash
uv run --group dev python -m unittest discover \
  -s agent_code/cnn_agent/tests -v
```

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training smoke --seed 0
```

## 3. Start coin training from scratch

This curriculum disables bombs, adds symmetric BFS progress toward the nearest
visible coin, and applies a small time cost. Since bomb-disabled rounds often
last 400 steps, 2,000 rounds can already provide about 800,000 transitions.

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training train \
  --device cuda \
  --scenario coin-heaven \
  --players 1 \
  --rounds 2000 \
  --seed 42 \
  --learning-rate 0.00025 \
  --gamma 0.99 \
  --n-step 5 \
  --batch-size 512 \
  --replay-capacity 100000 \
  --replay-warmup 10000 \
  --train-every 4 \
  --target-update-every 2000 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 200000 \
  --checkpoint-every 100 \
  --evaluation-every 500 \
  --evaluation-rounds 200 \
  --evaluation-seed 10000 \
  --disable-bombs \
  --coin-progress-reward 0.05 \
  --time-penalty 0.001 \
  --output-dir agent_code/cnn_agent/runs/coins_bfs_seed_42
```

Automatic evaluation is greedy (`epsilon = 0`) and uses the same 200 held-out
rounds every time. This run evaluates at rounds 500, 1,000, 1,500, and 2,000.
Do not interrupt while an evaluation is running.

For a strict reward ablation, first run the same command without
`--coin-progress-reward` and `--time-penalty`, using a separate directory such
as `coins_no_bomb_seed_42`. Never reuse the old `coins_gpu_seed_42` directory.

## 4. Monitor the run

Training rounds, including exploratory actions:

```bash
tail -f agent_code/cnn_agent/runs/coins_bfs_seed_42/metrics.jsonl
```

Completed held-out evaluations:

```bash
tail -f agent_code/cnn_agent/runs/coins_bfs_seed_42/evaluations.jsonl
```

Readable evaluation history:

```bash
python -m json.tool --json-lines \
  agent_code/cnn_agent/runs/coins_bfs_seed_42/evaluations.jsonl
```

The old manually maintained `eval.json` is not read or written by the trainer.
Use `evaluations.jsonl` for automatic results.

## 5. Understand the artifacts

- `config.json`: effective configuration for the current run;
- `metrics.jsonl`: one exploratory training record per completed round;
- `evaluations.jsonl`: one immutable held-out result per evaluation;
- `latest.pt`: most recent resumable training state;
- `checkpoints/round_XXXXXXXX.pt`: resumable state matching an evaluation;
- `models/round_XXXXXXXX.pt`: lightweight inference model matching that result;
- `model.pt`: most recently evaluated lightweight model.

`cnn_agent.log` belongs to official-framework matches, not standalone training.
Replay memory is not serialized. A resumed run restores the model, target,
optimizer, counters, epsilon schedule, and available RNG state, but collects a
fresh replay buffer and waits for `replay_warmup` before updating again.

## 6. Stop safely and resume

Press `Ctrl-C` once to save `latest.pt` after the last fully completed training
round. An interrupted, unevaluated state does not replace the versioned models.

Inspect the actual saved round before calculating `--additional-rounds`:

```bash
uv run python -c "import torch; c=torch.load(\
'agent_code/cnn_agent/runs/coins_bfs_seed_42/latest.pt', \
map_location='cpu', weights_only=True); \
print({k: c.get(k) for k in ('round', 'transitions_seen', 'updates')})"
```

`--additional-rounds` is counted from that checkpoint round, not from the value
currently written in `config.json`. Example: if the checkpoint says round 1,500
and the desired total is 2,500, pass 1,000.

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training resume \
  --run-dir agent_code/cnn_agent/runs/coins_bfs_seed_42 \
  --additional-rounds 1000 \
  --device cuda \
  --batch-size 512 \
  --evaluation-every 500 \
  --evaluation-rounds 200 \
  --evaluation-seed 10000
```

## 7. Coin-stage stop criteria

Never select a model from training loss or exploratory training return alone.
Continue coin training until all prerequisites hold:

1. `transitions_seen` exceeds the configured 200,000-step epsilon decay and
   evaluation therefore tests a policy trained after epsilon reached 0.05.
2. At least three consecutive automatic evaluations are available on the same
   seeds.
3. Mean held-out coins/score has stopped improving materially across those
   evaluations, while suicides do not increase.
4. The agent no longer times out at 400 steps in nearly every coin round.

A useful promotion target is approximately 35--40 of the 50 available coins per
round, no more than 0.05 suicides per round, and falling mean steps per coin.
These are gates for this curriculum, not guaranteed outcomes.

Stop early and diagnose instead of spending more compute when any of these hold
for three evaluations:

- mean coins remain flat or fall by more than about 5%;
- almost every round still reaches 400 steps;
- suicides rise while shaped training return improves;
- loss becomes non-finite (`NaN` or infinity).

In those cases inspect replays and revisit reward shaping, exploration, or the
curriculum. More rounds alone are unlikely to fix the behavior.

## 8. Compare and select historical coin checkpoints

Print the evaluation history, choose by `mean_coins`, `mean_suicides`, and
`mean_steps`, and manually confirm the candidate using the identical seed suite:

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training evaluate \
  --device cuda \
  --model agent_code/cnn_agent/runs/coins_bfs_seed_42/models/round_00001500.pt \
  --scenario coin-heaven \
  --players 1 \
  --rounds 200 \
  --seed 10000 \
  --disable-bombs
```

The coin experiment validates representation, navigation, and optimization. The
current trainer does not yet transfer this model into a different scenario, so
classic training below is an independent run from scratch.

## 9. Start classic four-player training

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training train \
  --device cuda \
  --scenario classic \
  --players 4 \
  --rounds 10000 \
  --seed 0 \
  --learning-rate 0.00025 \
  --gamma 0.99 \
  --n-step 5 \
  --batch-size 512 \
  --replay-capacity 100000 \
  --replay-warmup 20000 \
  --train-every 4 \
  --target-update-every 5000 \
  --epsilon-start 1.0 \
  --epsilon-end 0.05 \
  --epsilon-decay-steps 500000 \
  --checkpoint-every 250 \
  --evaluation-every 2500 \
  --evaluation-rounds 200 \
  --evaluation-seed 20000 \
  --output-dir agent_code/cnn_agent/runs/classic_gpu_seed_0
```

Classic evaluation command for a historical candidate:

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training evaluate \
  --device cuda \
  --model agent_code/cnn_agent/runs/classic_gpu_seed_0/models/round_00007500.pt \
  --scenario classic \
  --players 4 \
  --rounds 500 \
  --seed 20000
```

## 10. Classic-stage stop criteria

Do not stop before epsilon reaches 0.05 and at least three comparable held-out
evaluations exist. Prefer checkpoints with higher true `mean_score` and
`mean_kills`, then use lower `mean_suicides` and lower variance across official
matches as tie-breakers. Stop when score and kills plateau for three evaluations
without a compensating reduction in suicides.

Reject later checkpoints when shaped training return rises but held-out score or
kills fall. Self-play evaluation alone is not sufficient: the final selection
must also perform against the supplied agents through the official framework.

## 11. Promote and test the selected classic model

Promotion overwrites `agent_code/cnn_agent/model.pt`, which is the exact path
loaded by tournament callbacks:

```bash
uv run --group dev python \
  -m agent_code.cnn_agent.standalone_training promote \
  --model agent_code/cnn_agent/runs/classic_gpu_seed_0/models/round_00007500.pt
```

Mixed benchmark:

```bash
uv run python main.py play \
  --agents cnn_agent random_agent peaceful_agent coin_collector_agent \
  --scenario classic \
  --n-rounds 200 \
  --seed 30000 \
  --no-gui \
  --save-stats results/cnn_agent_mixed_seed_30000.json
```

Three rule-based opponents:

```bash
uv run python main.py play \
  --my-agent cnn_agent \
  --scenario classic \
  --n-rounds 200 \
  --seed 40000 \
  --no-gui \
  --save-stats results/cnn_agent_rule_based_seed_40000.json
```

Compatibility smoke test:

```bash
uv run python -m unittest test.py -v
```

Never pass `--train` during evaluation. An absent promoted
`agent_code/cnn_agent/model.pt` produces an untrained fallback policy and is not
a valid result.

## 12. GPU memory and batch-size guidance

Use batch size 512 as the main GPU configuration. Benchmark 1,024 in a separate
seed only after 512 is stable; keep the learning rate at 0.00025 so batch size is
the only changed variable. A larger batch can improve update throughput but may
reduce learning diversity, so select it by wall-clock held-out improvement, not
by memory occupancy.

Replay stays in system RAM: 100,000 transitions use roughly 0.6--1 GB of host
memory. Increasing replay capacity does not consume free GPU memory. Simulation
is CPU-bound, and the CNN is small, so low GPU utilization is normal. Four
shared-policy players use one batched inference call; single-player coin
training still performs inference with a batch of one.

After settling the configuration with seed 0, train final comparison runs with
seeds 1 and 2 in new directories. Report the mean and variation across seeds
instead of selecting a method from one lucky run.
