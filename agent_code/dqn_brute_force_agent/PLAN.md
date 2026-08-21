# Brute-force DQN implementation plan

## 1. Goal and boundaries

Build a baseline DQN agent that learns the unmodified `classic` scenario from
the beginning. The policy receives a direct, spatial encoding of the supplied
game state and learns its own features with a convolutional network.

For this baseline:

- train on the 17 x 17 `classic` board from round one;
- use all six actions: `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, and `BOMB`;
- do not add pathfinding, blast-line calculation, nearest-target directions,
  action masks, scripted escapes, demonstrations, or a task curriculum;
- do not modify the game engine for behavior required by the submitted agent;
- keep inference self-contained in `agent_code/dqn_brute_force_agent/` and
  below the tournament's 0.5 second CPU limit;
- begin with one learning agent against three `rule_based_agent` opponents.

This is intentionally a brute-force baseline. Failure to learn reliably is a
useful experimental result and a reference point for later improvements.

## 2. Findings that constrain the design

- `callbacks.setup` runs once, followed by `train.setup_training` in training
  mode. The same context object persists across rounds.
- `act` is timed during evaluation, but training callbacks have no time limit.
- A state contains a 17 x 17 field plus visible coins, players, bombs with
  timers, the dangerous explosion map, and bomb availability. The field's
  outermost tiles are permanent stone walls in every scenario.
- `game_events_occurred` is skipped when the learner dies. In that case,
  `end_of_round` is the only place that exposes the fatal transition.
- For a surviving learner, the final step is first sent to
  `game_events_occurred` and then repeated in `end_of_round`. It must be marked
  terminal without inserting it twice.
- Running several trainable copies of the same agent creates independent
  callback contexts that would all write the same checkpoint. Do not use
  `--train 2`, `3`, or `4` until checkpoint and parameter sharing are designed.
- Hidden coins are genuinely unobservable. Only revealed coins belong in the
  input.
- Official inference is single-threaded in spirit, CPU-only, limited to 8 GB,
  and must depend only on files shipped in the agent directory.

## 3. Lock down the baseline specification

Before implementation, record the following choices in `README.md` and a
single configuration object so runs are reproducible.

### Observation tensor

First verify that all four edges of `game_state["field"]` contain only stone
walls (`-1`). Then discard those fixed edges and encode the playable 15 x 15
interior as a `uint8` tensor with shape `(10, 15, 15)`. The check should fail
with a clear error if a future engine version changes this invariant.

Convert the engine's `[x, y]` arrays to image indexing `[y - 1, x - 1]`, so
moving `UP` decreases the image row. No entity should occur on the discarded
border; reject an input that violates that invariant instead of silently
dropping it.

| Channel | Contents |
| --- | --- |
| 0 | internal stone wall mask (the discarded outer wall is implicit) |
| 1 | crate mask |
| 2 | revealed coin mask |
| 3 | learner position |
| 4 | opponent positions |
| 5 | bomb occupancy |
| 6 | bomb timer encoded as `timer + 1` (preserves bombs at timer zero) |
| 7 | current dangerous explosion map |
| 8 | learner bomb availability at the learner position |
| 9 | opponent bomb availability at opponent positions |

Normalize channels to `[0, 1]` immediately before the network. Internal walls
arrive through `field`; the tensor extent represents the known, impassable
outer border. Cropping reduces each channel's spatial storage and convolution
work by about 22% without discarding any playable tile. Names, `user_input`,
round number, score, and step are omitted because they do not change the local
transition dynamics of this first baseline. One frame is enough to expose the
relevant dynamics because bomb timers and explosions are already part of the
state; four-frame Atari stacking would mostly duplicate information.

This is a state abstraction rather than a literal rendering. No derived safety
or target features are allowed in the baseline.

### Reward

Start with only environment score rewards:

- `COIN_COLLECTED`: `+1`;
- `KILLED_OPPONENT`: `+5`;
- every other event: `0`.

Do not add movement, crate, invalid-action, bombing, survival, or death shaping
to the initial run. Record both the raw episode return and score. Use unclipped
rewards first so that a kill remains worth five coins, as required by the game.
Reward clipping to `{-1, 0, +1}`, as used in the Atari paper, is the first
planned ablation because it changes the objective in Bomberman.

### DQN update

Use uniform experience replay and epsilon-greedy Q-learning as in the supplied
paper:

`target = reward` for a terminal transition, otherwise
`target = reward + gamma * max_a Q(next_state, a)`.

The implementation baseline should use a periodically copied target network.
This is a deliberate stability addition to the 2013 preprint and must be
reported as such. A strict single-network target is an ablation, not the
default, because it is more likely to diverge.

Initial hyperparameters (centralized in `config.py`):

| Parameter | Initial value |
| --- | ---: |
| discount `gamma` | 0.99 |
| replay capacity | 100,000 transitions |
| replay warm-up | 10,000 transitions |
| minibatch size | 32 |
| optimizer | RMSprop |
| learning rate | 0.00025 |
| gradient norm limit | 10 |
| train frequency | every 4 environment steps |
| target copy frequency | every 10,000 optimizer steps |
| epsilon start / end | 1.0 / 0.1 |
| epsilon decay | 1,000,000 environment steps, linear |
| evaluation epsilon | 0.05 |
| checkpoint interval | every 100 rounds |

Store replay states compactly as `uint8`; converting 100,000 pairs of states
to `float32` in advance would waste several gigabytes. Revisit capacity after
measuring actual resident memory.

### Network

Use one forward pass to produce all six Q-values:

1. `Conv2d(10, 32, kernel_size=3, stride=1, padding=1)` + ReLU;
2. `Conv2d(32, 64, kernel_size=3, stride=2, padding=1)` + ReLU;
3. `Conv2d(64, 64, kernel_size=3, stride=2, padding=1)` + ReLU;
4. flatten + `Linear(..., 256)` + ReLU;
5. `Linear(256, 6)` with no output activation.

The smaller kernels adapt the paper's 84 x 84 Atari CNN to the cropped 15 x 15
playable area. Set PyTorch CPU thread counts to one in inference and benchmark
the complete `act` callback, including encoding and model loading effects.

## 4. Implement in small, testable layers

### Step 1: shared constants and model

Add:

- `config.py`: action order, channel order, hyperparameters, local file names,
  and deterministic seeds;
- `model.py`: `DQN` only, with shape validation and no engine imports;
- `state.py`: pure `encode_state(game_state) -> np.ndarray`.

Keep engine-facing typing in `types.py` and extend `AgentContext` with the model,
device, random generator, replay/training counters, optimizer, and checkpoint
state used by the callbacks.

### Step 2: inference callback

Implement `callbacks.py` so that:

1. `setup` selects CPU, limits Torch threads, constructs the network, and loads
   a versioned checkpoint if one exists;
2. evaluation fails clearly if the trained checkpoint is absent instead of
   silently submitting a random policy;
3. training may either resume a checkpoint or initialize from a known seed,
   controlled by one explicit configuration flag;
4. `act` encodes the state, chooses a uniform random action with probability
   epsilon during training, and otherwise returns the argmax Q action under
   `torch.inference_mode()`;
5. no legality filter changes the learned policy's action distribution.

Use paths based on `Path(__file__).parent`, never repository-root assumptions.

### Step 3: replay memory and optimizer

Add `replay.py` containing a bounded ring buffer of
`(state, action_index, reward, next_state, done)`. It should sample uniformly
with the agent's seeded RNG.

Implement in `train.py`:

1. initialize the replay buffer, optimizer, target model, epsilon schedule, and
   counters in `setup_training`;
2. translate the action string to its fixed index;
3. derive the sparse score reward from events;
4. insert each transition once;
5. after warm-up, sample a minibatch at the configured frequency;
6. gather `Q(s, action)`, calculate detached Bellman targets, optimize Huber
   loss, clip gradients, and periodically copy the target network;
7. decay epsilon by environment transitions, not rounds;
8. save model, optimizer, epsilon/counters, configuration version, and RNG
   state so a run can truly resume.

The paper writes a squared TD loss. Use Huber loss as a numerical-stability
default and make mean-squared loss a named ablation.

### Step 4: terminal-transition correctness

Track the identity of the last inserted transition using `(round, step)`.

- If `game_events_occurred` runs, add the ordinary transition from
  `old_game_state` to `new_game_state`.
- If the learner dies, `game_events_occurred` does not run; add the transition
  from `last_game_state` with `next_state=None` and `done=True` in
  `end_of_round`.
- If the learner survives to a round ending, do not append the same transition
  again. Patch the already inserted last transition to `done=True`, preserving
  its observed successor state and reward.

Add assertions/logging for unexpected callback order. This is required to
prevent both missing deaths and duplicate final rewards.

### Step 5: checkpoint and metrics

Write checkpoints atomically (temporary file followed by replace), retaining a
`best` evaluation checkpoint separately from the latest resumable checkpoint.
Do not serialize the replay buffer into the tournament model; optionally save
it as a separate training-only artifact.

Append machine-readable round metrics to a CSV or JSONL file:

- environment step and optimizer step;
- epsilon, episode return, score, and lifetime length;
- coins, kills, deaths/suicides, invalid actions, and bombs;
- mean loss, mean/max Q, and gradient norm over the round;
- wall-clock training and inference time.

Keep logging aggregated per round; per-step INFO logging can dominate fast
headless training.

## 5. Verification before a long run

Add focused `unittest` tests inside the agent folder as `test_agent.py`, keeping
all implementation changes within the requested directory:

1. encoder shape, dtype, value ranges, border validation and cropping,
   `(x, y)` to `[y - 1, x - 1]` orientation, rejection of entities on the
   border, timer-zero visibility, and input immutability;
2. exact action-string/index round trips;
3. network batch input produces shape `(batch, 6)` and finite values;
4. replay capacity, uniform sample shapes, and terminal records;
5. Bellman targets do not bootstrap at terminal states;
6. epsilon schedule reaches its endpoints;
7. callback simulation proves fatal transitions are added once and survivor
   endings are not duplicated;
8. checkpoint save/load reproduces Q-values and training counters.

Then run:

```bash
uv run --extra torch python -m unittest agent_code.dqn_brute_force_agent.test_agent
uv run --extra torch python main.py play --no-gui --agents dqn_brute_force_agent random_agent random_agent random_agent --train 1 --n-rounds 2 --seed 1
uv run --extra torch python main.py play --no-gui --my-agent dqn_brute_force_agent --n-rounds 10 --seed 1 --save-stats results/dqn_brute_force_smoke.json
```

For the smoke run only, override replay warm-up and checkpoint interval through
a documented development configuration; do not change algorithm semantics in
the callbacks.

## 6. Train on standard mode from the beginning

The first real run uses exactly:

```bash
uv run --extra torch python main.py play \
  --no-gui \
  --my-agent dqn_brute_force_agent \
  --train 1 \
  --scenario classic \
  --n-rounds <N> \
  --save-stats results/dqn_brute_force_train.json
```

`--my-agent` supplies three rule-based opponents. Use a fresh training seed,
but evaluate on a fixed, disjoint list of seeds. Do not use `coin-heaven`, a
smaller board, imitation data, or hand-coded action selection for this model.

Run evaluation in a separate process with training disabled so epsilon,
optimizer updates, and replay insertion cannot leak into results. At regular
training milestones, evaluate at least 100 rounds against each of:

- three random agents (sanity baseline);
- three rule-based agents (primary benchmark);
- a mixed field of `rule_based_agent`, `coin_collector_agent`, and
  `peaceful_agent` (behavior diversity).

Report mean and uncertainty across multiple seeds, not only the best run.
Primary metrics are mean score, kills per round, survival steps, suicide rate,
and invalid-action rate. Keep training return separate from evaluation score.

## 7. Planned experiments after the baseline works

Change one factor at a time and retain the same evaluation seeds:

1. raw rewards versus paper-style sign clipping;
2. target network versus the preprint's single-network target;
3. Huber versus squared TD loss;
4. replay capacities and warm-up lengths;
5. target-copy frequency and learning rate;
6. one-state input versus four consecutive encoded states;
7. strict sparse reward versus a separately named shaped-reward agent;
8. rule-based opponents versus frozen-checkpoint self-play.

Self-play is deferred until the single learner is stable. Use frozen checkpoint
opponents or explicit shared parameters; never launch several independent
trainable copies that overwrite one checkpoint.

## 8. Submission readiness

Before considering the agent finished:

- benchmark p50/p95/max `act` latency on CPU over representative states;
- run hundreds of evaluation rounds without callback errors or timeouts;
- load the checkpoint in a clean process with `train=False`;
- verify the agent folder contains every imported module and the trained model;
- state the PyTorch dependency in the submission documentation (and add a
  local `requirements.txt` if the submission harness requires it);
- run the provided Docker compatibility test;
- exclude training logs, replay memory, optimizer-only artifacts, and temporary
  checkpoints from the final zip;
- document the exact command, seed set, commit, hyperparameters, and checkpoint
  used for every reported result.

## 9. Decisions selected for the implemented baseline

The first implementation uses:

1. the more stable target network proposed above;
2. unclipped rewards, preserving the game's 5:1 kill-to-coin value;
3. the default replay and evaluation settings listed above, adjustable only
   through documented environment overrides;
4. fresh training by default, with checkpoint resume requiring `DQN_RESUME=1`.
