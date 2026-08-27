# Bomberman RL — final project

Reinforcement-learning agents for the FML Bomberman competition harness. The
engine at the repository root is upstream and unmodified; all of our work lives
in `lib/`, `agent_code/`, `training/`, `tools/` and `tests/`.

## Layout

| path | what |
|---|---|
| `lib/` | single source of truth for shared primitives: the threat model, BFS, features, board-plane encoding, the Q-network, D4 symmetry, rewards |
| `agent_code/dqn_agent/` | **S3 — deep Q-network on board planes. The submission.** |
| `agent_code/q_tabular_agent/` | S1 — tabular Q-learning with D4 folding (second model) |
| `agent_code/q_linear_agent/` | S2 — linear function approximation on the same features (third model) |
| `agent_code/bfs_expert/` | deterministic reference agent and sparring partner (not submittable: it does not learn) |
| `training/` | driver, actors, in-environment evaluation, checkpointing, model selection. Never shipped |
| `tools/` | `sync_lib` (vendors `lib/` into each agent), `check_submission` (the pre-flight gate) |
| `tests/` | conformance tests for the threat model, D4 equivariance, and the training callbacks |
| `dev/experiments/` | one file per experiment: hypothesis, setup, numbers, decision |

`lib/` is vendored into each agent directory by `tools/sync_lib.py`, so an agent
directory is self-contained and the shipped zip needs no import rewriting.

## Results

400 rounds on the fixed gate seed block, stock `settings.py`, `train = False`,
the shipped `callbacks.py` and model file.

| agent | score/round | win rate | survival | suicide | coins | kills |
|---|---|---|---|---|---|---|
| `random_agent` | 0.00 | 0 % | 0 % | 100 % | 0.00 | 0.00 |
| `peaceful_agent` | 0.08 | 0 % | 0 % | 0 % | 0.08 | 0.00 |
| `coin_collector_agent` | 2.84 | 24 % | 35 % | 47 % | 2.56 | 0.06 |
| `rule_based_agent` | 3.38 | 26 % | 40 % | 53 % | 2.63 | 0.15 |
| `bfs_expert` (not submittable) | 4.79 | 43 % | 79 % | 17 % | 3.14 | 0.33 |
| `q_linear_agent` (S2a, 288 weights) | 4.96 | 49 % | 61 % | 32 % | – | – |
| `q_tabular_agent` (S1, ~600 states) | 6.10 | 54 % | 90 % | 9 % | 2.73 | 0.68 |
| **`dqn_agent`** (S3, 533 k parameters) | **6.39** [6.04, 6.74] | **60 %** | **95 %** | **4.5 %** | **3.68** | 0.54 |

**Head to head decides it.** All four in the *same* game, 400 rounds, seating
rotated across the four start corners:

| agent | score | 95 % CI | win rate |
|---|---|---|---|
| **`dqn_agent`** | **5.33** | [5.02, 5.64] | **43.8 %** |
| `q_tabular_agent` | 3.98 | [3.68, 4.29] | 28.9 % |
| `bfs_expert` | 3.27 | [3.02, 3.54] | 18.7 % |
| `rule_based_agent` | 2.27 | [2.06, 2.48] | 8.6 % |

Against three `bfs_expert`s — an opponent neither model was tuned for — the deep
agent scores 5.98 with its suicide rate unchanged at 5.8 %, where the tabular
agent scores 3.16 with its suicide rate doubling.

Think time, measured in a single process as the tournament runs it: `dqn_agent`
15.9 ms mean / 16.8 ms p99, `q_tabular_agent` 0.49 / 1.06, against a 500 ms
limit. `tools/check_submission.py` passes 19/19 on the shipped copy.

Repeating the Task-4 measurement on the same seeds and the same model gives
6.39, 6.46 and 6.32 — opponent-internal randomness is uncontrollable, so **6.4
is the honest figure** and single decimals should not be quoted alone.

Known failure: `dqn_agent` collects 40 of 50 coins in `coin-heaven`, a scenario
it never trained on, where the tabular agent's hand-written BFS feature gets all
50. Full battery, every ablation and the reasoning: `dev/experiments/`.

## Commands

```bash
uv sync --extra torch                                     # torch is optional, S3 needs it
uv run python -m unittest discover -s tests -t .          # conformance tests
uv run python -m training.evaluate --agent dqn_agent --rounds 400 --gate

# S1, tabular
uv run python -m training.driver --config training/configs/s1b.py --tag main

# S3, deep: record the teacher, clone it, then reinforcement-learn from there
uv run python -m training.bc collect --out data/expert --episodes 8000 --workers 24
uv run python -m training.bc train --data data/expert --out runs/bc/main --steps 40000
uv run python -m training.dqn_driver --config training/configs/s3_gamma95.py --tag main \
    --override pretrained='"runs/bc/main/model.pt"' \
    --override expert_episodes='"data/expert"'

uv run python -m training.select_checkpoint --run runs/<dir> --rounds 400 --install
uv run python -m training.gate_report --agent dqn_agent --rounds 400
uv run python -m training.tournament --agents dqn_agent q_tabular_agent bfs_expert \
    rule_based_agent --rounds 400
uv run python -m tools.check_submission --agent dqn_agent --rounds 400 --zip
```

## Things the engine does that cost us time

Recorded here because each one produced a real bug, and every one is invisible
until it bites:

1. `new_game_state` carries **the step number that just finished**, so both
   states of a transition share it. Caching state analysis on `(round, step)`
   silently makes every TD update bootstrap from the state itself.
2. Blasts pass **through crates**; only walls stop them.
3. A blast tile is lethal for **two** consecutive steps, and the second one is
   visible only through `explosion_map`.
4. The final transition of a surviving round is delivered **twice**, once by
   `game_events_occurred` and once by `end_of_round` with `SURVIVED_ROUND`
   appended.
5. A dying agent never receives `game_events_occurred` for its death step.
6. Every world and agent construction attaches a new `FileHandler` to a
   module-level logger, so one world per process, reused.
