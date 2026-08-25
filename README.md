# Bomberman RL — final project

Reinforcement-learning agents for the FML Bomberman competition harness. The
engine at the repository root is upstream and unmodified; all of our work lives
in `lib/`, `agent_code/`, `training/`, `tools/` and `tests/`.

## Layout

| path | what |
|---|---|
| `lib/` | single source of truth for shared primitives: the threat model, BFS, features, D4 symmetry, rewards |
| `agent_code/q_tabular_agent/` | **S1 — tabular Q-learning with D4 folding. The submission candidate.** |
| `agent_code/q_linear_agent/` | S2 — linear function approximation on the same features (second model) |
| `agent_code/bfs_expert/` | deterministic reference agent and sparring partner (not submittable: it does not learn) |
| `training/` | driver, actors, in-environment evaluation, checkpointing, model selection. Never shipped |
| `tools/` | `sync_lib` (vendors `lib/` into each agent), `check_submission` (the pre-flight gate) |
| `tests/` | conformance tests for the threat model, D4 equivariance, and the training callbacks |
| `dev/experiments/` | one file per experiment: hypothesis, setup, numbers, decision |

`lib/` is vendored into each agent directory by `tools/sync_lib.py`, so an agent
directory is self-contained and the shipped zip needs no import rewriting.

## Results

400 rounds on the fixed gate seed block, stock `settings.py`, `train = False`,
the shipped `callbacks.py` and `model.npz`.

| agent | score/round | win rate | survival | suicide | coins | kills |
|---|---|---|---|---|---|---|
| `random_agent` | 0.00 | 0 % | 0 % | 100 % | 0.00 | 0.00 |
| `peaceful_agent` | 0.08 | 0 % | 0 % | 0 % | 0.08 | 0.00 |
| `coin_collector_agent` | 2.84 | 24 % | 35 % | 47 % | 2.56 | 0.06 |
| `rule_based_agent` | 3.38 | 26 % | 40 % | 53 % | 2.63 | 0.15 |
| `bfs_expert` (not submittable) | 4.79 | 43 % | 79 % | 17 % | 3.14 | 0.33 |
| `q_linear_agent` (S2a, 288 weights) | 4.96 | 49 % | 61 % | 32 % | – | – |
| **`q_tabular_agent`** (S1, submission) | **5.91** [5.53, 6.30] | **54 %** | **89 %** | **9 %** | 2.77 | 0.63 |

Solo: 50/50 coins in `coin-heaven` and 42.3/50 coins with 105 crates in
`loot-crate`, both at 0 % suicide. Mean think time 0.9 ms, p99 2.9 ms, against a
500 ms limit. `tools/check_submission.py` passes 18/18 on the shipped copy.

**Head to head**, all four agents in the same game, 400 rounds, seating rotated:
`bfs_expert` 3.85, `q_linear_agent` 3.52, `q_tabular_agent` 3.50,
`rule_based_agent` 2.19. Beating `rule_based_agent` by 2.5 points does not mean
beating a strong field — see `dev/experiments/008-model-comparison.md`.

Full battery, the S5a mask ablation and the state-coverage check:
`dev/experiments/`.

## Commands

```bash
uv sync
uv run python -m unittest discover -s tests -t .          # conformance tests
uv run python -m training.evaluate --agent q_tabular_agent --rounds 300
uv run python -m training.driver --config training/configs/s1b.py --tag main
uv run python -m training.select_checkpoint --run runs/<dir> --rounds 400 --install
uv run python -m training.gate_report --agent q_tabular_agent --rounds 400
uv run python -m tools.check_submission --agent q_tabular_agent --rounds 300 --zip
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
