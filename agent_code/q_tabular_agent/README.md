# `q_tabular_agent` — S1, tabular Q-learning with D4 folding

**Submittable.** Pure numpy at inference; no torch, no `multiprocessing`, no
absolute paths. The whole model is `model.npz` (a few hundred kB) next to this
file.

## Model

`Q[state, action]`, `float32`, over the discrete feature tuple `TQ-M` of
[`lib/features.py`](lib/features.py):

| block | meaning | values |
|---|---|---|
| `target_dir` | first BFS step toward the current objective (nearest reachable coin, else a free tile next to a crate, else the nearest opponent) | 6 |
| `move_status` | base-3 code per direction: 0 blocked, 1 enterable but no provable escape, 2 survivable | 81 |
| `wait_ok` | standing still still has a provable escape | 2 |
| `danger_now` | steps until the agent's own tile becomes lethal | 6 |
| `bomb_ready` | `game_state['self'][2]` | 2 |
| `bomb_here_value` | 0 suicide, 1 pointless, 2 ok, 3 good | 4 |
| `opp_dir`, `opp_dist` | direction to and distance from the nearest opponent | 5, 4 |

933 120 tuples in principle; about 600 are reachable after folding, which is why
the shipped table is small.

**D4 folding.** The tuple is canonicalised over the eight symmetries of the
square, the table is read at the canonical entry, and the chosen action is
mapped back through the inverse group element. Every experience therefore counts
eight times. `tests/test_symmetry.py` proves the features are exactly
equivariant, which is what makes the fold sound rather than approximate.

## Training

```bash
uv run python -m training.driver --config training/configs/s1b.py --tag main
uv run python -m training.select_checkpoint --run runs/<dir> --rounds 500 --install
```

Curriculum (`training/configs/s1b.py`): `loot-crate` solo -> `classic` vs weak
opponents -> `classic` vs a mixture of `rule_based_agent`,
`coin_collector_agent`, `bfs_expert` and frozen copies of itself.

Key settings and the measurements behind them are in `dev/experiments/`:
three-step returns, a decaying step size with a Polyak-averaged export, and a
safety curriculum that shields the policy from provably lethal actions early and
removes the shield before the end, so the submitted policy does not depend on it
(`use_mask` is `False` in the shipped model).

## What is hand-written and what is learned

The features contain hand-written search: BFS distances, and a time-expanded
search that proves whether an escape route exists. **No feature returns an
action.** `move_status` marks which neighbours are provably fatal; it typically
leaves two to four actions open and says nothing about which of them is good.
Every choice among the surviving actions comes from the learned table, and the
survivability blocks are ablated in `dev/experiments/`.

## Measured

See `results/` and the table in the repository README.
