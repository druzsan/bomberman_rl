# `q_linear_agent` — S2a, linear Q-function approximation

**Submittable** (pure numpy), but the *second* model: it is kept for the
controlled comparison against `q_tabular_agent`, not as the submission
candidate.

`Q(s, a) = w_a . phi(s)` with **288 weights** — 48 features x 6 actions — on
exactly the same state analysis, reward function, curriculum and harness as the
tabular agent. `phi` is the tabular feature tuple one-hot encoded (with
`move_status` expanded as four one-hot(3) groups rather than a single 81-wide
group) plus the normalised distances the tabular agent has to bucket away:
distance to the objective, distance to the nearest opponent, `step / 400`,
opponents alive, crates in blast, and how many moves are survivable.

Learned with **3-step Expected SARSA**, on-policy on purpose: off-policy
bootstrapping plus function approximation plus bootstrapping is the "deadly
triad", and unlike a table a shared weight vector lets one overestimated action
drag every state with it. The update is normalised by `||phi||^2` and the TD
error clipped, because one update moves ~17 active weights instead of one cell.

## Measured

400 rounds, gate seeds, vs three `rule_based_agent`s: **4.96 points/round**,
48.6 % win rate, 60.7 % survival, 32.3 % suicide.

It reaches 43/50 coins on solo `loot-crate` with 288 parameters where the table
needs ~5 200 entries — far better sample efficiency — but caps lower, and the
whole gap is safety (32 % suicide against the table's 9 %). See
`dev/experiments/008-model-comparison.md`.

## Reproduce

```bash
uv run python -m training.driver --config training/configs/s2a.py --tag lin
uv run python -m training.select_checkpoint --run runs/<dir> --rounds 300 --install
```
