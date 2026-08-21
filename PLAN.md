# Project plan: Reinforcement Learning for Bomberman

Status: planning only. Nothing in this document is implemented yet.
Written 2026-08-21. All numbers below marked *(measured)* were produced by probe
runs on this machine on 2026-08-21 and are reproducible with the commands given
in [Appendix A](#appendix-a-reproducing-the-measurements).

---

## 0. Hard constraints, deadlines, grading

From `dev/rules.pdf`:

| Item | Value |
| --- | --- |
| Agent-code deadline | Mon 2026-09-21, 21:00 (31 days from today) |
| Pre-run / submission test deadline | Wed 2026-09-17, 21:00 |
| Report deadline | Mon 2026-09-28, 21:00 (~4000 words per team member) |
| Tournament hardware | 1 thread of an AMD Ryzen 5 2600, ≤ 8 GB RAM, CPU only |
| Per-step budget | `settings.TIMEOUT = 0.5 s`; overrun → forced `WAIT` and the excess is deducted from the next step |
| Submission | zip of a single `agent_code/<name>/` directory + `requirements.txt` |

Rules that shape the design:

1. **Must involve machine learning**, and the model must *learn from* the
   features. A feature that deterministically returns "the best action" is
   explicitly disallowed. Pathfinding features ("direction to the nearest
   coin") and life-saving features ("am I in a blast path") are explicitly
   endorsed.
2. **At least two different models** must be built and described; only one is
   submitted to the tournament.
3. **No multiprocessing inside the submitted agent.** Multiprocessing for
   *training* is explicitly allowed. GPUs for training are allowed; inference in
   the tournament is CPU.
4. Engine changes are allowed for training but **the tournament uses the
   original engine**. Nothing the submitted agent needs may live outside its own
   directory, and it must not depend on modified `settings.py` values.
5. The report (systematic, scientific experimentation) outweighs tournament
   placement in the grade. **The experiment log is a deliverable, not a
   by-product** — every run gets a config, a seed, and a machine-readable result
   file from day one.
6. Relative paths only (the engine `chdir`s into the agent directory around
   every callback, `agents.py:303`).

Team note: the rules forbid splitting labour so that each member owns a separate
agent, and require per-section authorship in the report. If this stays a
single-person project, the two-model requirement is still binding; if not,
Stage 2 and Stage 3 below are the natural parallel workstreams and each must
have both members' fingerprints (e.g. one owns training infrastructure and
evaluation, the other owns representation and reward design, for *both* models).

---

## 1. What "good" means: measured baselines

All numbers *(measured)*, `classic` scenario, per round and per agent.

**Four `rule_based_agent`s against each other** (20 rounds, seed 7):

| metric | value |
| --- | --- |
| score / round | 3.35 – 3.85 |
| coins / round | 1.85 – 2.45 |
| kills / round | 0.20 – 0.35 |
| **suicides / round** | **0.45 – 0.65** |
| mean round length | 336 steps |
| invalid actions | ~3 % of steps |
| `act` latency | 0.51 – 0.55 ms |

**One agent against three `rule_based_agent`s** (30 rounds, seed 11):

| agent | score/r | coins/r | kills/r | suicides/r | mean steps alive |
| --- | --- | --- | --- | --- | --- |
| `coin_collector_agent` | **3.17** | 2.67 | 0.10 | 0.57 | 215 |
| `rule_based_agent` (reference) | 2.7 – 3.3 | 1.7 – 2.3 | 0.13 – 0.20 | 0.40 – 0.53 | ~226 |
| `peaceful_agent` | 0.07 | 0.07 | 0.00 | 0.00 | 159 |
| `random_agent` | 0.00 | 0.00 | 0.00 | **1.00** | 16 |

**Ceiling with weak opponents** — `coin_collector_agent` vs 3 `peaceful_agent`
(30 rounds, seed 21): score **19.33**/round, 8.33 of 9 coins, 113.7 crates,
2.20 kills, 0.07 suicides, alive 391 of 400 steps.

### Consequences for the design

* The bar is low in absolute terms: **score ≈ 3.5 per round beats
  `rule_based_agent`.** A scripted coin collector with *no* combat logic already
  matches it.
* `rule_based_agent` **kills itself in roughly half of all rounds.** Suicide
  avoidance is the single highest-value skill; it is worth more than any
  offensive capability. Target: **< 0.05 suicides/round**.
* Being the last one alive is extremely valuable: with crates still on the board
  the round runs to step 400 (`environment.py:289` only stops early when nothing
  is left to do), so a lone survivor farms the remaining coins undisturbed. The
  9 coins of a round are the reliable score source; kills (5 points) are the
  high-variance upside.
* Per-round score variance is large (0 – 15), so **every comparison needs
  hundreds of paired rounds** and confidence intervals. See §4.3.

---

## 2. Engine facts pinned down by probes

These are the facts a safety model must encode. Each was verified by driving
`BombeRLeWorld` directly (Appendix A), not inferred from reading alone.

### 2.1 Bomb and explosion timeline *(measured)*

Drop a bomb during step `d`:

| step | what the agent observes in `game_state` | what happens |
| --- | --- | --- |
| `d` | bomb not yet in `bombs` | `BOMB_DROPPED`, `bombs_left` becomes False |
| `d+1` | `bombs = [((x,y), 3)]` | free move |
| `d+2` | timer 2 | free move |
| `d+3` | timer 1 | free move |
| `d+4` | timer 0 | action executes, **then** the bomb detonates; anyone on a blast tile dies |
| `d+5` | `explosion_map == 1` on blast tiles | those tiles are **still lethal** |
| `d+6` | `explosion_map == 0`, smoke only | harmless |
| `d+7` | `self[2] (bombs_left) == True` | next bomb may be dropped |

Derived rules:

* A bomb seen with timer `t` at step `k` detonates at the **end of step `k+t`**.
* After dropping, the agent has exactly **4 actions** before the blast lands.
  With `BOMB_POWER = 3`, escaping a straight corridor requires reaching
  distance ≥ 4 along the corridor, or turning a corner. Moving 3 tiles
  away and waiting **dies** *(verified)*.
* **`explosion_map[tile] > 0` ⇒ entering or staying on that tile this step is
  fatal.** The only value ever observed by an agent is 1 (blast tiles are lethal
  for the detonation step plus exactly one more step).
* **Bomb period is 7 steps** — an agent can place at most ~57 bombs per round.
* Bombs are obstacles (`environment.py:121`): a bomb blocks movement for
  everyone, including the owner, who may step off its tile but not back on. A
  bomb dropped in a corridor can seal an escape route.

### 2.2 Kill credit is not exclusive *(measured)*

`evaluate_explosions` (`environment.py:239`) loops over *all* explosions. If two
blasts cover the same victim, **both owners are credited +5**, and a victim who
dies on its own bomb *also* pays out to every other agent whose blast covers the
same tile. Tactical consequence: bombing a tile an opponent is about to die on
is free points. Given the measured 0.5 suicides/round of `rule_based_agent`,
"shadow-bombing a doomed opponent" is a cheap, low-risk source of kill points
and is worth an explicit feature (§5.1, feature group G) and an experiment.

### 2.3 Step order and simultaneity

`do_step` (`environment.py:158`): all agents are asked to `act` on the *same*
snapshot, then the actions are applied **in a random permutation**, then coins,
explosions, bombs, deaths, then `game_events_occurred`. So:

* the `new_game_state` seen in `game_events_occurred` is post-world-update, and
  the event list covers the whole step;
* whether a move into a contested tile succeeds depends on the random
  permutation — two agents cannot occupy one tile, and the loser gets
  `INVALID_ACTION`. Any adversarial reasoning must treat the outcome of
  contested tiles as stochastic;
* an agent that dies in step `k` gets **no** `game_events_occurred` call for
  that step; the fatal transition is only visible in `end_of_round`. Survivors
  get the final transition **twice** (once in `game_events_occurred`, once in
  `end_of_round`). Both cases must be handled explicitly or the learner will
  never see its own deaths, or double-count final rewards.

### 2.4 Symmetry *(measured)*

The wall layout (`arena[x,y] = -1 where (x+1)(y+1) % 2 == 1` plus the border) is
invariant under the full dihedral group **D4** (4 rotations × mirror; verified
for all 8 elements), and the four start positions map onto each other. 113 walls,
176 free tiles. Therefore **8× data augmentation** is exact — with the action
labels permuted accordingly — for any allocentric representation, and directional
feature blocks can be permuted the same way.

### 2.5 Board composition *(measured)*

`classic`: 123.6 crates on average (115 – 133), 52.4 initially free tiles,
9 coins, mostly hidden under crates. The agent must bomb to open the board at
all; coin collection without bombing is not viable in `classic`.

### 2.6 Compute budget *(measured)*

Single-threaded CPU, per `act` call, against a 0.5 s limit:

| policy | latency | calls affordable per step (at 80 % of budget) |
| --- | --- | --- |
| `rule_based_agent` (BFS python) | 0.54 ms | ~740 |
| small CNN (12×15×15, 3 conv + head), batch 1 | 0.55 ms | ~720 |
| same CNN, batch 32 | 11.4 ms | ~35 |
| MLP on 64 features (128-128) | 0.014 ms | ~28 000 |

**There is ~1000× headroom.** This is a major, under-used design lever: a
feature-based policy can afford a few thousand extra node evaluations per step,
which makes inference-time safety verification (§6.4) and even a shallow search
over a learned value function realistic. The tournament CPU (Ryzen 5 2600, Zen+ 2018) is roughly **2.5 – 3× slower per
core** than this machine's i7-14700KF, so keep a hard latency budget of
**≤ 50 ms p99 as measured here** (≈ 150 ms there).

### 2.7 Training throughput *(measured)*

| setting | throughput |
| --- | --- |
| `main.py`, 4 × `rule_based_agent`, `--no-gui` | ~2 rounds/s (~700 steps/s) |
| `BombeRLeWorld.do_step` driven directly, 4 trivial agents | 175 µs/step (5 700 steps/s) |
| same, with `logging.disable(logging.CRITICAL)` | **52 µs/step (19 200 steps/s)** |
| `get_state_for_agent` | 1.4 µs (negligible) |

Two conclusions: **engine logging costs 3.4× throughput** (disable it in
training), and the raw engine is fast enough that no reimplementation is needed
— rollout cost will be dominated by the policy, not the simulator. With 28 cores
available, 10⁴ – 10⁵ transitions/s is reachable for a feature-based policy and
~2 – 5 · 10³/s for a CNN with batched GPU inference.

Hardware on this machine: 28 cores, 62 GB RAM, RTX 4080 SUPER, torch 2.13 + CUDA
available *(measured)*.

---

## 3. Lessons from the two prior attempts in this repo

Commit `399be49` ("Add experiments") contained two agents, both deleted from the
working tree. Their design documents are worth mining:

* `dqn_brute_force_agent` — deliberate brute force: raw 10×15×15 tensor, sparse
  score-only reward, no features, no curriculum, `classic` from round one. This
  is the known-hardest possible starting point (sparse reward + instant-death
  dynamics + three strong opponents). Its `PLAN.md` contains an excellent
  checklist of *correctness* traps that should be carried over verbatim:
  terminal-transition handling (§2.3), atomic checkpoints, latency benchmarking,
  never shipping an untrained fallback silently.
* `cnn_agent` — factorized planes, D4 augmentation, masked 5-step Double DQN,
  and a **re-implemented standalone simulator** (`standalone_training/environment.py`,
  309 lines). Its runbook shows a coin-stage curriculum that never graduated to
  a validated `classic` model.

Carry over: factorized planes, D4 augmentation, n-step Double DQN, run
directories with `metrics.jsonl` / `evaluations.jsonl`, held-out evaluation seeds,
explicit promotion step.

Avoid: (a) re-implementing the dynamics — subclass the real engine instead
(§4.1), so training and tournament dynamics are identical by construction;
(b) starting the hard scenario with a sparse reward and no features; (c) letting
the deep model be the only candidate.

---

## 4. Stage 0 — infrastructure (build before any model)

Everything here is dev-only code. **Import direction is one-way: training code
imports the agent package; the agent package never imports training code.** That
keeps the shipped directory self-contained.

Proposed layout:

```
training/                      # dev only, never shipped
  driver.py                    # programmatic engine driver (§4.1)
  policies.py                  # Policy protocol + scripted/random/module-backed policies
  rollout.py                   # episode collection, transition assembly, terminal handling
  evaluate.py                  # evaluation suites, paired seeds, bootstrap CIs (§4.3)
  workers.py                   # multiprocess rollout / eval fan-out
  replay.py                    # uniform + prioritized replay, n-step, D4 augmentation
  metrics.py                   # JSONL run records, plots
  experiments/                 # one file per experiment: config + command + result path
runs/                          # gitignored: checkpoints, metrics.jsonl, evaluations.jsonl
agent_code/<agent>/            # shipped: callbacks.py, train.py, features/model, weights
```

### 4.1 Programmatic engine driver

Subclass `BombeRLeWorld` and override only the agent plumbing:

* `setup_agents` / `add_agent` → attach lightweight training agents (mimicking
  the `Agent` surface actually used by the world: `x, y, dead, score, events,
  bombs_left, trophies, name, train, add_event, note_stat, update_score,
  get_state`) instead of loading callback modules through
  `SequentialAgentBackend`. This removes the per-callback `os.chdir`, the
  `importlib` layer and the pygame avatar loading.
* `poll_and_run_agents` → collect all four observations, run **one batched
  inference call** for all agents sharing a policy, then apply actions using the
  inherited random permutation.
* `send_game_events` / `end_round` → hand transitions to the collector instead
  of calling agent callbacks.
* disable logging (`logging.disable`) and never write replays during training.

Everything that defines the *rules* — `build_arena`, `collect_coins`,
`update_bombs`, `update_explosions`, `evaluate_explosions`, `perform_agent_action`,
`get_state_for_agent`, `time_to_stop` — is inherited unchanged. That is the whole
point: **zero dynamics drift**.

Validation gate (must pass before any training run):
1. `rule_based_agent` driven through the driver reproduces the score/suicide/step
   distribution measured through `main.py` (two-sample test on ≥ 300 rounds,
   same seeds).
2. For a fixed seed, the arena, coin layout and the action permutation sequence
   are identical to the framework's.
3. A recorded driver episode replays step-for-step through
   `perform_agent_action` with matching events.

### 4.2 Danger oracle (shared by every model)

A single, unit-tested module `agent_code/<agent>/danger.py` computing, from a
`game_state`:

* `lethal_at[t]` — for `t = 0..6`, the set of tiles that are lethal at step
  `now + t`, from bomb positions/timers (§2.1) and the current `explosion_map`;
* `blocked[t]` — tiles blocked by walls, crates, bombs (crates are removed by
  blasts, so blocking is time-dependent);
* `survivable(action)` — a space-time BFS over (tile, t) up to horizon 6:
  does a path exist that never occupies a lethal tile at a lethal time,
  assuming opponents stand still (pessimistic variants: assume opponents may
  block the corridor);
* `escape_after_bomb()` — the same query with a hypothetical own bomb at the
  current tile;
* `blast_tiles(tile)` — the tiles a bomb at `tile` would cover, using the real
  `Bomb.get_blast_coords` logic against the current field.

Property tests: for randomly generated states, `survivable("WAIT") == False`
must imply the agent actually dies when the state is rolled forward in the
driver with a waiting policy; `escape_after_bomb()` must agree with a
brute-force 4-step lookahead. Cost budget: < 1 ms *(the whole board is 17×17;
the MLP figure in §2.6 shows the budget is not the constraint)*.

This module is shared by all models and is the backbone of the safety features.
It is a **feature provider, not a policy** — it never chooses an action.

### 4.3 Evaluation harness and statistics

* **Suites** (all `classic`, all with seeds disjoint from training seeds):
  E1 vs 3 × `random_agent` (sanity), E2 vs 3 × `peaceful_agent` (farming
  ceiling), E3 vs 3 × `coin_collector_agent`, **E4 vs 3 × `rule_based_agent`
  (primary)**, E5 mixed (`rule_based` + `coin_collector` + `peaceful`),
  E6 vs 3 frozen copies of our own current best, E7 self-mirror (4 copies).
* **Sample size**: ≥ 400 rounds per suite, always the *same* seed list, so all
  configurations are compared paired per seed. Report mean ± bootstrap 95 % CI
  of the paired difference. Measure the per-round score standard deviation in the
  first run and recompute the required N from it (with σ ≈ 3.5, N = 400 gives
  SE ≈ 0.18 — enough to resolve a 0.5-point difference, not a 0.1-point one).
* **Metrics**: mean score (primary), win rate (strictly highest score in a
  round), coins, kills, suicides, deaths, mean steps alive, invalid-action rate,
  `act` latency p50/p95/max, timeout count.
* **Final claims** always average over ≥ 3 *training* seeds, not just 3
  evaluation seeds; report the spread across training seeds.
* Every configuration also runs a **framework sanity match** through `main.py`
  with pristine `settings.py` before it is allowed to be promoted.

### 4.4 Experiment bookkeeping

One run directory per (model, config, seed): `config.json`, `metrics.jsonl`
(one record per training round), `evaluations.jsonl` (immutable held-out
results), `checkpoints/`, `models/`. Never reuse a directory. A one-line
`experiments/INDEX.md` entry per run: hypothesis, command, result, verdict. The
report's Experiments section is assembled from these files, so the schema is
fixed on day one.

---

## 5. Stage 1 — Model A: feature-based Q-learning (the workhorse)

Rationale: the rules note that previous competitions were won by *simple,
carefully engineered* models; §1 shows the bar is ≈ 3.5 points; §2.6 shows a
feature policy costs 0.014 ms. This model is the fallback that must be
tournament-ready **by 2026-09-12** — before the deep model is even finished.
It also satisfies "at least one model focuses on techniques from the lecture"
(tabular / linear Q-learning).

### 5.1 Feature design

All directional features are ordered `[UP, RIGHT, DOWN, LEFT]` so that a D4
transform is a permutation of feature blocks plus a permutation of actions.

| # | group | features | why |
| --- | --- | --- | --- |
| A | legality | `can_move[4]`, `can_bomb` | avoid `INVALID_ACTION` (~3 % of `rule_based`'s steps) |
| B | immediate danger | `lethal_now[5]` (stay + 4 dirs), `in_blast_line`, `steps_to_impact` (0–4, one-hot + none) | §2.1 |
| C | survivability | `survivable[5]`, `n_safe_escape_routes`, `escape_after_bomb` | the highest-value skill (§1) |
| D | coins | one-hot BFS direction to nearest reachable coin (5) + `1/(1+d)`, `n_coins_visible` | explicitly endorsed by the rules |
| E | crates | BFS direction to the nearest tile from which a bomb hits ≥1 crate + distance, `crates_hit_if_bomb_here` (0–4+), `crates_left/124` | crates gate coin access (§2.5) |
| F | mobility | `degree` of current tile, `degree` of each neighbour, `in_dead_end` | dead ends are death traps and bombing spots |
| G | opponents | BFS direction + distance bucket to nearest opponent, `n_alive`, `opponent_escape_routes` (are they trapped?), `bomb_here_threatens_opponent`, `opponent_is_doomed` (§2.2) | combat and free kill credit |
| H | context | `step/400`, `my_score_lead`, `bomb_cooldown_remaining` | endgame vs opening behaviour |

Two representations are compared:

* **A1 tabular.** A coarse discretization of a *subset* — e.g. coin direction
  (5) × danger pattern of the 5 tiles (2⁵) × `can_bomb` (2) × `escape_after_bomb`
  (2) × crate direction (5) ≈ 3 200 states × 6 actions. Fits in a dict, converges
  in minutes, is fully interpretable (the report can print the learned policy
  table). Debugging vehicle and the "lecture technique" reference point.
* **A2 function approximation.** The full ~60–90-dimensional vector with either
  (i) linear Q per action (`Q(s,·) = W φ(s)`, `W ∈ R^{6×d}`, semi-gradient
  n-step Q-learning) or (ii) gradient-boosted trees / extra-trees per action
  trained by **fitted-Q iteration** on the replay buffer
  (`sklearn` / `lightgbm`, both present in the tournament Dockerfile).
  Trees capture the feature interactions ("bomb only if `escape_after_bomb`")
  that linear Q cannot, at ~0.05 ms inference.

### 5.2 Learning algorithm

* n-step (n = 3–5) off-policy Q-learning, γ = 0.95 for the coin tasks, 0.99 for
  `classic`; frozen target copy refreshed every N updates (or per FQI round).
* Replay buffer 2 · 10⁵ transitions, uniform first, prioritized as an ablation.
* ε-greedy: 1.0 → 0.05 linearly over the first ~2 · 10⁵ transitions, with an
  ε = 0.02 floor; evaluation greedy.
* **D4 augmentation**: every stored transition is expanded to 8 by permuting the
  directional feature blocks and the action label. Exact by §2.4; effectively
  8× the data for free.
* Bootstrapping fix: transitions where the agent died must be terminal with the
  death penalty, and survivor endings must not be inserted twice (§2.3).

### 5.3 Curriculum (the four prescribed tasks)

| task | setup | gate to advance |
| --- | --- | --- |
| T1 coins, no crates | `--scenario coin-heaven`, 1 agent, bombs disabled | ≥ 45 of 50 coins per round, 0 suicides, mean steps/coin falling |
| T2 crates + hidden coins | `loot-crate`, then `classic` with 1 agent | ≥ 8 of 9 coins, **suicides < 0.02/round**, ≥ 80 crates destroyed |
| T3 easy opponents | `classic` vs 3 × `peaceful_agent`, then 3 × `coin_collector_agent` | score ≥ 12 vs peaceful; beat `coin_collector_agent` head-to-head |
| T4 full game | `classic` vs 3 × `rule_based_agent` | **mean score > `rule_based_agent`'s in the same rounds, paired CI excluding 0** |

Training-only environment relaxations are allowed and encouraged for T1–T2
(smaller board, disabled bombs, more coins) — but every gate is measured on the
*unmodified* `classic` settings, and the final model is validated with pristine
`settings.py`.

### 5.4 Reward design (shared by both models)

True objective = game score. Shaping is an ablation ladder, each level a separate
run, always reporting the *true* score:

* **R0** sparse only: `COIN_COLLECTED +1`, `KILLED_OPPONENT +5`.
* **R1** + death: `KILLED_SELF -5`, `GOT_KILLED -5`, `SURVIVED_ROUND +1`.
  (Expected to be the single biggest jump, given §1.)
* **R2** + instrumental events: `CRATE_DESTROYED +0.05`, `COIN_FOUND +0.05`,
  `INVALID_ACTION -0.02`, small per-step cost `-0.002`.
* **R3** + **potential-based shaping** (Ng et al., cited in the rules):
  `F(s,s') = γ φ(s') − φ(s)` with
  `φ(s) = w_c · (−d_coin(s)) + w_x · (−d_crate_target(s)) + w_s · safe(s)`,
  where `d_*` are BFS distances and `safe(s) ∈ {0,1}` is `survivable("WAIT")`.
  Because it is a potential difference, the optimal policy is provably unchanged
  — this is exactly the trap the rules warn about ("auxiliary rewards should only
  depend on states"), so **no reward may be attached to an action label**
  (no "+0.1 for moving toward the coin"), which otherwise makes back-and-forth
  oscillation profitable.
* **R4** tuned weights (small random search over `w_c, w_x, w_s`, death penalty,
  γ), selected on E4 held-out score.

Anti-reward-hacking checks per run: oscillation rate (fraction of steps
returning to the tile occupied 2 steps ago), bombs-per-crate ratio, and
"waiting in a safe corner forever" detection (steps with no BFS progress).

### 5.5 Optional warm start: imitation from `rule_based_agent`

Behaviour cloning on ~10⁶ `(features, action)` pairs from `rule_based_agent`
(collected through the driver at ~10⁴ steps/s, i.e. minutes) as an initializer,
then RL fine-tuning. This *is* machine learning (supervised), keeps the model
learning from our own features, and gives a strong starting policy. Risks:
inherits the 0.5 suicides/round habit, and caps at rule-based level. Treat as an
explicit experiment with an on/off comparison, not as the default; a cloned
policy must never be submitted without an RL phase on top.

---

## 6. Stage 2 — Model B: deep RL (the upside)

Only started once Model A clears T3, and it must not delay Model A's T4 gate.

### 6.1 Representation

Allocentric, border cropped: **15 × 15 × C** float planes.

| plane | contents |
| --- | --- |
| 0 | wall |
| 1 | crate |
| 2 | collectable coin |
| 3 | self |
| 4 | opponents |
| 5–8 | bomb timer one-hot (t = 3, 2, 1, 0) |
| 9 | explosion (lethal now) |
| 10 | own-bomb indicator (which bombs are mine) |
| 11 | opponent `bombs_left` at their tiles |
| 12 | `can_bomb` (broadcast scalar) |
| 13 | `step/400` (broadcast scalar) |
| 14–16 | *derived* "lethal at t = 1, 2, 3" planes from the danger oracle — **ablation**: with vs without |

Planes 14–16 are the bridge between the two models: they let the CNN inherit
Model A's hardest-won knowledge. The with/without comparison is one of the most
interesting experiments in the report (does the network learn blast dynamics
from raw timers, given the sample budget?).

Ablation: egocentric 9×9 crop with rotation canonicalization (cheaper, loses
global context) vs allocentric 15×15.

### 6.2 Network and algorithm

* ~4 residual blocks of 3×3 × 64 → dueling head → 6 Q-values; ≈ 300 k params;
  0.55 ms per CPU forward *(measured, §2.6)* — comfortably inside budget.
* **Double DQN + dueling + n-step (n = 3) + PER (α 0.6, β 0.4→1) + Huber +
  Adam 2.5e-4 + grad-norm clip 10**, target sync every 2 000 updates, batch 512
  on the GPU, replay 5 · 10⁵ (uint8 planes; ~1 GB), train every 4 env steps,
  ε 1 → 0.05 over 5 · 10⁵ env steps.
* **D4 augmentation at replay-sample time** (§2.4).
* Curriculum identical to §5.3 (T1 → T4), warm-started from the previous stage's
  weights each time the scenario changes.
* **Self-play league** for the final phase: opponents sampled from a pool of
  frozen own snapshots (60 %), `rule_based_agent` (25 %),
  `coin_collector_agent` / `peaceful_agent` (15 %) — the fixed-opponent mix
  prevents the collapse into a degenerate mutual-avoidance equilibrium and keeps
  E4 honest. Never run several trainable copies writing one checkpoint; a single
  learner with shared parameters across the four seats is fine and gives 4×
  transitions per step.
* Throughput plan: 8–16 rollout worker processes (§2.7) feeding one GPU learner;
  target ≥ 2 000 transitions/s, i.e. ~10⁷ transitions in ~1.5 h of wall clock —
  the budget allows several full runs plus ablations.

Alternative if DQN stalls: **PPO** with 64 parallel driver instances and GAE
(on-policy, no replay, tolerates self-play better, but no augmentation benefit
and less sample-efficient). Decide by 2026-09-08; do not run both to completion.

### 6.3 Model B risks

Non-convergence by the deadline is the historically most common failure (the
rules warn about it explicitly, and both prior attempts in this repo stopped at
the coin stage). Mitigation: Model A is the default submission; Model B only
replaces it if it wins on E4 with a paired CI excluding 0 *and* survives the
robustness checks in §7.

### 6.4 Stage 3 — optional upside, only if time remains

1. **Inference-time safety verification.** Use the 0.5 s budget: enumerate the
   ≤ 6 actions, drop those the danger oracle proves fatal, and let the learned
   Q choose among the rest. Defensible under the rules (a life-saving feature,
   not a "best action" oracle) *but* it must be reported as a variant with an
   explicit A/B against the unfiltered policy, and the submitted agent's action
   choice among survivable moves must remain learned. If only one action
   survives, the mask does decide — report how often that happens.
2. **Shallow search over the learned value function** (2–3 plies over own
   moves, opponents assumed to hold still or to play a cheap scripted policy):
   ~700 CNN evaluations/step are available *(measured)*, or ~28 000 MLP
   evaluations. Cheap and often worth more than more training.
3. **Opponent modelling**: predict opponent actions from a small classifier,
   feed the prediction as features. Only if the basics are done.
4. Distributional (C51/QR-DQN), R2D2-style recurrence — mention as outlook in
   the report rather than implementing.

---

## 7. Experiment matrix

Ordered by information per hour of compute. Each row = one row in the report's
results table, each measured on suites E1–E5 with paired seeds.

| # | question | comparison |
| --- | --- | --- |
| X1 | how much does the death penalty matter? | R0 vs R1 |
| X2 | do potential-based potentials beat event rewards? | R2 vs R3 (+ non-potential control, expected to oscillate) |
| X3 | which feature groups carry the model? | leave-one-group-out over A–H (Model A2) |
| X4 | tabular vs linear vs trees | A1 vs A2-linear vs A2-GBT |
| X5 | is D4 augmentation worth it? | on/off, both models |
| X6 | does the CNN need derived danger planes? | planes 14–16 on/off |
| X7 | curriculum vs `classic` from scratch | staged vs direct (this is the prior brute-force experiment, now with a proper control) |
| X8 | opponent mix during training | rule-based only vs self-play league vs mixed |
| X9 | imitation warm start | BC-init vs random-init, both RL-finetuned |
| X10 | safety mask / search at inference | off / mask / mask+search |
| X11 | hyperparameters | γ, n-step, lr, ε schedule, replay size — random search, report sensitivity |
| X12 | robustness | performance vs board size, crate density, coin count, 1v1 vs 4-player, unseen opponent (other teams' agents from Discord) |

Sample-size discipline: a difference is only reported as real if the paired
bootstrap CI on ≥ 400 rounds excludes 0 **and** it survives on a second training
seed.

---

## 8. Timeline (31 days to the code deadline)

| dates | milestone | exit criterion |
| --- | --- | --- |
| Aug 21 – 24 | Stage 0 infra: driver + validation gate, danger oracle + property tests, eval harness, run bookkeeping | driver reproduces framework statistics; oracle passes brute-force tests |
| Aug 25 – 27 | Model A1 tabular, T1 + T2 | ≥ 8/9 coins, < 0.02 suicides on `classic` solo |
| Aug 28 – Sep 1 | Model A2 (linear + GBT), T2 → T3, reward ladder R0–R3 | beats `coin_collector_agent` head-to-head |
| Sep 2 – Sep 8 | Model A2 on T4 + tuning (R4, X3, X4, X5, X11) | **paired win over `rule_based_agent` on E4** — this is the fallback submission, frozen and archived |
| Sep 5 – Sep 12 | Model B in parallel: representation, DQN, curriculum T1–T3 | T3 cleared; throughput ≥ 2 000 transitions/s |
| Sep 12 – Sep 15 | Model B self-play league on T4 + X6, X8 | E4 comparison against Model A decided |
| Sep 16 – Sep 17 | submission hardening: latency profiling, Docker build, clean-checkout test, `requirements.txt`, **upload the test zip (deadline Sep 17)** | pre-run passes with no stack trace |
| Sep 18 – Sep 21 | final training/selection only; no new features. Promote best checkpoint, re-validate, **submit by Sep 21 21:00** | zip contains only the agent directory + weights |
| Sep 22 – Sep 28 | report (figures generated from `evaluations.jsonl` throughout, not at the end) | 4000 words/member, per-section authorship |

Buffer rule: if Model A has not cleared T4 by **Sep 8**, stop all Model B work
and fix Model A.

---

## 9. Submission checklist (run as an automated script)

1. `uv run python -m unittest test.py` passes (engine smoke test).
2. Agent directory imports and plays with `train=False` in a **clean checkout**
   of the engine (`git stash` all local engine changes, pristine `settings.py`).
3. 400 rounds vs 3 × `random_agent` (what the graders' pre-run does) and 400 vs
   3 × `rule_based_agent` with **zero** exceptions and zero timeouts.
4. `act` latency p50/p95/max logged; p99 ≤ 50 ms; `torch.set_num_threads(1)`.
5. No multiprocessing, no threads, no GPU calls, no absolute paths, no imports
   from outside the agent directory, no reliance on non-default settings.
6. Weights load from a path built with `Path(__file__).parent`; a missing weight
   file raises a clear error instead of silently playing randomly.
7. `requirements.txt` lists exactly what is needed beyond the Dockerfile.
8. `docker build .` + the container run from the rules' §8 procedure.
9. Zip contains only the agent directory: no `runs/`, replay buffers, optimizer
   state, logs, or training code.
10. Record the exact commit, config, seeds and checkpoint used for the submitted
    model in `experiments/INDEX.md`.

---

## 10. Risk register

| risk | likelihood | mitigation |
| --- | --- | --- |
| Deep model not converged by the deadline | high | Model A frozen as the fallback by Sep 8; Model B must *beat* it to be submitted |
| Overfitting to `rule_based_agent` | high | E1–E7 suites, self-play league, X12 robustness, other teams' agents from Discord |
| Reward hacking (oscillation, corner camping) | medium | potential-based shaping only; oscillation/no-progress detectors in every run record |
| Terminal-transition bug ⇒ the learner never sees its own deaths | medium | explicit unit test of the callback sequence (§2.3) before any long run |
| Silent dynamics drift in the training loop | medium | driver subclasses the real engine + validation gate (§4.1) |
| Tournament timeout on slower CPU | low | 50 ms p99 budget, single-threaded torch, latency test in the checklist |
| Result unreproducible for the report | medium | seeds + config + commit in every run directory; figures generated from those files |
| Engine-modification leakage into the submission | low | clean-checkout validation in the checklist |

---

## Appendix A: reproducing the measurements

```bash
uv sync --extra torch

# Baselines (§1)
uv run python main.py play --no-gui --n-rounds 20 --seed 7  --save-stats results/rb4.json
uv run python main.py play --no-gui --n-rounds 30 --seed 11 --save-stats results/cc_vs_rb.json \
  --agents coin_collector_agent rule_based_agent rule_based_agent rule_based_agent
uv run python main.py play --no-gui --n-rounds 30 --seed 21 --save-stats results/cc_vs_peace.json \
  --agents coin_collector_agent peaceful_agent peaceful_agent peaceful_agent
```

Probe scripts (bomb timeline, bomb cooldown, kill credit, D4 symmetry, engine
throughput, inference latency) were run ad hoc against `BombeRLeWorld` driven by
`user_agent` and by direct manipulation of `world.bombs`. **Stage 0 must
re-create them as permanent tests** under `training/` and
`agent_code/<agent>/tests/`, because every fact in §2 is a load-bearing
assumption of the reward and feature design:

* timeline: drop a bomb, walk 3 tiles, `WAIT` ⇒ death; walk 4 tiles ⇒ survival;
  re-entering a blast tile one step after detonation ⇒ death.
* cooldown: `bombs_left` returns in the state at step `d+7`.
* credit: two blasts covering one victim credit both owners (+5 each).
* symmetry: all 8 D4 transforms leave the wall layout invariant.
* throughput: `logging.disable(logging.CRITICAL)` ⇒ 52 µs/step vs 175 µs/step.
