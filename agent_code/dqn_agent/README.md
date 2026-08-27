# `dqn_agent` — S3, a deep Q-network on board planes

**The submission.** One `model.pt` (2.14 MB) next to this file, loaded by
`callbacks.setup`; torch is imported, `torch.set_num_threads(1)` is called
before anything else, and there is no `multiprocessing`, no absolute path and no
import of the training harness. `tools/check_submission.py` passes **19/19**.

Where the tabular agent reads a hand-designed discrete tuple, this one reads the
board and learns its own features.

**Measured**, 400 rounds on the fixed gate seed block, stock `settings.py`,
`train = False`, this `callbacks.py` and this `model.pt`:

| | |
|---|---|
| vs 3 x `rule_based_agent` | **6.39** [6.04, 6.74] pts/round, 59.9 % win, 94.5 % survival, 4.5 % suicide (repeats on the same seeds: 6.46, 6.32) |
| vs 3 x `bfs_expert` (never tuned against) | 5.98 [5.64, 6.33], 5.8 % suicide |
| head to head vs `q_tabular_agent`, `bfs_expert`, `rule_based_agent` | **5.33** [5.02, 5.64] to 3.98 / 3.27 / 2.27 |
| think time, single process | 15.9 ms mean, 16.8 ms p99, against a 500 ms limit |

## Model

19 planes of 17x17 (`lib/encode.py`) into a dueling, fully convolutional
network (`lib/qnet.py`):

```
stem   Conv3x3(19 -> 64) + ReLU
trunk  6 x ResBlock(64)                receptive field 27 > board diameter 17
adv    Conv3x3(64->64) + ReLU -> Conv1x1(64->6), read out at the agent's tile
val    Conv3x3(64->64) + ReLU -> mean pool -> Linear(64->64) -> Linear(64->1)
Q      V + (A - mean A)
```

533 k parameters, **2.5 ms per forward pass on one CPU thread** — 0.5 % of the
tournament's 500 ms budget.

| planes | |
|---|---|
| 0-2 | wall, crate, collectable coin |
| 3 | self — also the read-out selector, so no coordinate is carried anywhere |
| 4-5 | opponents; opponents that still hold a bomb |
| 6-9 | bomb timer one-hot, `t = 0..3` |
| 10 | explosion active now |
| 11-16 | lethal at `tau = 0..5`, decoded from `lib.danger.lethal_bits` |
| 17-18 | own `bombs_left` and `step / 400`, broadcast |

There is no flatten and no dense trunk: the advantage head is a 1x1 convolution
that produces six values at *every* tile, and the six that matter are read out
at the agent's own tile as `(A * self_plane).sum(spatial)`. That keeps the
translation equivariance which makes the D4 augmentation and the small sample
budget work, and it lets one code path serve a single state and a minibatch.

## Training

```bash
# 1. record the teacher's games (2.5 M transitions, ~8 min on 24 cores)
uv run python -m training.bc collect --out data/expert --episodes 8000 --workers 24

# 2. behaviour-clone the same network (~6 min on the GPU)
uv run python -m training.bc train --data data/expert --out runs/bc/main --steps 40000

# 3. Ape-X fine-tuning: N CPU actors, one GPU learner
uv run python -m training.dqn_driver --config training/configs/s3_gamma95.py --tag main \
    --override pretrained='"runs/bc/main/model.pt"' \
    --override expert_episodes='"data/expert"'

# 4. pick the submission checkpoint on the gate seed block and install it
uv run python -m training.select_checkpoint --run runs/<dir> --top 6 --rounds 400 --install
```

Rainbow-lite: double Q-learning, dueling head, 3-step returns, prioritised
replay (`alpha = 0.6`, `beta` 0.4 -> 1), Huber loss, target sync every 2 500
gradient steps, D4 augmentation of every minibatch, and legal-action masking in
the behaviour policy **and** in the target `max`. `gamma = 0.95`, which is worth
0.4 points and most of the run-to-run stability against the 0.99 the plan
specified (`dev/experiments/011`).

Two things de-risk it, both measured in `dev/experiments/009` and `010`:

* **Behaviour cloning from `bfs_expert`** — the clone alone already scores 5.0
  points per round, so reinforcement learning starts from a competent policy
  rather than from noise.
* **DQfD** — the expert transitions stay in a *second* buffer the actors never
  overwrite, 25 % of every minibatch at the start, with a large-margin loss that
  anneals away. The policy can improve on its teacher but cannot collapse below
  it while it is still learning.

## Inference flags

Both live in `model.pt`'s `meta` block and are settable without retraining
(`tools/model_flags.py`):

| flag | shipped | why |
|---|---|---|
| `use_mask` (S5a proven-lethal action mask) | **off** | worth **-0.007 points** on 400 paired rounds — the learned suicide rate is already 5 %, so the mask only removes actions the policy was not choosing |
| `tta` (average over all eight D4 images) | **on** | **+0.68** on a held-out seed block, Wilcoxon p = 0.004, for 16.8 ms of a 500 ms budget |

## What is hand-written and what is learned

The only injected domain knowledge is planes 11-16 (the blast-timing model) and
the legality mask over the six actions. **Nothing returns an action.** With
`use_mask` off there is no hand-written action filtering at all, and the
ablation in `dev/experiments/012` §D shows that turning it on changes nothing —
the policy's safety is learned. Removing the blast-timing planes entirely costs
about 0.1 points (`dev/experiments/012` §B).

`train.py` ships for completeness and is imported only when `self.train` is set;
the agent runs correctly with the file deleted, which
`tools/check_submission.py` asserts.

## Known weakness

40 of 50 coins in `coin-heaven`, where the tabular agent collects all 50. The
deep agent trained only on `classic` and has never seen a board without crates;
its features are learned, so they do not transfer to a scenario outside the
training distribution the way a hand-written BFS does. The tournament is
`classic`. See `dev/experiments/013`.

## Full numbers

`results/`, `dev/experiments/009`-`013`, and the table in the repository README.
