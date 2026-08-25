"""Print what a tabular model actually learned.

A tabular agent's whole policy is inspectable, which is one of its advantages
over a deep model: the table below goes straight into the report and is also the
fastest way to see whether the escape behaviour was learned at all.

    uv run python -m training.inspect_model runs/<dir>/checkpoints/best/model.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lib.board import ACTIONS
from lib.features import CARDINALITY, FeatureSpec

DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "HERE", "NONE"]
ESC_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "DOOMED", "SAFE"]


def load(path: Path):
    with np.load(path) as data:
        spec = FeatureSpec(str(data["feature_set"]))
        q = np.zeros((spec.n_states, 6), dtype=np.float32)
        n = np.zeros((spec.n_states, 6), dtype=np.uint32)
        q[data["states"]] = data["values"]
        if "counts" in data.files:
            n[data["states"]] = data["counts"]
        return spec, q, n, bool(data["fold"])


def describe(spec: FeatureSpec, values) -> str:
    parts = []
    for block, v in zip(spec.blocks, values):
        if block == "target_dir":
            parts.append(f"target={DIR_NAMES[v]}")
        elif block == "escape_dir":
            parts.append(f"escape={ESC_NAMES[v]}")
        elif block == "danger_now":
            parts.append("danger=safe" if v == 5 else f"danger=t{v}")
        elif block == "walkable":
            parts.append("walk=" + "".join(d for i, d in enumerate("URDL") if v & (1 << i)))
        elif block == "bomb_ready":
            parts.append("bomb" if v else "nobomb")
        elif block == "bomb_here_value":
            parts.append(["drop=suicide", "drop=pointless", "drop=ok", "drop=good"][v])
        elif block == "opp_dist":
            parts.append(["opp=adj", "opp<=3", "opp<=6", "opp=far"][v])
        elif block == "opp_dir":
            parts.append(f"oppdir={DIR_NAMES[v]}")
        elif block == "in_dead_end":
            parts.append("deadend" if v else "open")
    return " ".join(parts)


def unindex(spec: FeatureSpec, idx: int) -> tuple[int, ...]:
    out = []
    for card, stride in zip(spec.cards, spec.strides):
        out.append((idx // stride) % card)
    return tuple(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--top", type=int, default=30, help="show the most-visited states")
    p.add_argument("--danger-only", action="store_true",
                   help="only states where the agent is inside a blast")
    args = p.parse_args(argv)

    spec, q, n, fold = load(Path(args.model))
    # n is uint32: negating it in place would wrap around, so widen first.
    visits = n.sum(axis=1).astype(np.int64)
    order = np.argsort(-visits)
    danger_index = spec.blocks.index("danger_now") if "danger_now" in spec.blocks else None

    print(f"{spec.name}  fold={fold}  visited states={int((visits > 0).sum())}  "
          f"total visits={int(visits.sum()):,}")
    escape_ok = escape_total = 0
    shown = 0
    for idx in order:
        if visits[idx] == 0:
            break
        values = unindex(spec, int(idx))
        in_danger = (danger_index is not None
                     and values[danger_index] != CARDINALITY["danger_now"] - 1)
        if in_danger and "escape_dir" in spec.blocks:
            esc = values[spec.blocks.index("escape_dir")]
            if esc < 4:
                escape_total += 1
                escape_ok += int(np.argmax(q[idx]) == esc)
        if args.danger_only and not in_danger:
            continue
        if shown >= args.top:
            continue
        shown += 1
        best = int(np.argmax(q[idx]))
        qs = "  ".join(f"{a}:{q[idx, i]:+7.2f}" for i, a in enumerate(ACTIONS))
        print(f"n={int(visits[idx]):>9}  {describe(spec, values):<70s} -> {ACTIONS[best]:<5s} | {qs}")
    if escape_total:
        print(f"\nescape agreement: greedy action equals escape_dir in "
              f"{escape_ok}/{escape_total} ({escape_ok / escape_total:.0%}) of visited "
              f"danger states with a directional escape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
