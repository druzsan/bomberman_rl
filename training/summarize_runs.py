"""Tabulate finished runs from their metric logs.

Reads ``runs/*/eval/<suite>/*.json``, so it works on a run that is still going
and never re-plays a single game.

    uv run python -m training.summarize_runs --pattern 'sweep-*' --suite stage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import REPO

FIELDS = [("score_mean", "score", "6.2f"), ("score_ci95_lo", "lo", "5.2f"),
          ("score_ci95_hi", "hi", "5.2f"), ("win_rate", "win", "5.1%"),
          ("survival_rate", "surv", "5.1%"), ("suicide_rate", "suic", "5.1%"),
          ("coins_mean", "coins", "6.2f"), ("kills_mean", "kills", "5.2f"),
          ("crates_mean", "crates", "6.1f")]


def best_eval(run: Path, suite: str) -> tuple[int, dict] | None:
    files = sorted((run / "eval" / suite).glob("step_*.json"))
    best = None
    for f in files:
        agg = json.loads(f.read_text())["aggregate"]
        progress = int(f.stem.split("_")[1])
        if best is None or agg["score_mean"] > best[1]["score_mean"]:
            best = (progress, agg)
    return best


def last_eval(run: Path, suite: str) -> tuple[int, dict] | None:
    files = sorted((run / "eval" / suite).glob("step_*.json"))
    if not files:
        return None
    agg = json.loads(files[-1].read_text())["aggregate"]
    return int(files[-1].stem.split("_")[1]), agg


def overrides(run: Path) -> str:
    cfg = json.loads((run / "config.json").read_text())
    bits = list(cfg.get("overrides", [])) + list(cfg.get("stage_overrides", []))
    return " ".join(bits)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="*")
    p.add_argument("--suite", default="anchor")
    p.add_argument("--which", choices=["best", "last"], default="best")
    args = p.parse_args(argv)

    runs = sorted((REPO / "runs").glob(args.pattern))
    header = f"{'run':<34s} {'step':>9s} " + " ".join(f"{n:>6s}" for _, n, _ in FIELDS)
    print(header)
    print("-" * len(header))
    for run in runs:
        if not (run / "config.json").exists():
            continue
        got = (best_eval if args.which == "best" else last_eval)(run, args.suite)
        if got is None:
            continue
        progress, agg = got
        cells = " ".join(f"{agg[k]:>{f}}" for k, _, f in FIELDS)
        print(f"{run.name[-34:]:<34s} {progress:>9d} {cells}   {overrides(run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
