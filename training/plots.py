"""Report figures, derived from ``metrics.jsonl``.

Plotting is deliberately outside the training driver (``dev/plan.md`` §9.9 R6):
``metrics.jsonl`` is append-only and is the single source of truth, figures are
derived and disposable, and a broken plot must never be able to kill a
ninety-minute run.  ``matplotlib`` is therefore a *dev* dependency only and is
never imported by agent code.

    uv run python -m training.plots --run runs/<dir> [--compare runs/<other> ...] \\
        --out dev/figures --format pdf

Every figure takes any number of runs and draws one line per run, which is what
makes the ablations (danger planes on/off, BC/no BC, network size) a single
command rather than a manual assembly job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .engine import REPO

#: One entry per figure: file stem, title, and the metrics drawn on it.
FIGURES = [
    ("anchor-score", "Anchor score (vs 3 x rule_based_agent)",
     [("eval/anchor/score_mean", "score / round")], "eval"),
    ("anchor-safety", "Survival and suicide rate",
     [("eval/anchor/survival_rate", "survival"),
      ("eval/anchor/suicide_rate", "suicide")], "eval"),
    ("anchor-yield", "Coins, kills and crates per round",
     [("eval/anchor/coins_mean", "coins"), ("eval/anchor/kills_mean", "kills"),
      ("eval/anchor/crates_mean", "crates / 10")], "eval"),
    ("optimisation", "Loss and mean Q",
     [("train/loss", "loss"), ("train/q_mean", "mean Q")], "train"),
    ("schedules", "Schedules",
     [("train/eps", "epsilon"), ("train/beta", "PER beta"),
      ("train/shaping_scale", "shaping"), ("train/mask_level", "mask level")], "train"),
    ("throughput", "Throughput",
     [("train/env_steps_per_s", "env steps/s"),
      ("train/grad_steps_per_s", "grad steps/s")], "train"),
    ("think-time", "Think time p99 (tournament limit 500 ms)",
     [("eval/anchor/think_ms_p99", "p99 ms")], "eval"),
    ("stage-score", "Stage-suite score",
     [("eval/stage/score_mean", "score / round")], "eval"),
]

#: Event curves worth their own small-multiple panel.
EVENT_KEYS = ["event/KILLED_SELF_per_episode", "event/SUICIDAL_BOMB_per_episode",
              "event/USELESS_BOMB_per_episode", "event/GOOD_BOMB_per_episode",
              "event/COIN_COLLECTED_per_episode", "event/INVALID_ACTION_per_episode",
              "event/CRATE_DESTROYED_per_episode", "event/KILLED_OPPONENT_per_episode"]


def load(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def series(rows: list[dict], key: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        if r["kind"] != kind or key not in r["metrics"]:
            continue
        xs.append(r["progress"])
        ys.append(r["metrics"][key])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def band(rows: list[dict], lo_key: str, hi_key: str) -> tuple[np.ndarray, np.ndarray,
                                                              np.ndarray]:
    xs, lo, hi = [], [], []
    for r in rows:
        if r["kind"] != "eval" or lo_key not in r["metrics"]:
            continue
        xs.append(r["progress"])
        lo.append(r["metrics"][lo_key])
        hi.append(r["metrics"][hi_key])
    return np.asarray(xs, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi,
                                                                                dtype=float)


def stage_marks(rows: list[dict]) -> list[float]:
    """Progress values at which the curriculum stage changed."""
    marks, last = [], None
    for r in rows:
        stage = r["metrics"].get("train/stage")
        if stage is None:
            continue
        if last is not None and stage != last:
            marks.append(r["progress"])
        last = stage
    return marks


def draw(runs: dict[str, list[dict]], out: Path, fmt: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, title, keys, kind in FIGURES:
        fig, ax = plt.subplots(figsize=(7, 4))
        drew = False
        for name, rows in runs.items():
            for key, label in keys:
                xs, ys = series(rows, key, kind)
                if not len(xs):
                    continue
                if key.endswith("crates_mean"):
                    ys = ys / 10.0
                ax.plot(xs / 1e6, ys, marker="o" if kind == "eval" else None,
                        markersize=3, linewidth=1.2,
                        label=f"{name} {label}" if len(runs) > 1 else label)
                drew = True
            if stem == "anchor-score":
                xs, lo, hi = band(rows, "eval/anchor/score_ci95_lo",
                                  "eval/anchor/score_ci95_hi")
                if len(xs):
                    ax.fill_between(xs / 1e6, lo, hi, alpha=0.15)
            for mark in stage_marks(rows):
                ax.axvline(mark / 1e6, color="0.7", linewidth=0.6, linestyle=":")
        if not drew:
            plt.close(fig)
            continue
        if stem == "think-time":
            ax.axhline(500.0, color="crimson", linewidth=0.8, linestyle="--",
                       label="tournament limit")
        ax.set_xlabel("environment steps (millions)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out / f"{stem}.{fmt}"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    # Event small multiples.
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    drew = False
    for ax, key in zip(axes.ravel(), EVENT_KEYS):
        for name, rows in runs.items():
            xs, ys = series(rows, key, "train")
            if len(xs):
                ax.plot(xs / 1e6, ys, linewidth=1.0, label=name)
                drew = True
        ax.set_title(key.split("/")[1].replace("_per_episode", ""), fontsize=9)
        ax.grid(alpha=0.25)
    if drew:
        for ax in axes[-1]:
            ax.set_xlabel("env steps (M)")
        if len(runs) > 1:
            axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        path = out / f"events.{fmt}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, action="append",
                   help="run directory; repeat to overlay several")
    p.add_argument("--compare", nargs="*", default=[],
                   help="further run directories to overlay")
    p.add_argument("--label", nargs="*", default=[],
                   help="names for the runs, in order")
    p.add_argument("--out", default=str(REPO / "dev" / "figures"))
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = p.parse_args(argv)

    paths = [Path(r) for r in args.run + list(args.compare)]
    labels = args.label or [p.name.split("-", 2)[-1] for p in paths]
    runs = {label: load(path) for label, path in zip(labels, paths)}
    written = draw(runs, Path(args.out), args.format)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
