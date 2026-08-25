"""Play one lineup of four agents against each other and rank them.

``training.evaluate`` measures one agent against fixed opponents. This reports
every seat, so a single set of rounds ranks the whole field on identical boards
-- which is the only way to compare our agents to each other honestly, since
"score against three rule_based_agents" says nothing about how two of our agents
fare when they have to share the nine coins.

Seating is rotated across the four start corners so a corner cannot favour one
agent.

    uv run python -m training.tournament --agents q_tabular_agent q_linear_agent \
        bfs_expert rule_based_agent --rounds 300
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from .engine import REPO, make_world, play_round, quiet_logging
from .evaluate import GATE_SEED_BASE
from .metrics import bootstrap_ci


def _run_slice(job) -> list[tuple[int, list[int]]]:
    """Play a slice of seeds with the lineup rotated by ``rotate`` seats."""
    agents, scenario, seeds, rotate = job
    quiet_logging()
    n = len(agents)
    lineup = [agents[(i + rotate) % n] for i in range(n)]
    world = make_world([(a, False) for a in lineup], scenario=scenario,
                       log_dir=str(REPO / "logs" / f"tourney-{os.getpid()}"))
    out = []
    for seed in seeds:
        record = play_round(world, seed, me=0).as_dict()
        seat_scores = [record["score"], *record["opponent_scores"]]
        # Undo the rotation: seat i held agent (i + rotate) % n.
        scores = [0] * n
        for seat, score in enumerate(seat_scores):
            scores[(seat + rotate) % n] = score
        out.append((seed, scores))
    return out


def _win_shares(scores: list[int]) -> list[float]:
    best = max(scores)
    tied = sum(1 for s in scores if s == best)
    return [1.0 / tied if s == best else 0.0 for s in scores]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs=4, required=True)
    p.add_argument("--scenario", default="classic")
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    n = len(args.agents)
    seeds = [GATE_SEED_BASE + i for i in range(args.rounds)]
    per_rotation = max(1, args.workers // n)
    jobs = []
    for rotate in range(n):
        chunk = seeds[rotate::n]
        for w in range(per_rotation):
            part = chunk[w::per_rotation]
            if part:
                jobs.append((tuple(args.agents), args.scenario, part, rotate))

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(len(jobs), args.workers)) as pool:
        parts = pool.map(_run_slice, jobs)

    rounds = [row for part in parts for row in part]
    scores = np.array([s for _, s in rounds], dtype=float)
    wins = np.array([_win_shares(s) for _, s in rounds], dtype=float)

    print(f"{'agent':<24s} {'score':>7s} {'ci95':>16s} {'win rate':>9s} {'rounds':>7s}")
    print("-" * 68)
    table = {}
    for i, name in enumerate(args.agents):
        lo, hi = bootstrap_ci(scores[:, i])
        table[name] = {"score_mean": float(scores[:, i].mean()), "ci95_lo": lo,
                       "ci95_hi": hi, "win_rate": float(wins[:, i].mean()),
                       "rounds": int(scores.shape[0])}
        print(f"{name:<24s} {scores[:, i].mean():7.2f} [{lo:6.2f},{hi:6.2f}] "
              f"{wins[:, i].mean():9.1%} {scores.shape[0]:7d}")
    out = Path(args.out) if args.out else REPO / "results" / "tournament.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
