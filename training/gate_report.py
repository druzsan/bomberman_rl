"""Run the complete gate battery for one agent and print the table.

Every suite uses the fixed gate seed block and the stock engine with
``train = False``, i.e. the shipped ``callbacks.py`` and the shipped model file.

    uv run python -m training.gate_report --agent q_tabular_agent --rounds 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import REPO
from .evaluate import GATE_SEED_BASE, evaluate, summarise

SUITES = [
    ("task1 coin-heaven solo", "coin-heaven", []),
    ("task2 loot-crate solo", "loot-crate", []),
    ("task3 vs 3 peaceful", "classic", ["peaceful_agent"] * 3),
    ("task3 vs 3 coin_collector", "classic", ["coin_collector_agent"] * 3),
    ("task4 vs 3 rule_based", "classic", ["rule_based_agent"] * 3),
    ("held-out vs 3 bfs_expert", "classic", ["bfs_expert"] * 3),
    ("self-consistency vs 3 copies", "classic", None),
]

HEADER = (f"{'suite':<30s} {'score':>14s} {'win':>6s} {'surv':>6s} {'suic':>6s} "
          f"{'coins':>6s} {'kills':>6s} {'crates':>7s} {'p99 ms':>7s}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--workers", type=int, default=26)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    print(HEADER)
    print("-" * len(HEADER))
    table = {}
    for name, scenario, opponents in SUITES:
        if opponents is None:
            opponents = [args.agent] * 3
        records = evaluate(args.agent, opponents, scenario=scenario, rounds=args.rounds,
                           seed_base=GATE_SEED_BASE, workers=args.workers,
                           model=args.model)
        a = summarise(records)
        table[name] = a
        print(f"{name:<30s} {a['score_mean']:6.2f} "
              f"[{a['score_ci95_lo']:4.2f},{a['score_ci95_hi']:5.2f}] "
              f"{a['win_rate']:6.1%} {a['survival_rate']:6.1%} {a['suicide_rate']:6.1%} "
              f"{a['coins_mean']:6.2f} {a['kills_mean']:6.2f} {a['crates_mean']:7.1f} "
              f"{a['think_ms_p99']:7.2f}")
    out = Path(args.out) if args.out else REPO / "results" / f"gate-{args.agent}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
