"""Gate-quality, in-environment evaluation of an agent or a checkpoint.

A checkpoint's score is measured by playing real games in the stock engine with
``train = False`` and the *shipped* ``callbacks.py``, never inferred from a
training loss.  That catches the classic bug where training-time and
inference-time feature extraction drift apart: what we measure is exactly what
we submit.

    uv run python -m training.evaluate --agent bfs_expert --rounds 300
    uv run python -m training.evaluate --agent q_tabular_agent \
        --model runs/.../checkpoints/best/model.npz --opponents rule_based_agent:3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from .engine import (
    REPO,
    ThinkTimer,
    assert_stock_settings,
    make_world,
    play_round,
    quiet_logging,
    set_model_env,
)
from .metrics import aggregate, event_rates

#: Fixed forever, so any two checkpoints -- from the same run or different
#: models -- are compared on identical boards.
MONITOR_SEED_BASE = 10_000
GATE_SEED_BASE = 20_000


def parse_opponents(spec: list[str]) -> list[str]:
    """``["rule_based_agent:3"]`` -> three opponents."""
    out: list[str] = []
    for item in spec:
        if ":" in item:
            name, count = item.rsplit(":", 1)
            out.extend([name] * int(count))
        else:
            out.append(item)
    return out


def _run_slice(job) -> list[dict]:
    agent, opponents, scenario, seeds, model, _worker_id, strict = job
    quiet_logging()
    if strict:
        assert_stock_settings()
    set_model_env(model)
    log_dir = REPO / "logs" / f"eval-{os.getpid()}"
    specs = [(agent, False)] + [(o, False) for o in opponents]
    world = make_world(specs, scenario=scenario, log_dir=str(log_dir))
    with ThinkTimer(world.agents[0].name) as timer:
        return [play_round(world, s, timer=timer).as_dict() for s in seeds]


#: A pool whose worker is killed never returns from ``map``, and a training run
#: whose evaluation thread is stuck in one silently stops producing monitoring
#: points for the rest of its life.  Measured once, at some cost.  Every
#: evaluation therefore has a deadline -- but a *generous* one: the deadline
#: exists to escape a hang, not to bound legitimate work, and an over-tight
#: value simply converts slow suites into failed ones.  The self-consistency
#: suite of four D4-averaged deep agents needs ~13 minutes at 400 rounds.
EVAL_TIMEOUT_S = 3600


def evaluate(agent: str, opponents: list[str], *, scenario: str = "classic",
             rounds: int = 100, seed_base: int = MONITOR_SEED_BASE, workers: int = 8,
             model: str | None = None, strict: bool = True,
             pool: mp.pool.Pool | None = None,
             timeout: float = EVAL_TIMEOUT_S) -> list[dict]:
    """Play ``rounds`` fixed-seed rounds and return the per-round records."""
    seeds = [seed_base + i for i in range(rounds)]
    workers = max(1, min(workers, rounds))
    chunks = [seeds[i::workers] for i in range(workers)]
    jobs = [(agent, opponents, scenario, chunk, model, i, strict)
            for i, chunk in enumerate(chunks) if chunk]
    if len(jobs) == 1:
        return _run_slice(jobs[0])
    if pool is not None:
        parts = pool.map_async(_run_slice, jobs).get(timeout)
    else:
        ctx = mp.get_context("spawn")
        p = ctx.Pool(len(jobs))
        try:
            parts = p.map_async(_run_slice, jobs).get(timeout)
        finally:
            # ``terminate`` rather than ``close``: closing a pool whose worker
            # has died waits for it forever, which is the failure this timeout
            # exists to escape.  After a successful map the workers are idle and
            # terminating them costs nothing.
            #
            # ``terminate`` alone is not enough either.  A worker interrupted
            # while it holds the result queue's write lock ignores SIGTERM and
            # ``Pool.join`` then blocks in ``futex_wait`` forever -- the same
            # hang, one function further on.  Measured, after the first version
            # of this fix deadlocked the gate battery.  So: terminate, give each
            # worker a moment, then ``SIGKILL`` whatever is left.
            p.terminate()
            for worker in getattr(p, "_pool", ()):
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
            p.join()
    records = [r for part in parts for r in part]
    records.sort(key=lambda r: r["seed"])
    return records


def summarise(records: list[dict]) -> dict:
    return {**aggregate(records), **event_rates(records)}


def format_table(name: str, agg: dict) -> str:
    return (
        f"{name}\n"
        f"  score      {agg['score_mean']:6.2f}  "
        f"[{agg['score_ci95_lo']:.2f}, {agg['score_ci95_hi']:.2f}]  "
        f"median {agg['score_median']:.1f}\n"
        f"  win rate   {agg['win_rate']:6.1%}   survival {agg['survival_rate']:6.1%}   "
        f"suicide {agg['suicide_rate']:6.1%}\n"
        f"  coins      {agg['coins_mean']:6.2f}   kills {agg['kills_mean']:6.2f}   "
        f"crates {agg['crates_mean']:6.1f}   bombs {agg['bombs_mean']:6.1f}\n"
        f"  invalid    {agg['invalid_rate']:6.2%}   steps alive {agg['steps_survived_mean']:6.1f}"
        f"   round len {agg['round_steps_mean']:6.1f}\n"
        f"  think ms   mean {agg['think_ms_mean']:.3f}  p99 {agg['think_ms_p99']:.3f}  "
        f"timeouts {agg['think_timeouts']}  ({agg['rounds']} rounds, "
        f"seeds {agg['seed_lo']}..{agg['seed_hi']})"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--opponents", nargs="*", default=["rule_based_agent:3"])
    p.add_argument("--scenario", default="classic")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=MONITOR_SEED_BASE)
    p.add_argument("--gate", action="store_true",
                   help=f"use the gate seed block ({GATE_SEED_BASE}...)")
    p.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 4))
    p.add_argument("--model", default=None, help="path to a checkpoint artifact")
    p.add_argument("--out", default=None, help="write raw records + aggregate here")
    p.add_argument("--no-strict", action="store_true",
                   help="skip the stock-settings assertion (curriculum debugging only)")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    records = evaluate(
        args.agent, parse_opponents(args.opponents), scenario=args.scenario,
        rounds=args.rounds, seed_base=GATE_SEED_BASE if args.gate else args.seed_base,
        workers=args.workers, model=args.model, strict=not args.no_strict,
    )
    agg = summarise(records)
    label = f"{args.agent} vs {' '.join(parse_opponents(args.opponents))} [{args.scenario}]"
    print(format_table(label, agg))
    print(f"  wall {time.perf_counter() - t0:.1f}s")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"agent": args.agent, "opponents": args.opponents,
                                   "scenario": args.scenario, "model": args.model,
                                   "aggregate": agg, "records": records}, indent=1,
                                  default=float))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
