"""Pick the submission checkpoint by a gate-quality evaluation, not by loss.

The driver promotes a checkpoint only when it beats the incumbent by more than
noise, which is the right rule *during* a run but leaves the final choice
between several statistically indistinguishable candidates. This re-plays the
top candidates on the fixed gate seed block -- a different, larger seed set than
the one used for monitoring, so the choice is not made on the same data that
produced the ranking.

    uv run python -m training.select_checkpoint --run runs/<dir> --rounds 500
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .checkpoint import copy_checkpoint
from .engine import REPO
from .evaluate import GATE_SEED_BASE, evaluate, format_table, summarise

#: The Task-4 gate of ``dev/plan.md`` §7.3. Suicide rate is the single most
#: diagnostic number in this game, so the default is to pick the highest scoring
#: checkpoint *that also passes the gate* rather than the highest scoring one
#: outright -- the two are usually within each other's confidence interval.
GATE = {"score_mean": (5.0, None), "win_rate": (0.45, None),
        "suicide_rate": (None, 0.08), "think_ms_p99": (None, 50.0)}


def gate_failures(agg: dict) -> list[str]:
    out = []
    for key, (lo, hi) in GATE.items():
        value = agg.get(key)
        if lo is not None and value < lo:
            out.append(f"{key}={value:.3f}<{lo}")
        if hi is not None and value > hi:
            out.append(f"{key}={value:.3f}>{hi}")
    return out


def artifact(path: Path) -> Path:
    """The inference artifact inside a checkpoint directory.

    S1/S2 export ``model.npz`` and S3 exports ``model.pt``; the selection logic
    is identical, so it asks the checkpoint what it holds rather than being told.
    """
    for name in ("model.npz", "model.pt"):
        if (path / name).exists():
            return path / name
    raise FileNotFoundError(f"no inference artifact in {path}")


def candidates(run: Path, top: int, every: bool = False) -> list[tuple[int, Path]]:
    """Checkpoints that still exist on disk, best monitoring score first.

    ``every`` ignores the monitoring ranking and takes them all, which is what
    is needed when a run's in-flight evaluation stalled or was never enabled:
    the ranking then has to come from this pass itself.
    """
    if every:
        out = [(int(p.name.split("_")[1]), p)
               for p in sorted((run / "checkpoints").glob("step_*")) if p.is_dir()]
        for name in ("best", "last"):
            path = run / "checkpoints" / name
            if path.exists() and all(path != q for _, q in out):
                meta = json.loads((path / "meta.json").read_text())
                out.append((int(meta.get("progress", -1)), path))
        return out
    scored: dict[int, float] = {}
    for f in sorted((run / "eval" / "anchor").glob("step_*.json")):
        progress = int(f.stem.split("_")[1])
        scored[progress] = json.loads(f.read_text())["aggregate"]["score_mean"]
    out = []
    for progress in sorted(scored, key=lambda p: -scored[p]):
        path = run / "checkpoints" / f"step_{progress:010d}"
        if path.exists():
            out.append((progress, path))
        if len(out) >= top:
            break
    for name in ("best", "last"):
        path = run / "checkpoints" / name
        if path.exists() and all(path != p for _, p in out):
            meta = json.loads((path / "meta.json").read_text())
            out.append((int(meta.get("progress", -1)), path))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--workers", type=int, default=26)
    p.add_argument("--agent", default=None)
    p.add_argument("--install", action="store_true",
                   help="copy the winner's model.npz into the agent directory")
    p.add_argument("--ignore-gate", action="store_true",
                   help="select purely by score, even if the gate fails")
    p.add_argument("--all", action="store_true",
                   help="evaluate every checkpoint on disk instead of the "
                        "top --top by monitoring score")
    args = p.parse_args(argv)

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())["config"]
    agent = args.agent or cfg["agent"]
    results = []
    for progress, path in candidates(run, args.top, every=args.all):
        records = evaluate(agent, ["rule_based_agent"] * 3, scenario="classic",
                           rounds=args.rounds, seed_base=GATE_SEED_BASE,
                           workers=args.workers, model=str(artifact(path)))
        agg = summarise(records)
        failures = gate_failures(agg)
        print(format_table(f"{path.name} (step {progress})", agg))
        print(f"  gate       {'PASS' if not failures else 'fail: ' + ', '.join(failures)}")
        results.append((agg["score_mean"], progress, path, agg, records, failures))

    results.sort(key=lambda r: -r[0])
    passing = [r for r in results if not r[5]]
    chosen = results if (args.ignore_gate or not passing) else passing
    score, progress, path, agg, records, failures = chosen[0]
    if not passing and not args.ignore_gate:
        print("\nno candidate passed the gate; selecting by score")
    print(f"\nselected {path} (step {progress}) with {score:.2f} points/round"
          f" ({'gate passed' if not failures else 'gate failed: ' + ', '.join(failures)})")
    out = REPO / "results" / f"{run.name}-gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": run.name, "selected": str(path),
                               "progress": progress, "aggregate": agg,
                               "records": records}, indent=1, default=float))
    print(f"wrote {out}")
    if args.install:
        copy_checkpoint(path, run / "checkpoints" / "selected")
        source = artifact(path)
        target = REPO / "agent_code" / agent / source.name
        shutil.copy2(source, target)
        print(f"installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
