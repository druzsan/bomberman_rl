"""Aggregation of per-round records into the reported metric set.

Per-round records are always kept; aggregates are derived.  That way the report
can recompute any statistic (bootstrap CIs, Wilcoxon on paired seeds) without
replaying a single game.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bootstrap_ci(values, n_resamples: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _win_credit(rec: dict) -> float:
    """Share of the win, splitting ties evenly."""
    others = rec.get("opponent_scores", [])
    if any(o > rec["score"] for o in others):
        return 0.0
    tied = sum(1 for o in others if o == rec["score"])
    return 1.0 / (1 + tied)


def aggregate(records: list[dict], prefix: str = "") -> dict[str, float]:
    """Reduce per-round records to the metric set of ``dev/plan.md`` §7.1."""
    if not records:
        return {}
    scores = np.array([r["score"] for r in records], dtype=float)
    lo, hi = bootstrap_ci(scores)
    steps = np.array([max(r["steps_survived"], 1) for r in records], dtype=float)
    invalid = np.array([r["invalid"] for r in records], dtype=float)
    out = {
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "score_ci95_lo": lo,
        "score_ci95_hi": hi,
        "score_median": float(np.median(scores)),
        "win_rate": float(np.mean([_win_credit(r) for r in records])),
        "survival_rate": float(np.mean([r["survived"] for r in records])),
        "suicide_rate": float(np.mean([r["suicide"] for r in records])),
        "coins_mean": float(np.mean([r["coins"] for r in records])),
        "kills_mean": float(np.mean([r["kills"] for r in records])),
        "crates_mean": float(np.mean([r["crates"] for r in records])),
        "bombs_mean": float(np.mean([r["bombs"] for r in records])),
        "invalid_rate": float((invalid / steps).mean()),
        "steps_survived_mean": float(steps.mean()),
        "round_steps_mean": float(np.mean([r["round_steps"] for r in records])),
        "think_ms_mean": float(np.mean([r["think_ms_mean"] for r in records])),
        "think_ms_p99": float(np.max([r["think_ms_p99"] for r in records])),
        "think_timeouts": int(np.sum([r["think_timeouts"] for r in records])),
        "rounds": len(records),
        "seed_lo": int(min(r["seed"] for r in records)),
        "seed_hi": int(max(r["seed"] for r in records)),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


def event_rates(records: list[dict]) -> dict[str, float]:
    """Mean occurrences per round of every event seen."""
    names = sorted({k for r in records for k in r.get("events", {})})
    return {f"event/{n}_per_round": float(np.mean([r["events"].get(n, 0) for r in records]))
            for n in names}


def paired_delta(a: list[dict], b: list[dict]) -> dict[str, float]:
    """Paired comparison of two arms evaluated on the same seeds."""
    by_seed_a = {r["seed"]: r["score"] for r in a}
    by_seed_b = {r["seed"]: r["score"] for r in b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    if not seeds:
        return {}
    d = np.array([by_seed_a[s] - by_seed_b[s] for s in seeds], dtype=float)
    lo, hi = bootstrap_ci(d)
    n_nonzero = int((d != 0).sum())
    out = {"n_paired": len(seeds), "delta_mean": float(d.mean()),
           "delta_ci95_lo": lo, "delta_ci95_hi": hi, "n_nonzero": n_nonzero}
    try:
        from scipy.stats import wilcoxon

        if n_nonzero:
            out["wilcoxon_p"] = float(wilcoxon(d).pvalue)
    except ImportError:
        out["wilcoxon_p"] = float("nan")
    return out


class JsonlSink:
    """Append-only metric log: the single source of truth for a run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # noqa: SIM115 - long-lived sink

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        self._fh.close()
