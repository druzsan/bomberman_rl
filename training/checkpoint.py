"""Atomic checkpoints and export of the shipped inference artifact.

``train_state`` (resume) and ``model.npz`` (submit) are deliberately different
objects: confusing them is how people ship a hundred megabytes of optimiser
state.  Only the second is ever loaded by ``callbacks.setup``.

The table is stored sparsely -- only states that were actually visited -- which
turns a 15 MB dense array into a few hundred kilobytes and makes the artifact
comfortably smaller than the 20 MB submission budget.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def export_model(path: Path, q: np.ndarray, n: np.ndarray, feature_set: str,
                 fold: bool) -> None:
    """Write exactly the artifact ``callbacks.setup`` loads."""
    visited = np.flatnonzero(n.sum(axis=1) > 0).astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=visited,
        values=q[visited].astype(np.float32),
        counts=n[visited].astype(np.uint32),
        feature_set=feature_set,
        fold=bool(fold),
        n_states=q.shape[0],
    )


def write_checkpoint(root: Path, progress: int, q: np.ndarray, n: np.ndarray,
                     feature_set: str, fold: bool, meta: dict) -> Path:
    """Atomically materialise one checkpoint directory."""
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{progress:010d}"
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    try:
        export_model(tmp / "model.npz", q, n, feature_set, fold)
        np.savez(tmp / "train_state.npz", Q=q, N=n)
        (tmp / "meta.json").write_text(json.dumps(
            {"progress": progress, "git_sha": git_sha(), "feature_set": feature_set,
             "fold": bool(fold), **meta}, indent=1, default=float))
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Copy, not symlink: a zip cannot carry a symlink."""
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=dst.parent))
    shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def prune(root: Path, keep_last: int = 5, keep_best: set[Path] | None = None,
          milestone_every: int = 1_000_000, keep_stages: set[int] | None = None) -> int:
    """Retention policy: last N, the best few, milestones and stage boundaries."""
    keep_best = keep_best or set()
    keep_stages = keep_stages or set()
    steps = sorted(p for p in root.glob("step_*") if p.is_dir())
    keep = set(steps[-keep_last:]) | keep_best
    for p in steps:
        progress = int(p.name.split("_")[1])
        if progress % milestone_every == 0 or progress in keep_stages:
            keep.add(p)
    removed = 0
    for p in steps:
        if p not in keep:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    return removed
