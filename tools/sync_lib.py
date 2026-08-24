"""Vendor ``lib/`` into every agent directory that uses it.

The submission must be a single self-contained directory, but all agents want
the same primitives.  Keeping one source of truth at the repository root and
copying it into ``agent_code/<name>/lib/`` means agent code always says
``from .lib import danger`` -- identical in development and in the shipped zip,
with no import rewriting and no symlinks (a zip cannot carry one).

    uv run python -m tools.sync_lib            # copy
    uv run python -m tools.sync_lib --check    # fail if any copy is stale
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "lib"
AGENT_CODE = REPO / "agent_code"
STAMP = "lib"


def agents_using_lib() -> list[Path]:
    """Agent directories that already contain a vendored ``lib/``, plus opt-ins."""
    marker = "USES_LIB"
    out = []
    for d in sorted(AGENT_CODE.iterdir()):
        if not d.is_dir():
            continue
        if (d / STAMP).exists() or (d / marker).exists():
            out.append(d)
    return out


def sync(target: Path, check: bool = False) -> bool:
    dest = target / STAMP
    if check:
        if not dest.exists():
            return False
        cmp = filecmp.dircmp(SOURCE, dest, ignore=["__pycache__"])
        return not (cmp.left_only or cmp.right_only or cmp.diff_files)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SOURCE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify, do not write")
    parser.add_argument("--agents", nargs="*", help="restrict to these agent names")
    args = parser.parse_args(argv)

    targets = agents_using_lib()
    if args.agents:
        wanted = set(args.agents)
        targets = [t for t in targets if t.name in wanted]
        for name in sorted(wanted - {t.name for t in targets}):
            targets.append(AGENT_CODE / name)

    stale = []
    for t in targets:
        ok = sync(t, check=args.check)
        if args.check and not ok:
            stale.append(t.name)
        elif not args.check:
            print(f"synced lib -> {t.relative_to(REPO)}/{STAMP}")
    if stale:
        print(f"stale vendored lib in: {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
