"""Vendor ``lib/`` into every agent directory that uses it.

The submission must be a single self-contained directory, but all agents want
the same primitives.  Keeping one source of truth at the repository root and
copying it into ``agent_code/<name>/lib/`` means agent code always says
``from .lib import danger`` -- identical in development and in the shipped zip,
with no import rewriting and no symlinks (a zip cannot carry one).

    uv run python -m tools.sync_lib            # copy
    uv run python -m tools.sync_lib --check    # fail if any copy is stale

One module is *not* copied everywhere.  ``lib/qnet.py`` imports torch at module
level, and vendoring it into an agent that never touches it makes the shipped
tree claim a dependency it does not have: ``check_submission`` reads the imports
out of the whole directory, so a pure-numpy agent ended up with ``torch`` in its
``requirements.txt`` and failing the "does it call ``set_num_threads``" check.
So torch-only modules follow the agent that imports them.
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

#: Modules vendored only into agents whose own code imports them, keyed by the
#: name that has to appear in an agent's sources.
OPTIONAL = {"qnet.py": "qnet"}


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


def skipped_for(target: Path) -> set[str]:
    """Optional modules this agent does not import, and must not be given."""
    sources = "\n".join(p.read_text() for p in target.glob("*.py"))
    return {name for name, token in OPTIONAL.items() if token not in sources}


def sync(target: Path, check: bool = False) -> bool:
    dest = target / STAMP
    skip = skipped_for(target)
    if check:
        if not dest.exists():
            return False
        cmp = filecmp.dircmp(SOURCE, dest, ignore=["__pycache__"])
        present = {p.name for p in dest.iterdir()}
        missing = set(cmp.left_only) - skip
        unwanted = set(OPTIONAL) & skip & present
        return not (missing or unwanted or set(cmp.right_only) or cmp.diff_files)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SOURCE, dest,
                    ignore=shutil.ignore_patterns("__pycache__", *skip))
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
