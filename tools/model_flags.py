"""Read or change the inference flags stored inside a model artifact.

Two of the deep agent's inference-time behaviours are *not* learned parameters
and must be decidable without retraining:

``use_mask``
    the S5a proven-lethal action mask (``dev/plan.md`` §S5a and §15.3);
``tta``
    D4 test-time augmentation, averaging the network over all eight symmetry
    images of the board (S6).

Both live in the artifact's ``meta`` block, so flipping one is a file edit
rather than a training run -- which is what makes the ablations cheap and what
lets the ship/no-ship decision be made on the final measured numbers.

    uv run python -m tools.model_flags runs/.../model.pt
    uv run python -m tools.model_flags runs/.../model.pt --tta on --out /tmp/tta.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FLAGS = ("use_mask", "tta")


def read(path: Path) -> dict:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    return dict(blob.get("meta", {}))


def write(path: Path, out: Path, changes: dict) -> dict:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    meta = dict(blob.get("meta", {}))
    meta.update(changes)
    blob["meta"] = meta
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    for flag in FLAGS:
        p.add_argument(f"--{flag.replace('_', '-')}", choices=["on", "off"], default=None)
    p.add_argument("--out", default=None,
                   help="write here instead of editing the file in place")
    args = p.parse_args(argv)

    path = Path(args.model)
    changes = {f: getattr(args, f) == "on" for f in FLAGS if getattr(args, f) is not None}
    if not changes:
        print(json.dumps(read(path), indent=1, default=str))
        return 0
    out = Path(args.out) if args.out else path
    if out != path and path.suffix == out.suffix and not out.exists():
        shutil.copy2(path, out)
    meta = write(path, out, changes)
    print(f"{out}: " + json.dumps({k: meta.get(k, False) for k in FLAGS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
