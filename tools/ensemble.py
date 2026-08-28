"""Bundle several trained checkpoints into one artifact the agent averages over.

D4 test-time augmentation is worth +0.68 points for eight network evaluations,
on the logic that asking one learned model the same question eight ways reduces
the variance of its answer.  An ensemble is the same argument with a different
axis: several *independently trained* networks answering once each.  The two
compose, and the measured budget is ~160 evaluations per step against the
shipped model's 8, so there is room for both.

The one precondition is that the members' value scales agree, because the agent
averages ``Q`` directly.  They do here -- every candidate was trained with the
same reward function and ``gamma = 0.95``, and their ``q_mean`` differs by under
2 % -- and this tool refuses members whose architecture differs.  Checkpoints
trained under *different* rewards would have to be combined by rank instead, and
this tool is the wrong one for that.

    uv run python -m tools.ensemble --out /tmp/ens.pt --tta on \\
        runs/A/checkpoints/step_0006401473/model.pt \\
        runs/B/checkpoints/step_0008000296/model.pt

Members are correlated when they share a behaviour-cloning initialisation and a
seed, which shrinks the gain; that is a reason to measure rather than a reason
not to try.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="checkpoint artifacts to bundle")
    p.add_argument("--out", required=True)
    p.add_argument("--tta", choices=["on", "off"], default=None,
                   help="set the D4 augmentation flag on the bundle")
    p.add_argument("--use-mask", choices=["on", "off"], default=None)
    args = p.parse_args(argv)

    from lib.qnet import load as load_qnet
    from lib.qnet import save_ensemble

    nets, metas = [], []
    for path in args.models:
        net, meta = load_qnet(Path(path))
        nets.append(net)
        metas.append(meta)

    # The bundle inherits the first member's meta, so a flag nobody sets keeps
    # the value the primary checkpoint shipped with rather than a fresh default.
    meta = dict(metas[0])
    meta["ensemble"] = [str(m) for m in args.models]
    meta["ensemble_size"] = len(nets)
    if args.tta is not None:
        meta["tta"] = args.tta == "on"
    if args.use_mask is not None:
        meta["use_mask"] = args.use_mask == "on"
    meta.pop("search", None)      # a searched ensemble is not a thing we measured

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble(out, nets, meta)
    size = out.stat().st_size / 1e6
    print(f"{out}: {len(nets)} members, {size:.2f} MB, "
          + json.dumps({k: meta.get(k) for k in ("tta", "use_mask", "ensemble_size")}))
    for path, m in zip(args.models, metas):
        print(f"  {path}  (step {m.get('progress', '?')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
