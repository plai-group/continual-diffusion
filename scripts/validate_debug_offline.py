"""
Run the issue-58 debug validation against a saved checkpoint.

Exists so a validation-set correction does not require retraining: point it at
an existing checkpoint and the right DB, and it reproduces the overlays and
val/video/* metrics offline.

  python scripts/validate_debug_offline.py \
      --checkpoint checkpoints/<run>/ema_0.999_100000.pt \
      --db   /ubc/.../plaicraft-model-pi0/data/debug_v2/validation_debug_static_40x24.db \
      --root /ubc/.../plaicraft-data-preprocessing/processed/debug_v2 \
      --out  results/validation_offline
"""

import argparse
import json
from pathlib import Path

import torch as th

from improved_diffusion import dist_util
from improved_diffusion.debug_validation import DebugValidationSet, run_debug_validation
from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="results/validation_offline")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk_size", type=int, default=3)
    args = p.parse_args()

    data = dist_util.load_state_dict(args.checkpoint, map_location="cpu")
    margs = dict(data["config"])
    margs.setdefault("model_type", "vdt")
    ns = argparse.Namespace(**margs)
    model, diffusion = create_model_and_diffusion(
        model_type=ns.model_type,
        **args_to_dict(ns, model_and_diffusion_defaults(model_type=ns.model_type).keys()),
    )
    model.load_state_dict(data["state_dict"])
    model = model.to(args.device).eval()
    step = data.get("step", 0)
    print(f"loaded {args.checkpoint} (step {step}), T={ns.T}")

    valset = DebugValidationSet(args.db, args.root, T=ns.T, n_observed=ns.T // 2)
    print(f"validation rows: {len(valset.rows)}")
    for r in valset.rows:
        print(f"  {r['num']:2d} {r['prompt'][:38]:38s} R_start={r['r_start_ms']:7d} win={r['window_start']}")

    res = run_debug_validation(
        model, diffusion, valset, args.device, out_dir=args.out,
        step=step, chunk_size=args.chunk_size, log_videos=True,
    )
    print("\nAGGREGATE")
    for k, v in sorted(res["aggregate"].items()):
        print(f"  {k:26s} {v:.4f}")

    out = Path(args.out) / f"report_step{step}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
