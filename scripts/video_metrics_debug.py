"""
Issue #58 offline distributional metrics for the plaicraft-debug VDT run.

Scores generated continuations against held-out *generated* sessions (the 20
sessions debug_dataset.py withholds), not the validation recording -- so this
measures "can VDT model this world" without the OBS-vs-headless domain gap, and
without depending on actions the model was never shown.

FVD is deliberately not used: frechet_video_distance.py needs tensorflow==2.15
(no py3.12 wheel), TF1 hub.Module, and the retired tfhub.dev. JEDi is this
repo's current, torch-only equivalent.

Usage:
  python scripts/video_metrics_debug.py --checkpoint checkpoints/ema_0.9999_NNNN.pt \\
      --num_videos 256 --out results/metrics_debug.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch as th

from improved_diffusion import dist_util
from improved_diffusion.debug_dataset import ChunkedDebugDataset
from improved_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def load_model(checkpoint_path, device):
    data = dist_util.load_state_dict(checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    margs = dict(data["config"])
    margs.setdefault("model_type", "vdt")
    ns = argparse.Namespace(**margs)
    model, diffusion = create_model_and_diffusion(
        model_type=ns.model_type,
        **args_to_dict(ns, model_and_diffusion_defaults(model_type=ns.model_type).keys()),
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval(), diffusion, ns


@th.no_grad()
def generate(model, diffusion, dataset, device, num_videos, batch_size, T, n_obs):
    """Return (gt, pred) uint8 arrays of shape (N, T, 3, H, W)."""
    n = min(num_videos, len(dataset))
    gts, preds = [], []
    for lo in range(0, n, batch_size):
        idxs = range(lo, min(lo + batch_size, n))
        x0 = th.stack([dataset[i][0] for i in idxs]).to(device)
        b = x0.shape[0]
        obs = th.zeros(b, T, 1, 1, 1, device=device)
        obs[:, :n_obs] = 1.0
        lat = th.zeros_like(obs)
        lat[:, n_obs:] = 1.0

        s, _ = diffusion.heun_sample(
            model, x0.shape, clip_denoised=True,
            model_kwargs={"frame_indices": None, "x0": x0,
                          "obs_mask": obs, "latent_mask": lat},
            latent_mask=lat.cpu(), return_decoded=False,
        )
        s = s.to(device) * lat + x0 * obs
        to8 = lambda t: ((t + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
        gts.append(to8(x0))
        preds.append(to8(s))
        print(f"  sampled {min(lo + batch_size, n)}/{n}", flush=True)
    return np.concatenate(gts), np.concatenate(preds)


def compute_fid(gt, pred, device, batch_size=64):
    """FID over all frames. Only the generated half is scored on the pred side."""
    from pytorch_fid.inception import InceptionV3
    from scipy import linalg

    block = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    for p in block.parameters():
        p.requires_grad_(False)

    def feats(arr):
        # (N, T, 3, H, W) uint8 -> (N*T, 2048)
        x = th.from_numpy(arr).float().div(255.0)
        x = x.reshape(-1, *x.shape[2:])
        out = []
        for i in range(0, x.shape[0], batch_size):
            b = x[i : i + batch_size].to(device)
            b = th.nn.functional.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            out.append(block(b)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    f1, f2 = feats(gt), feats(pred)
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2.mean(0), np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def compute_jedi(gt, pred, feature_path, num_videos, batch_size=16):
    """JEDi (V-JEPA feature space). Needs the V-JEPA weights cached locally."""
    from videojedi import JEDiMetric

    class _Arr(th.utils.data.Dataset):
        def __init__(self, a):
            self.a = a
        def __len__(self):
            return len(self.a)
        def __getitem__(self, i):
            return th.from_numpy(self.a[i]).float().div(255.0), 0

    coll = lambda batch: (th.stack([b[0] for b in batch]), {})
    mk = lambda a: th.utils.data.DataLoader(_Arr(a), batch_size=batch_size,
                                            shuffle=False, collate_fn=coll)
    j = JEDiMetric(feature_path=str(feature_path),
                   model_dir=os.environ.get("JEDI_MODEL_DIR", "."))
    j.load_features(mk(gt), mk(pred), num_samples=num_videos)
    return float(j.compute_metric())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_videos", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--out", default="results/metrics_debug.json")
    p.add_argument("--device", default="cuda")
    # Off by default: JEDi needs a 10.4GB V-JEPA ViT-H download and would score
    # 24x40 frames upsampled ~9x, so its absolute value is not meaningful here.
    p.add_argument("--jedi", action="store_true", help="also compute JEDi (downloads V-JEPA)")
    args = p.parse_args()

    model, diffusion, margs = load_model(args.checkpoint, args.device)
    T = margs.T
    n_obs = T // 2

    root = os.environ.get(
        "DEBUG_TOY_ROOT",
        "/ubc/cs/research/plai-scratch/ctardy/projects/plaicraft-data-preprocessing/processed/vdt_corpus/debug_24x40",
    )
    ds = ChunkedDebugDataset(root, window_length=T)
    ds.set_test()
    print(f"held-out test windows: {len(ds)}")

    gt, pred = generate(model, diffusion, ds, args.device,
                        args.num_videos, args.batch_size, T, n_obs)
    print(f"generated {gt.shape[0]} videos {gt.shape[1:]}")

    res = {"checkpoint": args.checkpoint, "num_videos": int(gt.shape[0]),
           "T": int(T), "n_observed": int(n_obs)}

    # Score only the generated half; including observed frames would flatter the model.
    res["fid_generated_half"] = compute_fid(gt[:, n_obs:], pred[:, n_obs:], args.device)
    print("FID (generated half):", res["fid_generated_half"])

    if args.jedi:
        try:
            out = Path(args.out).parent / "jedi_features"
            out.mkdir(parents=True, exist_ok=True)
            res["jedi"] = compute_jedi(gt, pred, out, int(gt.shape[0]))
            print("JEDi:", res["jedi"])
        except Exception as e:
            # V-JEPA weights need a network fetch the compute nodes may not have.
            res["jedi_error"] = repr(e)
            print("JEDi failed:", repr(e))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
