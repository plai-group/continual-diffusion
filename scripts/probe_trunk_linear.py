"""plaicraft-debug#106: ridge probe of a frozen VDT trunk's action-token features -> per-tick km_ctx14 code.

Video is clean (t=0) or pure noise (t=max); every action tick is pure noise at the max action timestep, so no GT action is visible.

  python scripts/probe_trunk_linear.py --checkpoint checkpoints/<run>/ema_0.9999_650000.pt --video_t 0 --out results/probe_106/x
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch as th

import improved_diffusion.debug_validation as dv
from improved_diffusion import action_ce, debug_actions, dist_util
from improved_diffusion.corpus_validation import CorpusValidationSet
from improved_diffusion.gaussian_diffusion import ModelMeanType
from improved_diffusion.script_util import (
    args_to_dict, backfill_action_encoding, backfill_action_tick_offset, backfill_attention, backfill_attn_window,
    backfill_cond_combine, backfill_label_embedding_frozen, backfill_label_init_std, backfill_pos_encoding,
    backfill_ticks_per_latent, create_model_and_diffusion, model_and_diffusion_defaults,
)
from improved_diffusion.video_datasets import load_data, plaicraft_valset_kwargs, video_data_paths_dict

LAMS = 10.0 ** np.arange(-6, 3)  # ridge strength relative to n, on unit-variance features


@th.no_grad()
def extract(model, diffusion, x0, y, video_t0, gen, bf16=False):
    """Action-token hidden states (embed + every block, (L, B, (T-1)*k, D)) and the head's own one-step x0 action ((B, (T-1)*k, P)); all ticks pure noise at t_max."""
    B, T = x0.shape[:2]
    k, N, dev = model.ticks_per_latent, model.x_embedder.num_patches, x0.device
    t = diffusion.sigma2timestep(float(diffusion.timestep2sigma(diffusion.num_timesteps - 1)))  # the sampler's first step
    scale = diffusion.timestep2sigma(t) * diffusion.sqrt_alphas_cumprod[t]  # sampler: scaled_x = (randn * sigma) * sqrt(abar)
    x, a = (th.randn(*s, device=dev, generator=gen) * scale for s in (x0.shape, (B, T, model.action_dim)))
    ts = diffusion._scale_timesteps(th.full((B,), t, device=dev))
    kw = dict(frame_indices=None, actions=a, y=y)
    if video_t0:  # IDM path: clean GT video at noise level 0, actions at the sampler's level
        kw.update(x0=x0, obs_mask=th.ones(B, T, 1, 1, 1, device=dev), latent_mask=th.zeros(B, T, 1, 1, 1, device=dev))
        fwd = dv._IDMWrapper(model, True)
    else:
        fwd = lambda x_, ts_, **kw_: model(x_, timesteps=ts_, action_timesteps=ts_, **kw_)
    feats, head = [], {}
    hs = [model.blocks[0].register_forward_pre_hook(lambda m, i: feats.append(i[0][:, N:N + k].float().clone()))]
    hs += [b.register_forward_hook(lambda m, i, o: feats.append(o[:, N:N + k].float().clone())) for b in model.blocks]
    hs.append(model.action_head.register_forward_pre_hook(lambda m, i: head.update(args=i)))
    try:
        with th.autocast(dev.type, dtype=th.bfloat16, enabled=bf16 and dev.type == "cuda"):
            fwd(x, ts, **kw)
    finally:
        [h.remove() for h in hs]
    with th.autocast(dev.type, enabled=False):  # fp32 head: x0 = 158 * (x_t - eps) at t_max amplifies bf16 eps error
        eps = model.action_head(*(h.float() for h in head["args"])).reshape(B, T, -1)
    tt = th.full((B,), t, device=dev)
    x0a = diffusion._predict_xstart_from_eps(a, tt, eps) if diffusion.model_mean_type == ModelMeanType.EPSILON else eps
    f = th.stack(feats).reshape(len(feats), B, T, k, -1)[:, :, 1:].reshape(len(feats), B, (T - 1) * k, -1)
    return f, x0a.reshape(B, T * k, -1)[:, k:]


class Ridge:
    """Streaming closed-form ridge per layer: fp64 sufficient statistics, features standardised analytically."""

    def __init__(self, L, D, P, dev):
        self.n, self.sy = 0, th.zeros(P, dtype=th.float64, device=dev)
        self.sx, self.sxx, self.sxy = (th.zeros(s, dtype=th.float64, device=dev) for s in ((L, D), (L, D, D), (L, D, P)))

    def add(self, X, Y):  # X (L, n, D), Y (n, P)
        X, Y = X.double(), Y.double()
        self.n += Y.shape[0]
        self.sx += X.sum(1)
        self.sxx += X.transpose(1, 2) @ X
        self.sxy += X.transpose(1, 2) @ Y
        self.sy += Y.sum(0)

    def fit(self, lam):  # lam (L,) relative strength; returns (mu, sd, W, ybar)
        n, mu, yb = self.n, self.sx / self.n, self.sy / self.n
        sd = (self.sxx.diagonal(dim1=1, dim2=2) / n - mu ** 2).clamp_min(0).sqrt()
        sd = sd.clamp_min(1e-5 * sd.mean(1, keepdim=True) + 1e-12)
        G = (self.sxx - n * mu[:, :, None] * mu[:, None]) / (sd[:, :, None] * sd[:, None])
        b = (self.sxy - n * mu[:, :, None] * yb) / sd[:, :, None]
        W = th.linalg.solve(G + (lam.double() * n)[:, None, None] * th.eye(G.shape[-1], dtype=G.dtype, device=G.device), b)
        return mu, sd, W, yb

    def merge(self, o):
        self.n += o.n
        for f in ("sx", "sxx", "sxy", "sy"):
            getattr(self, f).add_(getattr(o, f))


def predict(fit, X):  # X (L, n, D) -> (L, n, P)
    mu, sd, W, yb = fit
    return ((X.double() - mu[:, None]) / sd[:, None]) @ W + yb


def load_model(path, device):
    data = dist_util.load_state_dict(path, map_location="cpu")
    ns = Namespace(**dict(data["config"]))
    ns.model_type = getattr(ns, "model_type", "vdt")
    for f in (backfill_action_encoding, backfill_cond_combine, backfill_label_embedding_frozen, backfill_label_init_std, backfill_attn_window,
              backfill_pos_encoding, backfill_attention, backfill_ticks_per_latent, backfill_action_tick_offset):
        f(ns)
    model, diffusion = create_model_and_diffusion(
        model_type=ns.model_type, **args_to_dict(ns, model_and_diffusion_defaults(model_type=ns.model_type).keys()))
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval(), diffusion, ns, data.get("step", 0)


def train_batches(ns, split, n, bs, workers, seed):
    """(frames, codes, player) batches of n clips drawn at random from the corpus's train or test split."""
    try:  # load_data is a generator, so its return_dataset=True value rides on StopIteration
        next(load_data(ns.dataset, None, T=ns.T, return_dataset=True, action_encoding=ns.action_encoding, tokenizer_checkpoint=ns.km_tokenizer_checkpoint,
                       num_classes=ns.num_classes, global_database_path=ns.global_database_path or None))
    except StopIteration as e:
        ds = e.value
    for _, _, path in ds.file_boundaries:  # build any missing action-code cache here on the GPU: DataLoader workers are forked and cannot use CUDA
        ds._get_action_arrays(path.parent.parent)
    if split == "test":
        ds.set_test()
    idx = np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False).tolist()
    for fr, _, kp, _, pl in th.utils.data.DataLoader(th.utils.data.Subset(ds, idx), batch_size=bs, num_workers=workers):
        yield fr, kp, pl


def ticks(codes):  # (B, T, k*32) -> (B, T*k, 32)
    return codes.reshape(codes.shape[0], -1, 32)


def score(tok, pred, gt, key_true, q, snap, dev):
    """Per-row action metrics averaged over rows, decoded exactly as run_debug_validation does (gt: GT codes; key_true None -> their decoded keys)."""
    out, rows = [], []
    for i in range(0, len(gt), 100):
        g, p = gt[i:i + 100].to(dev), pred[i:i + 100].to(dev)
        p = debug_actions.quantize_km_ctx14(p) if snap else p
        gk, gm, gp = dv._decode_km_actions(tok, g, True)
        pk, pm, pp = dv._decode_km_actions(tok, p, True)
        kt = gk if key_true is None else key_true[i:i + 100].to(dev)
        for j in range(len(g)):
            rows.append(dv._action_metrics(pk[j], gk[j], pm[j], gm[j], slice(None), is_km_fsq=True, pred_key_prob=pp[j], gt_key_true=kt[j], key_baserate_q=q))
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["r2"] = float(1 - ((pred - gt) ** 2).sum() / ((gt - gt.mean((0, 1))) ** 2).sum())
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video_t", choices=["0", "max"], default="0")
    p.add_argument("--out", required=True)
    p.add_argument("--debug_validation_dir", default=None)
    p.add_argument("--train_clips", type=int, default=4000)
    p.add_argument("--test_clips", type=int, default=1000)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev, bf16, t0 = th.device("cuda"), a.precision == "bf16", a.video_t == "0"
    model, diffusion, ns, step = load_model(a.checkpoint, dev)
    k, P = model.ticks_per_latent, model.tick_action_dim
    tok = dv._get_km_tokenizer(dev, ns.km_tokenizer_checkpoint, encoding="km_ctx14")
    names = ["embed"] + [f"L{i + 1}" for i in range(len(model.blocks))] + ["head"]
    L = len(model.blocks) + 1
    rows = names + ["head_clip", "persistence"]
    print(f"loaded {a.checkpoint} (step {step}) video_t={a.video_t} {a.precision}", flush=True)

    gen = th.Generator(dev).manual_seed(a.seed)
    D, n_fit, seen, hold, keysum, nk = model.action_head.linear.in_features, int(a.train_clips * (1 - a.holdout)), 0, [], 0.0, 0
    fit_acc = Ridge(L, D, P, dev)
    for fr, kp, pl in train_batches(ns, "train", a.train_clips, a.batch_size, a.workers, a.seed):
        f, _ = extract(model, diffusion, fr.to(dev), pl.to(dev), t0, gen, bf16)
        Y, n_in = ticks(kp)[:, k:].reshape(-1, P).to(dev), max(0, min(len(fr), n_fit - seen))  # n_in: clips of this batch on the fit side
        per = f.shape[2]
        if n_in:
            fit_acc.add(f[:, :n_in].reshape(L, -1, D), Y[:n_in * per])
        if n_in < len(fr):
            hold.append((f[:, n_in:].reshape(L, -1, D), Y[n_in * per:]))
        gk = dv._decode_km_actions(tok, ticks(kp)[:, k:].to(dev), True)[0]
        keysum, nk, seen = keysum + gk.sum((0, 1)), nk + gk.shape[0] * gk.shape[1], seen + len(fr)
        print(f"train clips {seen}", flush=True) if seen % 400 < len(fr) else None
    q = (keysum + 1) / (nk + 2)  # train-split key base rate, Laplace-smoothed so unseen keys have finite CE
    Xh, Yh = th.cat([h[0] for h in hold], 1), th.cat([h[1] for h in hold])
    errs = th.stack([((predict(fit_acc.fit(th.full((L,), g, device=dev)), Xh) - Yh.double()) ** 2).mean((1, 2)) for g in LAMS])  # (G, L)
    best = th.tensor(LAMS, device=dev)[errs.argmin(0)]
    fit_acc.add(Xh, Yh)
    fit = fit_acc.fit(best)
    print("lambda per layer", best.tolist(), flush=True)

    def collect(batches):  # per row name -> (N, (T-1)*k, 32) predictions, plus GT / persistence codes
        gen = th.Generator(dev).manual_seed(a.seed + 1)
        out = {n: [] for n in rows + ["gt"]}
        for fr, kp, pl in batches:
            f, h = extract(model, diffusion, fr.to(dev), pl.to(dev), t0, gen, bf16)
            pr = predict(fit, f.reshape(L, -1, f.shape[-1])).reshape(L, len(fr), -1, P).float()
            for n, v in zip(names[:-1], pr):
                out[n].append(v.cpu())
            out["head"].append(h.float().cpu()), out["head_clip"].append(h.float().clamp(-1, 1).cpu())  # codes live in [-1, 1]
            c = ticks(kp)
            out["gt"].append(c[:, k:]), out["persistence"].append(c[:, k - 1:-1])
        return {n: th.cat(v) for n, v in out.items()}

    def report(c, key_true, qq):
        res = {}
        for snap in (True, False):
            res["snap" if snap else "nosnap"] = {n: score(tok, c[n], c["gt"], key_true, qq, snap, dev) for n in rows}
        return res

    results = dict(args=vars(a), step=step, lam=best.tolist(), n_train_clips=seen)
    c = collect(train_batches(ns, "test", a.test_clips, a.batch_size, a.workers, a.seed + 1))
    results["test"] = report(c, None, q)
    vs = CorpusValidationSet(a.debug_validation_dir or str(Path(video_data_paths_dict[ns.dataset]) / "validation_rope47"), T=ns.T, n_observed=ns.T // 2,
                             action_encoding=ns.action_encoding, tokenizer_checkpoint=ns.km_tokenizer_checkpoint, device=dev, ticks_per_latent=k,
                             **plaicraft_valset_kwargs(ns.dataset, ns.num_classes, ns.global_database_path, ns.action_encoding))
    x0, codes, pl = vs.load_all(), vs.load_all_actions()[0], (vs.player_labels if vs.player_labels is not None else th.zeros(len(vs.rows), dtype=th.long))
    kt = vs.load_all_keys79()[:, k:].float()
    c = collect(((x0[i:i + 8], codes[i:i + 8], pl[i:i + 8]) for i in range(0, len(x0), 8)))
    results["val47"] = report(c, kt, q)
    results["val47"]["key_ce_baserate_pool"] = float(action_ce.keypress_ce_baserate(kt.reshape(-1, kt.shape[-1])))
    gp = dv._decode_km_actions(tok, c["gt"].to(dev), True)[2]
    results["val47"]["gt_decode_floor_key_ce"] = float(action_ce.keypress_cross_entropy(gp, kt.to(dev)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    lines = [f"# {Path(a.checkpoint).name} video_t={a.video_t} ({a.precision}, {seen} train clips)", ""]
    for split in ("test", "val47"):
        for var in ("snap", "nosnap"):
            r = results[split][var]
            lines += [f"## {split} ({var}); baserate CE {r['head']['key_ce_baserate']:.3f}, mouse trivial {r['head']['mouse_l1_trivial']:.3f}", "",
                      "| row | key CE | jaccard dist | mouse L1 | code R2 |", "|---|---|---|---|---|"]
            lines += [f"| {n} | {m['key_cross_entropy']:.3f} | {m['key_jaccard_distance']:.3f} | {m['mouse_l1']:.3f} | {m['r2']:.3f} |" for n, m in r.items()] + [""]
    lines.append(f"val47 pool baserate CE {results['val47']['key_ce_baserate_pool']:.3f}; GT-decode floor CE {results['val47']['gt_decode_floor_key_ce']:.3f}")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
