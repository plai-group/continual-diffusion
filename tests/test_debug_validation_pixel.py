"""plaicraft-debug#100: diffusion_space=pixel validation on a 96x160 (3-channel) package -- no VAE decode,
metrics computed, km_ctx14 overlays rendered from 96x160 frames at 10fps."""
import numpy as np
import torch
import wandb

import improved_diffusion.debug_validation as dv
import improved_diffusion.frechet_video_distance as fvdmod
from improved_diffusion import debug_actions
from improved_diffusion.script_util import create_gaussian_diffusion

from test_debug_validation_latent_decode import _KmDiffusion, _KmModel, _KmValset, _features

H, W = 96, 160


class _PixelValset(_KmValset):
    def __init__(self):
        super().__init__()
        self._frames = torch.rand(3, 20, 3, H, W) * 2 - 1
        self.rows = [{**r, "swap_kind": None} for r in self.rows]

    def load_all_actions(self):
        return torch.zeros(3, 20, debug_actions.KM_CTX14_CODE_DIM), torch.zeros(3, 20, 0)


class _PixelDiffusion(_KmDiffusion):
    action_encoding, diffusion_space = "km_ctx14", "pixel"

    def heun_sample(self, model, shape, **kwargs):
        self.clip_seen = kwargs.get("clip_denoised")
        (video, (_, mouse)), none = super().heun_sample(model, shape, **kwargs)
        return (video, (torch.zeros(shape[0], shape[1], debug_actions.KM_CTX14_CODE_DIM).double(), mouse)), none

    def decode(self, *a, **k):
        raise AssertionError("pixel space must not decode")


def test_pixel_validation_metrics_and_96x160_overlays(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(dv, "_get_km_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(dv, "_decode_km_actions", lambda t, codes, all_keys=False: (
        torch.zeros(*codes.shape[:2], 79), torch.zeros(*codes.shape[:2], 2), torch.zeros(*codes.shape[:2], 79)))
    shapes = []
    monkeypatch.setattr(dv, "_render_panels", lambda frames, acts, labels, n, path, ncols, fps=None: shapes.append((frames[0].shape, fps)))
    monkeypatch.setattr(wandb, "Video", lambda p: p)
    monkeypatch.setattr(dv, "_METRICS", lambda pred, gt: seen.append(pred.shape) or dict.fromkeys(("psnr", "ssim", "lpips", "l2", "rmse", "l1"), 1.0))
    monkeypatch.setattr(fvdmod, "_FEATURES", _features)
    diffusion = _PixelDiffusion()
    res = dv.run_debug_validation(_KmModel(), diffusion, _PixelValset(), "cpu", out_dir=tmp_path, swap_test=False)
    assert seen and all(s[-3:] == (3, H, W) for s in seen) and "val/video/psnr" in res["aggregate"]
    assert diffusion.clip_seen is True and len(res["per_row"]) == 3
    assert shapes and all(s == ((20, 3, H, W), debug_actions.KM_CTX14_FPS) for s in shapes)


def test_pixel_x0_clamps_and_patch8_forward_and_loss_on_96x160():
    from improved_diffusion.vdt import VDT_S_2
    d = create_gaussian_diffusion(steps=1000, diffusion_space_kwargs={"diffusion_space": "pixel", "pre_encoded": False})
    x = torch.full((1, 3, 2, 2), 3.0)
    assert d.p_mean_variance(lambda x, **k: (torch.zeros_like(x), None), x, torch.tensor([0]), clip_denoised=True)["pred_xstart"].max() == 1.0
    m = VDT_S_2(input_size=(H, W), patch_size=8, in_channels=3, num_frames=4, learn_sigma=False)
    assert m.x_embedder.num_patches == 12 * 20
    x0 = torch.rand(2, 4, 3, H, W) * 2 - 1
    out = m(x0, torch.tensor([5, 500]))[0]
    assert out.shape == x0.shape
    loss = d.training_losses(m, x0, torch.tensor([5, 500]))["loss"]
    assert torch.isfinite(loss).all()
