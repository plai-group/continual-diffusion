"""plaicraft-debug#93 B3: latent (C=4) validation -- decode-before-metrics, and the swap
test skipped when a corpus manifest carries no swap package (e.g. plaicraft_morgan's).

Stubs mirror test_debug_validation_harness.py's; only diffusion_space/decode and the rows'
swap_kind differ.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import improved_diffusion.debug_validation as dv
import improved_diffusion.frechet_video_distance as fvdmod

T, N_OBS, H, W = 20, 10, 24, 40
KEY_RATES = torch.tensor([0.19, 0.03, 0.05, 0.03, 0.02, 0.03, 0.05, 0.045])


class _LatentValset:
    """4-channel frames, rows carrying swap_kind=None -- a corpus_valset manifest with no
    swap package (plaicraft_morgan's, plaicraft-debug#93)."""

    def __init__(self, n_rows=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.T, self.n_observed = T, N_OBS
        self._frames = torch.rand(n_rows, T, 4, H, W, generator=g) * 2 - 1
        self._keys = (torch.rand(n_rows, T, 8, generator=g) < KEY_RATES).float()
        self._mouse = torch.randn(n_rows, T, 2, generator=g)
        self.rows = [{"num": i, "prompt": f"ex{i}", "test_type": "exercise", "swap_kind": None}
                     for i in range(n_rows)]

    def load_all(self):
        return self._frames

    def load_all_actions(self):
        return self._keys, self._mouse

    def load_all_actions_raw(self):
        return self._keys, self._mouse

    def slug(self, row):
        return f"{row['num']:02d}_ex"


class _LatentDiffusion:
    num_timesteps = 100
    action_quantization = "none"
    action_encoding = "raw"
    diffusion_space = "latent"

    def __init__(self):
        self.decode_calls = 0

    def timestep2sigma(self, t):
        return 1000.0

    def heun_sample(self, model, shape, **kwargs):
        video = (torch.rand(*shape) * 2 - 1).double()
        act = torch.rand(shape[0], shape[1], 8).double()
        mouse = torch.randn(shape[0], shape[1], 2).double()
        return (video, (act, mouse)), None

    def decode(self, video, chunk_size=20):
        self.decode_calls += 1
        n, t, _c, h, w = video.shape
        return torch.zeros(n, t, 3, h, w, dtype=video.dtype)


class _StubModel(nn.Module):
    action_dim, mouse_dim = 8, 2
    generate_actions, generate_mouse = True, True


def _features(videos, batch_size=16):
    return np.random.randn(videos.shape[0], 64).astype(np.float32)


def _run(monkeypatch, tmp_path, diffusion, **kwargs):
    monkeypatch.setattr(dv, "_METRICS", lambda pred, gt: {
        "psnr": 1.0, "ssim": 1.0, "lpips": 1.0, "l2": 1.0, "rmse": 1.0, "l1": 1.0})
    monkeypatch.setattr(fvdmod, "_FEATURES", _features)
    torch.manual_seed(0)
    np.random.seed(0)
    return dv.run_debug_validation(
        _StubModel(), diffusion, _LatentValset(), "cpu",
        out_dir=tmp_path, log_videos=False, **kwargs,
    )


def test_c4_package_loads_and_decode_is_called(monkeypatch, tmp_path):
    diffusion = _LatentDiffusion()
    res = _run(monkeypatch, tmp_path, diffusion, swap_test=True)
    assert len(res["per_row"]) == 3
    assert diffusion.decode_calls > 0


def test_swap_skipped_when_rows_have_no_swap_kind(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, _LatentDiffusion(), swap_test=True)
    assert not any(k.startswith("val/swap/") for k in res["aggregate"])
    assert list(Path(tmp_path).iterdir()) == []  # no swap mp4s written


def test_swap_test_false_also_skips_it(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, _LatentDiffusion(), swap_test=False)
    assert not any(k.startswith("val/swap/") for k in res["aggregate"])
