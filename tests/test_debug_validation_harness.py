"""Execute run_debug_validation end to end against stubs.

Every other test in this suite exercises a helper in isolation, so the function that
actually assembles the wandb payload has never run under test -- a metric can be
computed correctly and still never reach wandb, and nothing would catch it.

Everything heavy is stubbed: the diffusion sampler returns noise, _METRICS returns
fixed floats (so LPIPS never loads), and the S3D extractor returns random features at
a small feature dim (the real 1024 only costs sqrtm time; the dim is not what is under
test). Runs on CPU in seconds with no checkpoints and no network.

Driven with swap_test=False, which also disables the only mp4 write (log_videos is read
only inside the swap block), so out_dir stays empty.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import improved_diffusion.debug_validation as dv
import improved_diffusion.frechet_video_distance as fvdmod

T, N_OBS, H, W = 20, 10, 24, 40
FEATURE_DIM = 64
# Roughly the corpus policy's per-key rates, so the base rate is not degenerate.
KEY_RATES = torch.tensor([0.19, 0.03, 0.05, 0.03, 0.02, 0.03, 0.05, 0.045])


class _StubValset:
    def __init__(self, n_rows=6, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.T, self.n_observed = T, N_OBS
        self._frames = torch.rand(n_rows, T, 3, H, W, generator=g) * 2 - 1
        self._keys = (torch.rand(n_rows, T, 8, generator=g) < KEY_RATES).float()
        self._mouse = torch.randn(n_rows, T, 2, generator=g)
        # No swap_kind: keeps is_corpus_valset False, matching the DebugValidationSet path.
        self.rows = [{"num": i, "prompt": f"exercise {i}", "test_type": "stub"}
                     for i in range(n_rows)]

    def load_all(self):
        return self._frames

    def load_all_actions(self):
        return self._keys, self._mouse

    def load_all_actions_raw(self):
        return self._keys, self._mouse

    def slug(self, row):
        return f"{row['num']:02d}_stub"


class _StubDiffusion:
    num_timesteps = 100
    action_quantization = "none"
    action_encoding = "raw"

    def timestep2sigma(self, t):
        return 1000.0

    def heun_sample(self, model, shape, **kwargs):
        # Global RNG, so repeat draws for the fake FVD pool genuinely differ.
        video = torch.rand(*shape) * 2 - 1
        act = torch.rand(shape[0], shape[1], 8)
        mouse = torch.randn(shape[0], shape[1], 2)
        return (video, (act, mouse)), None


class _StubModel(nn.Module):
    action_dim, mouse_dim = 8, 2
    generate_actions, generate_mouse = True, True


def _features(videos, batch_size=16):
    return np.random.randn(videos.shape[0], FEATURE_DIM).astype(np.float32)


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(dv, "_METRICS", lambda pred, gt: {
        "psnr": 1.0, "ssim": 1.0, "lpips": 1.0, "l2": 1.0, "rmse": 1.0, "l1": 1.0})
    monkeypatch.setattr(fvdmod, "_FEATURES", _features)
    torch.manual_seed(0)
    np.random.seed(0)

    def run(tmp_path, valset=None, **kwargs):
        return dv.run_debug_validation(
            _StubModel(), _StubDiffusion(), valset or _StubValset(), "cpu",
            out_dir=tmp_path, log_videos=False, swap_test=False, **kwargs)

    return run


def test_validation_returns_aggregate_and_per_row(harness, tmp_path):
    res = harness(tmp_path)
    assert set(res) == {"aggregate", "per_row"}
    assert len(res["per_row"]) == 6


def test_frame_and_action_metrics_reach_the_aggregate(harness, tmp_path):
    agg = harness(tmp_path)["aggregate"]
    for k in ("val/video/psnr", "val/video_roll/psnr",
              "val/action/key_jaccard_distance", "val/action/key_cross_entropy",
              "val/action_roll/key_cross_entropy"):
        assert k in agg, k


def test_fvd_keys_reach_the_aggregate(harness, tmp_path):
    agg = harness(tmp_path)["aggregate"]
    for k in ("val/video/fvd", "val/video/kvd", "val/video/kvd_subset_spread"):
        assert k in agg and np.isfinite(agg[k]), k


def test_fvd_repeats_zero_disables_the_video_distances(harness, tmp_path):
    agg = harness(tmp_path, fvd_repeats=0)["aggregate"]
    assert "val/video/fvd" not in agg
    assert "val/video/psnr" in agg  # the frame metrics share the prefix and must survive


def test_key_ce_baserate_is_nonzero_at_both_scopes(harness, tmp_path):
    # The next scope slices a single frame. Deriving q from that slice made this
    # exactly 0.0 at val/action/key_ce_baserate, so the anchor could never be beaten.
    agg = harness(tmp_path)["aggregate"]
    assert agg["val/action/key_ce_baserate"] > 0.0
    assert agg["val/action_roll/key_ce_baserate"] > 0.0
