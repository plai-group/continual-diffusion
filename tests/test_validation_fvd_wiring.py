"""FVD is pool-level, so it bypasses ACT_METRIC_KEYS and sets agg keys directly.

Nothing else in run_debug_validation survives the chunk loop -- x0 and samples are
rebound every iteration -- so the features must be taken in-loop and reduced after.
These pin the two things that fail silently: a renamed wandb key (the panel just
never appears) and a shape mismatch between what the accumulator collects and what
video_distances expects.
"""
import inspect

import numpy as np
import torch

import improved_diffusion.frechet_video_distance as fvdmod
from improved_diffusion.debug_validation import run_debug_validation

T, N_OBS, H, W = 20, 10, 24, 40


def test_fvd_repeats_defaults_to_four():
    # Below 4 the fake side stops being larger than the real side and kvd_std goes
    # degenerate (measured 1.9e-07 at 13-vs-13 against 2.4e-02 at 13-vs-52).
    assert inspect.signature(run_debug_validation).parameters["fvd_repeats"].default == 4


def test_wandb_key_names_are_stable():
    src = inspect.getsource(run_debug_validation)
    for k in ("val/video/fvd", "val/video/kvd", "val/video/kvd_subset_spread"):
        assert k in src


def test_accumulated_chunk_features_feed_video_distances():
    # The real geometry: per-chunk (b, 10, 3, 24, 40) generated halves in [-1,1],
    # collected across chunks, concatenated, then reduced once.
    extract = fvdmod._get_video_features("cpu")
    g = torch.Generator().manual_seed(0)
    real, fake = [], []
    for _ in range(2):
        real.append(extract(torch.rand(2, T - N_OBS, 3, H, W, generator=g) * 2 - 1))
        for _ in range(2):
            fake.append(extract(torch.rand(2, T - N_OBS, 3, H, W, generator=g) * 2 - 1))
    r, f = np.concatenate(real), np.concatenate(fake)
    assert r.shape == (4, 1024) and f.shape == (8, 1024)
    out = fvdmod.video_distances(r, f)
    assert all(np.isfinite(out[k]) for k in ("fvd", "kvd", "kvd_std"))
