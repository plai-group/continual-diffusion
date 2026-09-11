"""plaicraft-debug#86: FVD/KVD must stay torch-native.

The reference implementation was Google's TF1 I3D, which the container can no
longer run (no tensorflow/tensorflow_hub, and the I3D graph lived on the now-
retired tfhub.dev). These tests guard the torchvision-S3D replacement: that the
module still imports without TF, that its Frechet/kernel math is sane on tiny
hand-built feature matrices, and that the small in-training validation pool
(N=13) doesn't trip kid_features_to_metric's kid_subset_size>=1000 assert.
"""
import numpy as np
import torch

import improved_diffusion.frechet_video_distance as fvd


def test_import_does_not_require_tensorflow():
    import improved_diffusion.frechet_video_distance  # noqa: F401


def test_video_distances_identical_features_are_near_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(30, 8, generator=g).numpy()
    out = fvd.video_distances(x, x)
    assert abs(out["fvd"]) < 1e-6
    # kvd is deliberately NOT ~0 here. subset_size == N makes kid draw the whole pool
    # for both sides, so K_XY carries the self-similarities Kt_XX/Kt_YY exclude -- a
    # negative ~1/N artifact of scoring a set against itself, which cannot arise in
    # production because real and fake are always different point sets.
    assert out["kvd"] < 0


def test_kvd_is_unbiased_on_disjoint_samples():
    # The production geometry: two different point sets from one distribution -> ~0.
    g = torch.Generator().manual_seed(4)
    pool = torch.randn(512, 64, generator=g).numpy()
    out = fvd.video_distances(pool[:256], pool[256:])
    assert abs(out["kvd"]) < 0.05


def test_video_distances_handles_small_asymmetric_pool():
    # Production shapes: 13 real validation clips vs n_rows * fvd_repeats generated
    # ones, at S3D's real 1024-d. kid_subset_size defaults to 1000 and hard-asserts
    # both inputs are at least that large, so the min() clamp is load-bearing here.
    g = torch.Generator().manual_seed(1)
    x = torch.randn(13, 1024, generator=g).numpy()
    y = torch.randn(52, 1024, generator=g).numpy()
    out = fvd.video_distances(x, y)
    assert all(np.isfinite(out[k]) for k in ("fvd", "kvd", "kvd_std"))


def test_fvd_grows_with_distributional_shift():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(30, 8, generator=g).numpy()
    y_near = x + torch.randn(30, 8, generator=torch.Generator().manual_seed(3)).numpy() * 0.01
    y_far = x + 10.0
    d_near = fvd.video_distances(x, y_near)["fvd"]
    d_far = fvd.video_distances(x, y_far)["fvd"]
    assert d_far > d_near


def test_get_video_features_is_cached_singleton(monkeypatch):
    # Stub the constructor so this never downloads or builds real S3D weights.
    monkeypatch.setattr(fvd, "_VideoFeatures", lambda device: object())
    fvd._FEATURES = None
    f1 = fvd._get_video_features("cpu")
    f2 = fvd._get_video_features("cpu")
    assert f1 is f2
