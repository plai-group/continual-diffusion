# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.

The feature extractor here is torchvision S3D (Kinetics-400), I3D's
separable-3D descendant trained on the same Kinetics-400 set: the original
needs TF1 and pulls its I3D graph from the now-retired tfhub.dev.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import scipy
import torch
import torch.nn as nn


######################################################################
## Feature extraction                                               ##
######################################################################

class _VideoFeatures:
    """Kinetics-400 S3D penultimate features: I3D's separable-3D descendant, same training
    set, torch-native (the tfhub I3D needs TF1 and a retired host)."""

    # Resize straight to 224x224 rather than S3D_Weights' packaged 256-then-crop-224
    # transform, which would throw away the edges of a small generated frame.
    RESIZE = 224
    MEAN = [0.43216, 0.394666, 0.37645]
    STD = [0.22803, 0.22145, 0.216989]
    # S3D_Weights.KINETICS400_V1.meta["min_temporal_size"] == 14; our generated half is 10.
    NUM_FRAMES = 16

    def __init__(self, device):
        from torchvision.models.video import S3D_Weights, s3d

        self.device = device
        m = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.net = nn.Sequential(m.features, nn.AdaptiveAvgPool3d(1)).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, videos, batch_size=16):
        """videos: (N, T, 3, H, W) float in [-1,1] -> (N, 1024) float32 numpy."""
        n, t = videos.shape[:2]
        idx = torch.linspace(0, t - 1, self.NUM_FRAMES).round().long()  # nearest-neighbour temporal resize
        videos = videos[:, idx]
        out = []
        for i in range(0, n, batch_size):
            v = videos[i : i + batch_size].to(self.device)
            v = (v + 1.0) / 2.0
            b, tt, c, h, w = v.shape
            v = v.reshape(b * tt, c, h, w)
            v = nn.functional.interpolate(v, size=(self.RESIZE, self.RESIZE), mode="bilinear", align_corners=False)
            v = v.reshape(b, tt, c, self.RESIZE, self.RESIZE).permute(0, 2, 1, 3, 4)  # (N,T,3,H,W) -> (N,3,T,H,W)
            v = (v - self.mean) / self.std
            out.append(self.net(v).flatten(1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


_FEATURES = None


def _get_video_features(device):
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = _VideoFeatures(device)
    return _FEATURES


######################################################################
## Frechet distance computation                                     ##
######################################################################

# Adopted from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_fid.py
def frechet_statistics_from_features(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        'mu': mu,
        'sigma': sigma,
    }

def frechet_statistics_to_frechet_metric(stat_1, stat_2):
    eps = 1e-6

    mu1, sigma1 = stat_1['mu'], stat_1['sigma']
    mu2, sigma2 = stat_2['mu'], stat_2['sigma']
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all() or (
        np.iscomplexobj(covmean) and not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
    ):
        print(
            f'WARNING: fid calculation produces singular product; '
            f'adding {eps} to diagonal of cov estimates'
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=True)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, 'Imaginary component {}'.format(m)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return out

def fid_features_to_metric(features_1, features_2):
    assert isinstance(features_1, np.ndarray) and features_1.ndim == 2
    assert isinstance(features_2, np.ndarray) and features_2.ndim == 2
    assert features_1.shape[1] == features_2.shape[1]

    stat_1 = frechet_statistics_from_features(features_1)
    stat_2 = frechet_statistics_from_features(features_2)
    return frechet_statistics_to_frechet_metric(stat_1, stat_2)


######################################################################
## Kernel distance computation                                      ##
######################################################################

# Adopted from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_kid.py
KEY_METRIC_KID_MEAN = 'kernel_inception_distance_mean'
KEY_METRIC_KID_STD = 'kernel_inception_distance_std'

def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    assert mmd_est in ('biased', 'unbiased', 'u-statistic'), 'Invalid value of mmd_est'

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    k_11 = polynomial_kernel(features_1, features_1, degree=degree, gamma=gamma, coef0=coef0)
    k_22 = polynomial_kernel(features_2, features_2, degree=degree, gamma=gamma, coef0=coef0)
    k_12 = polynomial_kernel(features_1, features_2, degree=degree, gamma=gamma, coef0=coef0)
    return mmd2(k_11, k_12, k_22)


def kid_features_to_metric(features_1, features_2,
                           kid_subsets=100, kid_subset_size=1000,
                           kid_degree=3, kid_gamma=None, kid_coef0=1,
                           rng_seed=2020, verbose=False):
    # Default arguments from https://github.com/toshas/torch-fidelity/blob/a5cba01a8edc2b0f5303570e13c0f48eb6e96819/torch_fidelity/defaults.py
    assert isinstance(features_1, np.ndarray) and features_1.ndim == 2
    assert isinstance(features_2, np.ndarray) and features_2.ndim == 2
    assert features_1.shape[1] == features_2.shape[1]

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    assert \
        n_samples_1 >= kid_subset_size and n_samples_2 >= kid_subset_size,\
        f'KID subset size {kid_subset_size} cannot be smaller than the number of samples (input_1: {n_samples_1}, '\
        f'input_2: {n_samples_2}). Consider using "kid_subset_size" kwarg or "--kid-subset-size" command line key to '\
        f'proceed.'

    mmds = np.zeros(kid_subsets)
    rng = np.random.RandomState(rng_seed)

    for i in range(kid_subsets):
        f1 = features_1[rng.choice(n_samples_1, kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(n_samples_2, kid_subset_size, replace=False)]
        o = polynomial_mmd(
            f1,
            f2,
            kid_degree,
            kid_gamma,
            kid_coef0,
        )
        mmds[i] = o

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

    return out


######################################################################
## Entry point                                                      ##
######################################################################

def video_distances(feats_real, feats_fake):
    """(N,D) and (M,D) feature arrays -> {"fvd", "kvd", "kvd_std"}."""
    fvd = fid_features_to_metric(feats_real, feats_fake)
    kid_subset_size = min(len(feats_real), len(feats_fake), 1000)
    kid = kid_features_to_metric(feats_real, feats_fake, kid_subset_size=kid_subset_size)
    return {"fvd": fvd, "kvd": kid[KEY_METRIC_KID_MEAN], "kvd_std": kid[KEY_METRIC_KID_STD]}
