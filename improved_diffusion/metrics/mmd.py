import numpy as np
from sklearn import metrics
from functools import partial


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)


def _mmd2_and_variance(
    K_XX,
    K_XY,
    K_YY,
    unit_diagonal=False,
    mmd_est="unbiased",
    var_at_m=None,
    ret_var=True,
):
    # Taken from
    # https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == "biased":
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2 * K_XY_sum / (m * m)
        )
    else:
        assert mmd_est in {"unbiased", "u-statistic"}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == "unbiased":
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1
        / (m * m1 * m2)
        * (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (_sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (
        4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
        + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est
    )

    return mmd2, var_est


def mmd2_from_kernel(
    X, Y, kernel, unit_diagonal=False, mmd_est="unbiased", ret_var=False
):
    """
    Calculate the Maximum Mean Discrepancy (MMD) using a given kernel function.

    Args:
        X (array-like): The first input samples. Shape (N1 x d).
        Y (array-like): The second input samples. Shape: (N2 x d).
        kernel (callable): The kernel function.

    Keyword Args:
        unit_diagonal (bool): Whether the kernel matrix has ones on the diagonal (default: False).
        mmd_est (str): The MMD estimator to use, 'unbiased' or 'u-statistic' (default: 'unbiased').
        ret_var (bool): Whether to return the variance estimate (default: True).

    Returns:
        float: The MMD value.
    """
    if not isinstance(unit_diagonal, bool):
        raise ValueError("unit_diagonal must be a boolean")
    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)

    return _mmd2_and_variance(
        K_XX, K_XY, K_YY, unit_diagonal=unit_diagonal, mmd_est=mmd_est, ret_var=ret_var
    )


def mmd2_rbf(X, Y, gamma=1.0, **kwargs):
    metrics.pairwise.rbf_kernel(X, X, gamma)
    kernel = partial(metrics.pairwise.rbf_kernel, gamma=gamma)
    return mmd2_from_kernel(X, Y, kernel, **kwargs)


def mmd2_poly(X, Y, degree=3, gamma=None, coef0=1, **kwargs):
    kernel = partial(
        metrics.pairwise.polynomial_kernel, degree=degree, gamma=gamma, coef0=coef0
    )
    return mmd2_from_kernel(X, Y, kernel, **kwargs)


def mmd2_energy(X, Y, **kwargs):
    def kernel(X, Y):
        return -metrics.pairwise_distances(X, Y, metric='euclidean')

    return mmd2_from_kernel(X, Y, kernel, **kwargs)