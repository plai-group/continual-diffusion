"""Swap/zero interventions select their rows from a mask.

An action-generating model is probed by rewriting its action HISTORY and
letting it generate the future, so the intervention lands on the observed
rows and the generated half stays free. A conditioned-only model generates
no actions, so the counterfactual is imposed on the latent rows instead.
"""
import torch

from improved_diffusion.action_masks import frame_mask_to_action_mask
from improved_diffusion.debug_validation import _invert_actions, _swap_actions, _zero_actions


T, n_obs = 20, 10


def _actions(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(1, T, 10)
    a[..., :8] = (torch.rand(1, T, 8, generator=g) > 0.5).float()
    a[..., 8:] = torch.randn(1, T, 2, generator=g)
    return a


def _obs_action_mask():
    obs = torch.zeros(1, T, 1, 1, 1)
    obs[:, :n_obs] = 1.0
    return frame_mask_to_action_mask(obs)


def test_inversion_opposes_every_axis():
    a = _actions()
    inv = _invert_actions(a)
    for i, j in ((0, 2), (1, 3), (4, 5), (6, 7)):
        assert torch.equal(inv[..., i], a[..., j])
        assert torch.equal(inv[..., j], a[..., i])
    assert torch.equal(inv[..., 8], -a[..., 8])
    assert torch.equal(inv[..., 9], -a[..., 9])


def test_generation_mode_rewrites_history_and_leaves_the_future_free():
    a, where = _actions(), _obs_action_mask()
    swapped = _swap_actions(a, where)

    # History (rows 0..n_obs, inclusive of the boundary action) is inverted.
    assert torch.equal(swapped[:, : n_obs + 1], _invert_actions(a)[:, : n_obs + 1])
    # The future is untouched -- the model generates it, so its value here is
    # only the noise seed, never a pinned counterfactual.
    assert torch.equal(swapped[:, n_obs + 1 :], a[:, n_obs + 1 :])


def test_conditioned_mode_intervenes_on_the_future():
    a = _actions()
    where = 1.0 - _obs_action_mask()
    swapped = _swap_actions(a, where)
    assert torch.equal(swapped[:, : n_obs + 1], a[:, : n_obs + 1])
    assert torch.equal(swapped[:, n_obs + 1 :], _invert_actions(a)[:, n_obs + 1 :])


def test_zero_follows_the_same_mask():
    a, where = _actions(), _obs_action_mask()
    zeroed = _zero_actions(a, where)
    assert zeroed[:, : n_obs + 1].eq(0).all()
    assert torch.equal(zeroed[:, n_obs + 1 :], a[:, n_obs + 1 :])


def test_interventions_do_not_mutate_their_input():
    a, where = _actions(), _obs_action_mask()
    before = a.clone()
    _swap_actions(a, where)
    _zero_actions(a, where)
    _invert_actions(a)
    assert torch.equal(a, before)
