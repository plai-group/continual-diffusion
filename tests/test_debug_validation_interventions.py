"""Swap/zero interventions select their rows from a mask.

The intervention lands on the actions that DRIVE THE GENERATED FRAMES --
rows n_obs..T-1 -- and the history stays true. Two things depend on that
window being exactly right, and both were broken when it was the action
mask's observed half instead:

  * The context frames are pinned to ground truth. Flip the history and the
    overlay paints an action bar that contradicts the video beneath it: the
    bar reads `d` while the frames strafe left.
  * Row n_obs is the action producing the FIRST generated frame, and the
    action mask marks it observed. Taking `1 - action_mask` therefore starts
    the swap one row late and leaves that first frame driven by the true
    action.
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


def _obs_frame_mask():
    obs = torch.zeros(1, T, 1, 1, 1)
    obs[:, :n_obs] = 1.0
    return obs


def _obs_action_mask():
    return frame_mask_to_action_mask(_obs_frame_mask())


def _intervene_on():
    """The window debug_validation actually uses: complement of the FRAME mask."""
    m = _obs_frame_mask()
    return 1.0 - m.reshape(m.shape[0], m.shape[1], 1)


def test_inversion_opposes_every_axis():
    a = _actions()
    inv = _invert_actions(a)
    for i, j in ((0, 2), (1, 3), (4, 5), (6, 7)):
        assert torch.equal(inv[..., i], a[..., j])
        assert torch.equal(inv[..., j], a[..., i])
    assert torch.equal(inv[..., 8], -a[..., 8])
    assert torch.equal(inv[..., 9], -a[..., 9])


def test_swap_starts_at_the_action_driving_the_first_generated_frame():
    a = _swap_actions(_actions(), _intervene_on())
    orig = _actions()
    # History untouched, so the SWAP panel's context bar matches the GT frames.
    assert torch.equal(a[:, :n_obs], orig[:, :n_obs])
    # Row n_obs is the action that produces the first generated frame.
    assert torch.equal(a[:, n_obs:], _invert_actions(orig)[:, n_obs:])


def test_the_window_is_not_the_action_masks_complement():
    """`1 - action_mask` starts one row late -- the off-by-one this test exists for."""
    late = 1.0 - _obs_action_mask()
    orig = _actions()
    assert torch.equal(_swap_actions(orig, late)[:, n_obs], orig[:, n_obs]), \
        "sanity: the action-mask complement leaves row n_obs true"
    assert not torch.equal(_swap_actions(orig, _intervene_on())[:, n_obs], orig[:, n_obs])


def test_history_is_identical_across_all_three_passes():
    """True/swap/zero differ only from n_obs on, or the comparison is not controlled."""
    orig = _actions()
    where = _intervene_on()
    for variant in (_swap_actions(orig, where), _zero_actions(orig, where)):
        assert torch.equal(variant[:, :n_obs], orig[:, :n_obs])


def test_zero_follows_the_same_mask():
    a, where = _actions(), _intervene_on()
    zeroed = _zero_actions(a, where)
    assert zeroed[:, n_obs:].eq(0).all()
    assert torch.equal(zeroed[:, :n_obs], a[:, :n_obs])


def test_interventions_do_not_mutate_their_input():
    a, where = _actions(), _intervene_on()
    before = a.clone()
    _swap_actions(a, where)
    _zero_actions(a, where)
    _invert_actions(a)
    assert torch.equal(a, before)
