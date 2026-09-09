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

Keypress (8-d) and mouse (2-d) are separate tensors since issue #71; every
helper here takes and returns a (keypress, mouse) pair.
"""
import torch

from improved_diffusion.action_masks import frame_mask_to_action_mask
from improved_diffusion.debug_validation import _action_metrics, _invert_actions, _swap_actions, _zero_actions


T, n_obs = 20, 10


def _actions(seed=0):
    g = torch.Generator().manual_seed(seed)
    keypress = (torch.rand(1, T, 8, generator=g) > 0.5).float()
    mouse = torch.randn(1, T, 2, generator=g)
    return keypress, mouse


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
    keypress, mouse = _actions()
    inv_k, inv_m = _invert_actions(keypress, mouse)
    for i, j in ((0, 2), (1, 3), (4, 5), (6, 7)):
        assert torch.equal(inv_k[..., i], keypress[..., j])
        assert torch.equal(inv_k[..., j], keypress[..., i])
    assert torch.equal(inv_m, -mouse)


def test_swap_starts_at_the_action_driving_the_first_generated_frame():
    orig_k, orig_m = _actions()
    swap_k, swap_m = _swap_actions(orig_k, orig_m, _intervene_on())
    # History untouched, so the SWAP panel's context bar matches the GT frames.
    assert torch.equal(swap_k[:, :n_obs], orig_k[:, :n_obs])
    assert torch.equal(swap_m[:, :n_obs], orig_m[:, :n_obs])
    # Row n_obs is the action that produces the first generated frame.
    inv_k, inv_m = _invert_actions(orig_k, orig_m)
    assert torch.equal(swap_k[:, n_obs:], inv_k[:, n_obs:])
    assert torch.equal(swap_m[:, n_obs:], inv_m[:, n_obs:])


def test_the_window_is_not_the_action_masks_complement():
    """`1 - action_mask` starts one row late -- the off-by-one this test exists for."""
    late = 1.0 - _obs_action_mask()
    orig_k, orig_m = _actions()
    late_k, _ = _swap_actions(orig_k, orig_m, late)
    assert torch.equal(late_k[:, n_obs], orig_k[:, n_obs]), \
        "sanity: the action-mask complement leaves row n_obs true"
    on_time_k, _ = _swap_actions(orig_k, orig_m, _intervene_on())
    assert not torch.equal(on_time_k[:, n_obs], orig_k[:, n_obs])


def test_history_is_identical_across_all_three_passes():
    """True/swap/zero differ only from n_obs on, or the comparison is not controlled."""
    orig_k, orig_m = _actions()
    where = _intervene_on()
    for k, m in (_swap_actions(orig_k, orig_m, where), _zero_actions(orig_k, orig_m, where)):
        assert torch.equal(k[:, :n_obs], orig_k[:, :n_obs])
        assert torch.equal(m[:, :n_obs], orig_m[:, :n_obs])


def test_zero_follows_the_same_mask():
    k, m = _actions()
    where = _intervene_on()
    zeroed_k, zeroed_m = _zero_actions(k, m, where)
    assert zeroed_k[:, n_obs:].eq(0).all()
    assert zeroed_m[:, n_obs:].eq(0).all()
    assert torch.equal(zeroed_k[:, :n_obs], k[:, :n_obs])
    assert torch.equal(zeroed_m[:, :n_obs], m[:, :n_obs])


def test_action_metrics_key_jaccard_distance_hand_computed():
    # gt holds {0,1,2}, pred holds {1,2,3} -> tp=2, fp=1, fn=1, distance=1-2/4=0.5.
    gt = torch.tensor([[1., 1., 1., 0., 0., 0., 0., 0.]])
    pred = torch.tensor([[0., 1., 1., 1., 0., 0., 0., 0.]])
    out = _action_metrics(pred, gt, None, None, slice(0, 1))
    assert out["key_jaccard_distance"] == 0.5


def test_action_metrics_key_jaccard_distance_no_keys_held_is_zero():
    z = torch.zeros(1, 8)
    out = _action_metrics(z, z, None, None, slice(0, 1))
    assert out["key_jaccard_distance"] == 0.0


def test_interventions_do_not_mutate_their_input():
    k, m = _actions()
    where = _intervene_on()
    before_k, before_m = k.clone(), m.clone()
    _swap_actions(k, m, where)
    _zero_actions(k, m, where)
    _invert_actions(k, m)
    assert torch.equal(k, before_k)
    assert torch.equal(m, before_m)
