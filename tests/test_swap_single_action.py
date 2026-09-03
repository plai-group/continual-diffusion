"""Issue #81: _swap_single_action -- the targeted, boundary-tick-only intervention
used against CorpusValidationSet's exercises. Unlike _swap_actions (whole-region
flip), only row `boundary_idx` may change; every other row, including the rest of
the generated region, must come back byte-identical to the input."""
import pytest
import torch

from improved_diffusion.debug_validation import _swap_single_action


def _sample():
    B, T = 2, 5
    keypress = torch.zeros(B, T, 8)
    keypress[:, :, 0] = 1.0  # w held throughout
    mouse = torch.ones(B, T, 2) * 3.0
    return keypress, mouse


def _other_rows(t, boundary):
    return [i for i in range(t) if i != boundary]


def test_keypress_swap_touches_only_boundary_row():
    keypress, mouse = _sample()
    boundary = 2
    out_k, out_m = _swap_single_action(keypress, mouse, boundary, "keypress",
                                        swap_dim=0, swap_counterpart_dim=2)
    assert torch.all(out_k[:, boundary, 0] == 0.0)
    assert torch.all(out_k[:, boundary, 2] == 1.0)
    other = _other_rows(keypress.shape[1], boundary)
    assert torch.equal(out_k[:, other], keypress[:, other])
    assert torch.equal(out_m, mouse)


def test_mouse_dx_negates_only_boundary():
    keypress, mouse = _sample()
    boundary = 1
    out_k, out_m = _swap_single_action(keypress, mouse, boundary, "mouse_dx")
    assert torch.equal(out_k, keypress)
    assert torch.all(out_m[:, boundary, 0] == -3.0)
    assert torch.all(out_m[:, boundary, 1] == 3.0)
    other = _other_rows(mouse.shape[1], boundary)
    assert torch.equal(out_m[:, other], mouse[:, other])


def test_mouse_dy_negates_only_boundary():
    keypress, mouse = _sample()
    boundary = 3
    out_k, out_m = _swap_single_action(keypress, mouse, boundary, "mouse_dy")
    assert torch.equal(out_k, keypress)
    assert torch.all(out_m[:, boundary, 1] == -3.0)
    assert torch.all(out_m[:, boundary, 0] == 3.0)
    other = _other_rows(mouse.shape[1], boundary)
    assert torch.equal(out_m[:, other], mouse[:, other])


def test_unknown_swap_kind_raises():
    keypress, mouse = _sample()
    with pytest.raises(ValueError):
        _swap_single_action(keypress, mouse, 0, "bogus")


def test_inputs_not_mutated():
    keypress, mouse = _sample()
    k0, m0 = keypress.clone(), mouse.clone()
    _swap_single_action(keypress, mouse, 1, "mouse_dx")
    assert torch.equal(keypress, k0) and torch.equal(mouse, m0)
