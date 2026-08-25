"""The action mask must lag the frame mask by one row.

The action cache is causal (row i produced frame i) while the environment is
a_t -> s_{t+1}, so the action taken at frame i lives at cache row i+1.
Observing n frames means observing the n actions taken during them, which is
cache rows 1..n -- one row further than the frame mask reaches.
"""
import torch

from improved_diffusion.action_masks import frame_mask_to_action_mask


def test_prefix_mask_gains_the_boundary_row():
    B, T, n_obs = 2, 20, 10
    obs = torch.zeros(B, T, 1, 1, 1)
    obs[:, :n_obs] = 1.0

    act = frame_mask_to_action_mask(obs)

    assert act.shape == (B, T, 1)
    # The action taken at the last observed frame produces the first generated frame, so it is observed too.
    assert act[:, n_obs].eq(1).all(), 'boundary action must be ground truth'
    assert act[:, : n_obs + 1].eq(1).all()
    assert act[:, n_obs + 1 :].eq(0).all()


def test_row_zero_follows_frame_zero():
    # Cache row 0 has no predecessor frame; it inherits frame 0's status.
    obs = torch.zeros(1, 5, 1, 1, 1)
    obs[:, :2] = 1.0
    assert frame_mask_to_action_mask(obs)[0, 0].item() == 1.0

    obs = torch.zeros(1, 5, 1, 1, 1)
    obs[:, 3:] = 1.0
    assert frame_mask_to_action_mask(obs)[0, 0].item() == 0.0


def test_interleaved_mask_shifts_elementwise():
    obs = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]).view(1, 5, 1)
    act = frame_mask_to_action_mask(obs)
    assert act.view(-1).tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]


def test_accepts_2d_and_3d_masks():
    flat = torch.ones(3, 7)
    assert frame_mask_to_action_mask(flat).shape == (3, 7, 1)
    already = torch.ones(3, 7, 1)
    assert frame_mask_to_action_mask(already).shape == (3, 7, 1)


def test_does_not_mutate_input():
    obs = torch.zeros(1, 4, 1)
    obs[:, :2] = 1.0
    before = obs.clone()
    frame_mask_to_action_mask(obs)
    assert torch.equal(obs, before)
