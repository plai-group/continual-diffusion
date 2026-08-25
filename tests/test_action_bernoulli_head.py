"""The action branch is a Bernoulli head: sigmoid + BCE on the key/click dims.

Unlike the video branch (eps-prediction), the action head predicts x0 directly.
Its raw output is logits on dims 0..7 and real values on the mouse dims, so
pred_actstart is a probability on the binary dims and the loss is BCE there.
"""
import math

import torch

from improved_diffusion.gaussian_diffusion import (
    N_BINARY_ACTION_DIMS, actions_from_logits,
)
from improved_diffusion.script_util import create_vdt_model_and_diffusion

B, T, C, H, W = 2, 8, 3, 32, 32
ACTION_DIM = 10
N_OBS = 4


def _build():
    return create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=ACTION_DIM,
        action_dropout_prob=0.0, generate_actions=True, action_loss_weight=1.0,
    )


def _binary_actions():
    a = torch.zeros(B, T, ACTION_DIM)
    a[:, ::2, 0] = 1.0
    a[:, 1::3, 4] = 1.0
    return a


def _masks():
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :N_OBS] = 1.0
    return obs_mask, 1.0 - obs_mask


def test_actions_from_logits_splits_binary_and_mouse():
    raw = torch.randn(B, T, ACTION_DIM) * 5.0
    out = actions_from_logits(raw)
    n = N_BINARY_ACTION_DIMS
    assert torch.allclose(out[..., :n], torch.sigmoid(raw[..., :n]))
    assert torch.allclose(out[..., n:], raw[..., n:])
    assert out[..., :n].min() > 0.0 and out[..., :n].max() < 1.0


def test_actions_from_logits_handles_all_binary_action_dim():
    raw = torch.randn(B, T, 4)
    assert torch.allclose(actions_from_logits(raw), torch.sigmoid(raw))


def test_pred_actstart_is_a_probability_under_eps_video_branch():
    model, diffusion = _build()
    from improved_diffusion.gaussian_diffusion import ModelMeanType
    # The video branch stays eps-prediction; the action branch must not follow it.
    assert diffusion.model_mean_type == ModelMeanType.EPSILON

    x = torch.randn(B, T, C, H, W)
    obs_mask, latent_mask = _masks()
    with torch.no_grad():
        for p in model.action_head.linear.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        out = diffusion.p_mean_variance(
            model, x, torch.tensor([50] * B),
            model_kwargs=dict(x0=x, obs_mask=obs_mask, latent_mask=latent_mask,
                              actions=_binary_actions()),
        )
    act = out["pred_actstart"]
    assert act is not None and act.shape == (B, T, ACTION_DIM)
    n = N_BINARY_ACTION_DIMS
    assert act[..., :n].min() >= 0.0 and act[..., :n].max() <= 1.0
    assert act[..., :n].std() > 0.0


def test_zero_init_action_head_gives_ln2_bce_on_binary_dims():
    model, diffusion = _build()
    x = torch.randn(B, T, C, H, W)
    obs_mask, latent_mask = _masks()
    actions = _binary_actions()
    actions[..., N_BINARY_ACTION_DIMS:] = 0.0
    terms = diffusion.training_losses(
        model, x, torch.tensor([25, 75]),
        model_kwargs=dict(x0=x, obs_mask=obs_mask, latent_mask=latent_mask,
                          actions=actions),
        latent_mask=latent_mask,
    )
    # The action mask is the frame mask shifted one row later, so row N_OBS is pinned, not supervised.
    n_latent_rows = T - N_OBS - 1
    expected = n_latent_rows * N_BINARY_ACTION_DIMS * math.log(2.0) / (T * ACTION_DIM)
    assert torch.allclose(terms["loss_action"], torch.full((B,), expected), atol=1e-5)
