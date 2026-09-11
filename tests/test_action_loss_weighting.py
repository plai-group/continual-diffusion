"""action_dim_ratio/mouse_dim_ratio restore parity between the video, keypress,
and mouse loss terms.

mean_flat divides by the FULL element count, so loss_video is a per-pixel
mean (T*C*H*W) while loss_action/loss_mouse are per-action-number means
(T*action_dim / T*mouse_dim). Adding these with equal weight over-weights the
action/mouse heads by roughly the ratio of pixel count to action count.
training_losses corrects this by scaling each term by its own numel()/x.numel()
before applying keypress_loss_weight/mouse_loss_weight, so each weight reads as
a multiple of parity rather than an absolute scale, independently of the other.
"""
import math

import torch

from improved_diffusion.script_util import create_vdt_model_and_diffusion

B, T, C, H, W = 2, 8, 3, 32, 32


def _build(action_dim, mouse_dim=0, keypress_loss_weight=1.0, mouse_loss_weight=1.0):
    return create_vdt_model_and_diffusion(
        model_name="VDT-S",
        patch_size=4,
        input_size=(32, 32),
        in_channels=C,
        num_frames=T,
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_edm_scaling=False,
        action_dim=action_dim,
        mouse_dim=mouse_dim,
        action_dropout_prob=0.0,
        generate_actions=True,
        generate_mouse=mouse_dim > 0,
        keypress_loss_weight=keypress_loss_weight,
        mouse_loss_weight=mouse_loss_weight,
    )


def _losses(model, diffusion, action_dim, mouse_dim=0, keypress_loss_weight=1.0, mouse_loss_weight=1.0):
    x = torch.randn(B, T, C, H, W)
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :4] = 1.0
    latent_mask = 1.0 - obs_mask
    t = torch.tensor([25, 75])
    model_kwargs = {
        "x0": x, "obs_mask": obs_mask, "latent_mask": latent_mask,
        "actions": torch.randn(B, T, action_dim),
        "keypress_loss_weight": keypress_loss_weight,
    }
    if mouse_dim > 0:
        model_kwargs["mouse"] = torch.randn(B, T, mouse_dim)
        model_kwargs["mouse_loss_weight"] = mouse_loss_weight
    return diffusion.training_losses(model, x, t, model_kwargs=model_kwargs, latent_mask=latent_mask)


def test_dim_ratio_matches_element_count_fraction():
    # action_dim_ratio is a pure shape computation, so a single call pins it down.
    model, diffusion = _build(action_dim=10)
    terms = _losses(model, diffusion, action_dim=10)

    expected_ratio = (T * 10) / (T * C * H * W)  # 80 / 24576
    assert terms["action_dim_ratio"].shape == (B,)
    for v in terms["action_dim_ratio"].tolist():
        # Stored as float32, so compare with a tolerance against the float64 fraction.
        assert math.isclose(v, expected_ratio, rel_tol=1e-6)


def test_doubling_action_dim_doubles_ratio():
    # With the video shape fixed the ratio is linear in action_dim, which is what makes it self-correcting.
    model10, diffusion10 = _build(action_dim=10)
    model20, diffusion20 = _build(action_dim=20)

    ratio10 = _losses(model10, diffusion10, action_dim=10)["action_dim_ratio"][0].item()
    ratio20 = _losses(model20, diffusion20, action_dim=20)["action_dim_ratio"][0].item()

    assert math.isclose(ratio20, 2 * ratio10, rel_tol=1e-6)


def test_action_loss_weight_multiplies_dim_ratio_on_top():
    # keypress_loss_weight multiplies dim_ratio rather than replacing it, and w=0 must zero the action term exactly.
    model, diffusion = _build(action_dim=10)
    for w in (0.0, 1.0, 4.0):
        terms = _losses(model, diffusion, action_dim=10, keypress_loss_weight=w)
        ratio = terms["action_dim_ratio"]
        expected_total = terms["loss_video"] + w * ratio * terms["loss_action"]
        assert torch.allclose(terms["loss"], expected_total, atol=1e-5)
        if w == 0.0:
            assert torch.equal(terms["loss"], terms["loss_video"])


def test_video_loss_unaffected_by_action_weight():
    # Regression guard: loss_video/mse are computed before the dim_ratio branch and must not move when keypress_loss_weight changes.
    model, diffusion = _build(action_dim=10)

    mse_by_weight = {}
    loss_video_by_weight = {}
    for w in (0.0, 1.0, 4.0):
        # Reseed before each call so q_sample's noise draws are identical across weights.
        torch.manual_seed(0)
        terms = _losses(model, diffusion, action_dim=10, keypress_loss_weight=w)
        mse_by_weight[w] = terms["mse"]
        loss_video_by_weight[w] = terms["loss_video"]

    base_mse = mse_by_weight[0.0]
    base_loss_video = loss_video_by_weight[0.0]
    for w in (1.0, 4.0):
        assert torch.allclose(mse_by_weight[w], base_mse, atol=1e-6)
        assert torch.allclose(loss_video_by_weight[w], base_loss_video, atol=1e-6)


def test_mouse_dim_ratio_matches_element_count_fraction():
    model, diffusion = _build(action_dim=8, mouse_dim=2)
    terms = _losses(model, diffusion, action_dim=8, mouse_dim=2)

    expected_ratio = (T * 2) / (T * C * H * W)
    assert terms["mouse_dim_ratio"].shape == (B,)
    for v in terms["mouse_dim_ratio"].tolist():
        assert math.isclose(v, expected_ratio, rel_tol=1e-6)


def test_mouse_loss_weight_is_independent_of_keypress_weight():
    # Changing mouse_loss_weight must not move loss_action, and vice versa -- the two terms are independent by design.
    model, diffusion = _build(action_dim=8, mouse_dim=2)
    torch.manual_seed(0)
    terms_low = _losses(model, diffusion, action_dim=8, mouse_dim=2, keypress_loss_weight=1.0, mouse_loss_weight=0.0)
    torch.manual_seed(0)
    terms_high = _losses(model, diffusion, action_dim=8, mouse_dim=2, keypress_loss_weight=1.0, mouse_loss_weight=5.0)

    assert torch.allclose(terms_low["loss_action"], terms_high["loss_action"], atol=1e-6)
    assert torch.equal(terms_low["loss"], terms_low["loss_video"] + terms_low["action_dim_ratio"] * terms_low["loss_action"])
    ratio_m = terms_high["mouse_dim_ratio"]
    expected_high = terms_high["loss_video"] + terms_high["action_dim_ratio"] * terms_high["loss_action"] + 5.0 * ratio_m * terms_high["loss_mouse"]
    assert torch.allclose(terms_high["loss"], expected_high, atol=1e-5)
