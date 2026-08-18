"""
Comprehensive Test Suite for Issue #69: Action Generation in VDT
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

from improved_diffusion.vdt import VDT, VDT_S_2, VDT_SM_2, VDT_M_2, VDT_L_2
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    vdt_model_and_diffusion_defaults,
    create_vdt_model_and_diffusion,
    create_gaussian_diffusion,
)
from improved_diffusion.resample import UniformSampler


def test_vdt_shapes_and_grads():
    print("=" * 60)
    print("1. Testing VDT Architecture & Forward/Backward")
    print("=" * 60)
    B, T, C, H, W = 2, 8, 3, 32, 32
    action_dim = 10

    # Test 1a: Unconditional / video-only VDT
    vdt_uncond = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=0,
    )
    x = torch.randn(B, T, C, H, W)
    t = torch.tensor([50, 150])
    v_out, a_out = vdt_uncond(x, timesteps=t)
    assert v_out.shape == (B, T, C, H, W), f"Expected {(B, T, C, H, W)}, got {v_out.shape}"
    assert a_out is None, f"Expected None for unconditional action out, got {a_out}"
    print("  [PASS] Unconditional VDT returns (video, None)")

    # Test 1b: Action conditioning mode (generate_actions=False)
    vdt_cond = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=action_dim, generate_actions=False,
    )
    actions = torch.randn(B, T, action_dim)
    v_out_c, a_out_c = vdt_cond(x, timesteps=t, actions=actions)
    assert v_out_c.shape == (B, T, C, H, W)
    assert a_out_c is None
    assert vdt_cond.action_embedder is not None
    assert vdt_cond.action_x_embedder is None
    print("  [PASS] Action-conditioned VDT (legacy) returns (video, None)")

    # Test 1c: Action generation mode (generate_actions=True)
    vdt_gen = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=action_dim, generate_actions=True,
    )
    assert vdt_gen.action_x_embedder is not None
    assert vdt_gen.action_pos_embed is not None
    assert vdt_gen.action_pos_embed.shape == (1, 1, 384)
    assert vdt_gen.action_final_layer is not None

    v_out_g, a_out_g = vdt_gen(x, timesteps=t, actions=actions)
    assert v_out_g.shape == (B, T, C, H, W), f"Expected {(B, T, C, H, W)}, got {v_out_g.shape}"
    assert a_out_g.shape == (B, T, action_dim), f"Expected {(B, T, action_dim)}, got {a_out_g.shape}"
    print("  [PASS] Action-generation VDT returns ((B,T,C,H,W), (B,T,action_dim))")

    # Test 1d: With obs_action_mask and actions0
    actions0 = torch.randn(B, T, action_dim)
    obs_action_mask = torch.zeros(B, T, 1)
    obs_action_mask[:, :4] = 1.0
    v_out_m, a_out_m = vdt_gen(
        x, timesteps=t, actions=actions, actions0=actions0, obs_action_mask=obs_action_mask
    )
    assert v_out_m.shape == (B, T, C, H, W)
    assert a_out_m.shape == (B, T, action_dim)
    print("  [PASS] Action-generation VDT works with obs_action_mask & actions0")

    # Test 1e: Gradient flow
    loss = v_out_g.sum() + a_out_g.sum()
    loss.backward()
    assert vdt_gen.action_pos_embed.grad is not None, "action_pos_embed received no gradient"
    assert vdt_gen.action_x_embedder.weight.grad is not None, "action_x_embedder received no gradient"
    assert vdt_gen.action_final_layer.linear.weight.grad is not None, "action_final_layer received no gradient"
    assert vdt_gen.final_layer.linear.weight.grad is not None, "final_layer received no gradient"
    print("  [PASS] All action and video sub-modules received gradients during backward pass")

    # Test 1f: Verify other model scales (SM, M, L)
    for name, ctor, hidden in [("VDT-SM", VDT_SM_2, 640), ("VDT-M", VDT_M_2, 1024), ("VDT-L", VDT_L_2, 1152)]:
        m = ctor(input_size=32, patch_size=4, in_channels=C, num_frames=T, learn_sigma=False,
                 action_dim=action_dim, generate_actions=True)
        assert m.action_pos_embed.shape == (1, 1, hidden)
        vo, ao = m(x, timesteps=t, actions=actions)
        assert vo.shape == (B, T, C, H, W)
        assert ao.shape == (B, T, action_dim)
        print(f"  [PASS] {name} created and forward pass verified (hidden_size={hidden})")


def test_gaussian_diffusion_training_losses():
    print("\n" + "=" * 60)
    print("2. Testing GaussianDiffusion & SpacedDiffusion Training Losses")
    print("=" * 60)
    B, T, C, H, W = 2, 8, 3, 32, 32
    action_dim = 10

    model, diffusion = create_vdt_model_and_diffusion(
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
        action_dropout_prob=0.0,
        generate_actions=True,
        action_loss_weight=1.5,
    )

    x = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T, action_dim)
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :4] = 1.0
    latent_mask = 1.0 - obs_mask
    t = torch.tensor([25, 75])

    model_kwargs = {
        "x0": x,
        "obs_mask": obs_mask,
        "latent_mask": latent_mask,
        "actions": actions,
        "action_loss_weight": 1.5,
    }

    terms = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs, latent_mask=latent_mask)

    required_keys = ["loss", "loss_video", "loss_action", "loss_vid", "loss_act", "loss_total", "mse", "eval-mse"]
    for k in required_keys:
        assert k in terms, f"Missing key '{k}' in diffusion terms: {list(terms.keys())}"

    # Check loss equation: loss_total == loss_video + 1.5 * loss_action
    expected_total = terms["loss_video"] + 1.5 * terms["loss_action"]
    assert torch.allclose(terms["loss_total"], expected_total, atol=1e-5), "Loss total formula mismatch"
    assert torch.allclose(terms["loss"], terms["loss_total"], atol=1e-5), "Loss mismatch"
    print(f"  [PASS] training_losses computed all terms:")
    print(f"         loss_video={terms['loss_video'].mean().item():.4f}, "
          f"loss_action={terms['loss_action'].mean().item():.4f}, "
          f"loss_total={terms['loss_total'].mean().item():.4f}")

    # Test backward pass from training_losses
    loss = terms["loss"].mean()
    loss.backward()
    assert model.action_x_embedder.weight.grad is not None
    assert model.action_final_layer.linear.weight.grad is not None
    assert model.final_layer.linear.weight.grad is not None
    print("  [PASS] Backward pass through total loss completed successfully")


def test_heun_sample():
    print("\n" + "=" * 60)
    print("3. Testing Joint Heun Sampling (Video + Action)")
    print("=" * 60)
    B, T, C, H, W = 2, 8, 3, 32, 32
    action_dim = 10

    model, diffusion = create_vdt_model_and_diffusion(
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
        action_dropout_prob=0.0,
        generate_actions=True,
    )

    x0 = torch.randn(B, T, C, H, W)
    actions0 = torch.randn(B, T, action_dim)
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :4] = 1.0
    latent_mask = 1.0 - obs_mask

    model_kwargs = {
        "x0": x0,
        "actions0": actions0,
        "obs_mask": obs_mask,
        "latent_mask": latent_mask,
    }

    samples, history = diffusion.heun_sample(
        model,
        (B, T, C, H, W),
        model_kwargs=model_kwargs,
        num_steps=5,
        return_decoded=False,
    )

    assert isinstance(samples, tuple), f"Expected tuple from heun_sample with action generation, got {type(samples)}"
    samples_video, samples_action = samples

    assert samples_video.shape == (B, T, C, H, W), f"Expected video shape {(B, T, C, H, W)}, got {samples_video.shape}"
    assert samples_action.shape == (B, T, action_dim), f"Expected action shape {(B, T, action_dim)}, got {samples_action.shape}"
    print("  [PASS] Joint Heun sample returned video shape", samples_video.shape, "and action shape", samples_action.shape)

    # Test joint sampling with custom initial noise tuple
    noise_v = torch.randn(B, T, C, H, W)
    noise_a = torch.randn(B, T, action_dim)
    samples_custom, _ = diffusion.heun_sample(
        model,
        (B, T, C, H, W),
        noise=(noise_v, noise_a),
        model_kwargs=model_kwargs,
        num_steps=5,
        return_decoded=False,
    )
    sv_c, sa_c = samples_custom
    assert sv_c.shape == (B, T, C, H, W)
    assert sa_c.shape == (B, T, action_dim)
    print("  [PASS] Joint Heun sample with tuple noise (noise_v, noise_a) succeeded")


def test_cli_and_defaults():
    print("\n" + "=" * 60)
    print("4. Testing CLI Parsing and Config Defaults")
    print("=" * 60)
    try:
        import mpi4py
    except ImportError:
        from unittest.mock import MagicMock
        mock_mpi = MagicMock()
        sys.modules['mpi4py'] = mock_mpi
        sys.modules['mpi4py.MPI'] = mock_mpi.MPI

    from scripts.video_train_vdt import create_argparser

    parser = create_argparser()
    args = parser.parse_args([
        "--generate_actions", "True",
        "--action_loss_weight", "2.5",
        "--action_dim", "10",
    ])
    assert args.generate_actions is True
    assert args.action_loss_weight == 2.5
    assert args.action_dim == 10
    print("  [PASS] Argument parser correctly parsed --generate_actions and --action_loss_weight")

    args2 = parser.parse_args([
        "--action_generation", "True",
        "--action_dim", "10",
    ])
    assert args2.action_generation is True
    print("  [PASS] Argument parser alias --action_generation parsed successfully")


if __name__ == "__main__":
    print("Running Issue #69 Test Suite...")
    test_vdt_shapes_and_grads()
    test_gaussian_diffusion_training_losses()
    test_heun_sample()
    test_cli_and_defaults()
    print("\n" + "=" * 60)
    print("ALL ISSUE #69 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
