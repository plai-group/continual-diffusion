"""Action generation in VDT: the keypress and mouse tokens' shapes and gradients,
their terms in the diffusion loss, joint video+action(s) sampling, and the CLI
flags that switch them on."""

import sys
import torch

from improved_diffusion.vdt import VDT_S_2, VDT_SM_2, VDT_M_2, VDT_L_2
from improved_diffusion.script_util import create_vdt_model_and_diffusion


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
    v_out, (a_out, m_out) = vdt_uncond(x, timesteps=t)
    assert v_out.shape == (B, T, C, H, W), f"Expected {(B, T, C, H, W)}, got {v_out.shape}"
    assert a_out is None and m_out is None, f"Expected (None, None) for unconditional out, got {(a_out, m_out)}"
    print("  [PASS] Unconditional VDT returns (video, (None, None))")

    # Test 1b: Action conditioning mode (generate_actions=False)
    vdt_cond = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=action_dim, generate_actions=False,
    )
    actions = torch.randn(B, T, action_dim)
    v_out_c, (a_out_c, m_out_c) = vdt_cond(x, timesteps=t, actions=actions)
    assert v_out_c.shape == (B, T, C, H, W)
    assert a_out_c is None and m_out_c is None
    assert vdt_cond.action_embedder is not None
    assert vdt_cond.action_x_embedder is None
    print("  [PASS] Action-conditioned VDT (legacy) returns (video, (None, None))")

    # Test 1c: Action generation mode (generate_actions=True)
    vdt_gen = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=action_dim, generate_actions=True,
    )
    assert vdt_gen.action_x_embedder is not None
    assert vdt_gen.action_pos_embed is not None
    assert vdt_gen.action_pos_embed.shape == (1, 1, 384)
    assert vdt_gen.action_head is not None

    v_out_g, (a_out_g, m_out_g) = vdt_gen(x, timesteps=t, actions=actions)
    assert v_out_g.shape == (B, T, C, H, W), f"Expected {(B, T, C, H, W)}, got {v_out_g.shape}"
    assert a_out_g.shape == (B, T, action_dim), f"Expected {(B, T, action_dim)}, got {a_out_g.shape}"
    assert m_out_g is None, "mouse_dim=0 must keep the mouse slot None"
    print("  [PASS] Action-generation VDT returns ((B,T,C,H,W), ((B,T,action_dim), None))")

    # Test 1d: With obs_action_mask and actions0
    actions0 = torch.randn(B, T, action_dim)
    obs_action_mask = torch.zeros(B, T, 1)
    obs_action_mask[:, :4] = 1.0
    v_out_m, (a_out_m, _) = vdt_gen(
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
    assert vdt_gen.action_head.linear.weight.grad is not None, "action_head received no gradient"
    assert vdt_gen.final_layer.linear.weight.grad is not None, "final_layer received no gradient"
    print("  [PASS] All action and video sub-modules received gradients during backward pass")

    # Test 1f: Verify other model scales (SM, M, L)
    for name, ctor, hidden in [("VDT-SM", VDT_SM_2, 640), ("VDT-M", VDT_M_2, 1024), ("VDT-L", VDT_L_2, 1152)]:
        m = ctor(input_size=32, patch_size=4, in_channels=C, num_frames=T, learn_sigma=False,
                 action_dim=action_dim, generate_actions=True)
        assert m.action_pos_embed.shape == (1, 1, hidden)
        vo, (ao, mo) = m(x, timesteps=t, actions=actions)
        assert vo.shape == (B, T, C, H, W)
        assert ao.shape == (B, T, action_dim)
        assert mo is None
        print(f"  [PASS] {name} created and forward pass verified (hidden_size={hidden})")

    # Test 1g: Mouse token alongside keypress -- issue #71's split. Order patch -> keypress -> mouse.
    # action_dim=80 mirrors production (issue #74's encoded-keypress latent), not the 8-d raw vector.
    mouse_dim = 2
    keypress_dim = 80
    vdt_both = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=keypress_dim, mouse_dim=mouse_dim,
        generate_actions=True, generate_mouse=True,
    )
    assert vdt_both.mouse_x_embedder is not None
    assert vdt_both.mouse_pos_embed is not None
    assert vdt_both.mouse_head is not None
    keypress = torch.randn(B, T, keypress_dim)
    mouse = torch.randn(B, T, mouse_dim)
    v_out_km, act_out_km = vdt_both(x, timesteps=t, actions=keypress, mouse=mouse)
    assert v_out_km.shape == (B, T, C, H, W)
    assert isinstance(act_out_km, tuple), "both tokens active -> (act_out, mouse_out) tuple"
    a_out_km, m_out_km = act_out_km
    assert a_out_km.shape == (B, T, keypress_dim)
    assert m_out_km.shape == (B, T, mouse_dim)
    print("  [PASS] Keypress+mouse VDT returns (video, (keypress, mouse))")

    loss_km = v_out_km.sum() + a_out_km.sum() + m_out_km.sum()
    loss_km.backward()
    assert vdt_both.mouse_pos_embed.grad is not None
    assert vdt_both.mouse_x_embedder.weight.grad is not None
    assert vdt_both.mouse_head.linear.weight.grad is not None
    print("  [PASS] Mouse sub-modules received gradients during backward pass")

    # Test 1h: Mouse-only (action_dim=0, generate_mouse=True) -- keypress slot stays None.
    vdt_mouse_only = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T,
        learn_sigma=False, action_dim=0, mouse_dim=mouse_dim, generate_mouse=True,
    )
    v_out_m, (a_out_only, m_out_only) = vdt_mouse_only(x, timesteps=t, mouse=mouse)
    assert v_out_m.shape == (B, T, C, H, W)
    assert a_out_only is None, "no keypress token was built -- act_out must be None"
    assert m_out_only.shape == (B, T, mouse_dim)
    print("  [PASS] Mouse-only VDT returns (video, mouse) -- no keypress token built")

    # Test 1i: keypress and mouse modes are fully independent, not mirrored (issue #71 review fix).
    vdt_keypress_cond_mouse_gen = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T, learn_sigma=False,
        action_dim=keypress_dim, mouse_dim=mouse_dim, action_token_cond=True, generate_mouse=True,
    )
    assert vdt_keypress_cond_mouse_gen.mouse_head is not None
    assert vdt_keypress_cond_mouse_gen.action_head is None
    print("  [PASS] action_token_cond=True + generate_mouse=True -> mouse_head only")

    vdt_keypress_gen_mouse_cond = VDT_S_2(
        input_size=32, patch_size=4, in_channels=C, num_frames=T, learn_sigma=False,
        action_dim=keypress_dim, mouse_dim=mouse_dim, generate_actions=True, mouse_token_cond=True,
    )
    assert vdt_keypress_gen_mouse_cond.action_head is not None
    assert vdt_keypress_gen_mouse_cond.mouse_head is None
    print("  [PASS] generate_actions=True + mouse_token_cond=True -> action_head only")


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
        keypress_loss_weight=1.5,
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
        "keypress_loss_weight": 1.5,
    }

    terms = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs, latent_mask=latent_mask)

    required_keys = ["loss", "loss_video", "loss_action", "mse", "eval-mse",
                     "action_dim_ratio"]
    # loss_vid / loss_act were aliases of loss_video / loss_action; one name each.
    for k in ("loss_vid", "loss_act"):
        assert k not in terms, f"Stale alias '{k}' is back in diffusion terms"
    for k in required_keys:
        assert k in terms, f"Missing key '{k}' in diffusion terms: {list(terms.keys())}"

    # loss == loss_video + weight * (D_a/D_v) * loss_action; the ratio keeps the two mean_flat terms comparable per element.
    ratio = actions[0].numel() / x[0].numel()
    assert torch.allclose(terms["action_dim_ratio"], torch.full_like(terms["loss_video"], ratio))
    expected_total = terms["loss_video"] + 1.5 * ratio * terms["loss_action"]
    assert torch.allclose(terms["loss"], expected_total, atol=1e-5), "Loss total formula mismatch"
    print(f"  [PASS] training_losses computed all terms:")
    print(f"         loss_video={terms['loss_video'].mean().item():.4f}, "
          f"loss_action={terms['loss_action'].mean().item():.4f}, "
          f"loss={terms['loss'].mean().item():.4f}")

    # Test backward pass from training_losses
    loss = terms["loss"].mean()
    loss.backward()
    assert model.action_x_embedder.weight.grad is not None
    assert model.action_head.linear.weight.grad is not None
    assert model.final_layer.linear.weight.grad is not None
    print("  [PASS] Backward pass through total loss completed successfully")


def test_mouse_training_losses():
    print("\n" + "=" * 60)
    print("2b. Testing mouse token's independent loss term")
    print("=" * 60)
    B, T, C, H, W = 2, 8, 3, 32, 32
    keypress_dim, mouse_dim = 80, 2

    model, diffusion = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False,
        action_dim=keypress_dim, mouse_dim=mouse_dim, generate_actions=True, generate_mouse=True,
        keypress_loss_weight=1.0, mouse_loss_weight=2.0,
    )

    x = torch.randn(B, T, C, H, W)
    actions = torch.randn(B, T, keypress_dim)
    mouse = torch.randn(B, T, mouse_dim)
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :4] = 1.0
    latent_mask = 1.0 - obs_mask
    t = torch.tensor([25, 75])

    model_kwargs = {
        "x0": x, "obs_mask": obs_mask, "latent_mask": latent_mask,
        "actions": actions, "mouse": mouse,
        "keypress_loss_weight": 1.0, "mouse_loss_weight": 2.0,
    }
    terms = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs, latent_mask=latent_mask)

    for k in ("loss_action", "loss_mouse", "action_dim_ratio", "mouse_dim_ratio", "loss_video"):
        assert k in terms, f"Missing key '{k}' in diffusion terms: {list(terms.keys())}"

    ratio_a = actions[0].numel() / x[0].numel()
    ratio_m = mouse[0].numel() / x[0].numel()
    expected_total = terms["loss_video"] + 1.0 * ratio_a * terms["loss_action"] + 2.0 * ratio_m * terms["loss_mouse"]
    assert torch.allclose(terms["loss"], expected_total, atol=1e-5), "keypress+mouse loss formula mismatch"
    print("  [PASS] loss combines keypress and mouse terms with their own weight * dim_ratio")

    loss = terms["loss"].mean()
    loss.backward()
    assert model.mouse_x_embedder.weight.grad is not None
    assert model.mouse_head.linear.weight.grad is not None
    assert model.action_x_embedder.weight.grad is not None
    print("  [PASS] Backward pass reaches both keypress and mouse sub-modules")


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
    samples_video, second = samples
    assert isinstance(second, tuple), "action-only still returns a fixed (action, mouse) pair"
    samples_action, samples_mouse = second

    assert samples_video.shape == (B, T, C, H, W), f"Expected video shape {(B, T, C, H, W)}, got {samples_video.shape}"
    assert samples_action.shape == (B, T, action_dim), f"Expected action shape {(B, T, action_dim)}, got {samples_action.shape}"
    assert samples_mouse is None, "no mouse token was generated -- mouse slot must be None"
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
    sv_c, (sa_c, sm_c) = samples_custom
    assert sv_c.shape == (B, T, C, H, W)
    assert sa_c.shape == (B, T, action_dim)
    assert sm_c is None
    print("  [PASS] Joint Heun sample with tuple noise (noise_v, noise_a) succeeded")

    # Test neither modality active: return type stays a bare tensor, not a pair.
    model_novideo, diffusion_novideo = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=0,
    )
    samples_none, _ = diffusion_novideo.heun_sample(
        model_novideo, (B, T, C, H, W), num_steps=5, return_decoded=False,
    )
    assert not isinstance(samples_none, tuple), "neither modality active -- must return bare video tensor"
    assert samples_none.shape == (B, T, C, H, W)
    print("  [PASS] Neither modality active -> bare video tensor returned")


def test_heun_sample_with_mouse():
    print("\n" + "=" * 60)
    print("3b. Testing Joint Heun Sampling (Video + Keypress + Mouse)")
    print("=" * 60)
    B, T, C, H, W = 2, 8, 3, 32, 32
    keypress_dim, mouse_dim = 80, 2

    model, diffusion = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False,
        action_dim=keypress_dim, mouse_dim=mouse_dim, generate_actions=True, generate_mouse=True,
    )

    x0 = torch.randn(B, T, C, H, W)
    actions0 = torch.randn(B, T, keypress_dim)
    mouse0 = torch.randn(B, T, mouse_dim)
    obs_mask = torch.zeros(B, T, 1, 1, 1)
    obs_mask[:, :4] = 1.0
    latent_mask = 1.0 - obs_mask

    model_kwargs = {
        "x0": x0, "actions0": actions0, "mouse0": mouse0,
        "obs_mask": obs_mask, "latent_mask": latent_mask,
    }

    samples, _ = diffusion.heun_sample(
        model, (B, T, C, H, W), model_kwargs=model_kwargs, num_steps=5, return_decoded=False,
    )
    assert isinstance(samples, tuple)
    samples_video, second = samples
    assert samples_video.shape == (B, T, C, H, W)
    assert isinstance(second, tuple), "keypress + mouse both active -> (action, mouse) tuple"
    samples_action, samples_mouse = second
    assert samples_action.shape == (B, T, keypress_dim)
    assert samples_mouse.shape == (B, T, mouse_dim)
    print("  [PASS] Joint Heun sample returns (video, (keypress, mouse)) when both are generated")

    # 3-way noise tuple: (noise_v, noise_a, noise_m).
    noise_v = torch.randn(B, T, C, H, W)
    noise_a = torch.randn(B, T, keypress_dim)
    noise_m = torch.randn(B, T, mouse_dim)
    samples_custom, _ = diffusion.heun_sample(
        model, (B, T, C, H, W), noise=(noise_v, noise_a, noise_m),
        model_kwargs=model_kwargs, num_steps=5, return_decoded=False,
    )
    sv_c, (sa_c, sm_c) = samples_custom
    assert sv_c.shape == (B, T, C, H, W)
    assert sa_c.shape == (B, T, keypress_dim)
    assert sm_c.shape == (B, T, mouse_dim)
    print("  [PASS] Joint Heun sample with (noise_v, noise_a, noise_m) tuple succeeded")

    # Mouse-only: sample_actions False, sample_mouse True -- still returns a fixed pair.
    model_mo, diffusion_mo = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False,
        action_dim=0, mouse_dim=mouse_dim, generate_mouse=True,
    )
    mouse_kwargs = {
        "x0": x0, "mouse0": mouse0, "obs_mask": obs_mask, "latent_mask": latent_mask,
    }
    samples_mo, _ = diffusion_mo.heun_sample(
        model_mo, (B, T, C, H, W), model_kwargs=mouse_kwargs, num_steps=5, return_decoded=False,
    )
    assert isinstance(samples_mo, tuple)
    sv_mo, second_mo = samples_mo
    assert isinstance(second_mo, tuple), "mouse-only still returns a fixed (action, mouse) pair"
    sa_mo, sm_mo = second_mo
    assert sv_mo.shape == (B, T, C, H, W)
    assert sa_mo is None, "no keypress token was generated -- action slot must be None"
    assert sm_mo.shape == (B, T, mouse_dim)
    print("  [PASS] Mouse-only Heun sample returns (video, (None, mouse))")


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
        "--keypress_loss_weight", "2.5",
        "--mouse_loss_weight", "0.5",
        "--action_dim", "80",
        "--mouse_dim", "2",
    ])
    assert args.generate_actions is True
    assert args.keypress_loss_weight == 2.5
    assert args.mouse_loss_weight == 0.5
    assert args.action_dim == 80
    assert args.mouse_dim == 2
    print("  [PASS] Argument parser correctly parsed --generate_actions/--keypress_loss_weight/--mouse_loss_weight")

    # generate_actions is the only flag; the old --action_generation alias was removed.
    args2 = parser.parse_args(["--action_dim", "10"])
    assert args2.generate_actions is False
    assert not hasattr(args2, "action_generation")
    print("  [PASS] generate_actions defaults to False and has no alias")

    # generate_mouse/mouse_token_cond are their own plain bool flags, independent of generate_actions.
    args3 = parser.parse_args(["--generate_actions", "True", "--generate_mouse", "False"])
    assert args3.generate_actions is True
    assert args3.generate_mouse is False
    args4 = parser.parse_args([])
    assert args4.generate_mouse is False
    assert args4.mouse_token_cond is False
    print("  [PASS] generate_mouse/mouse_token_cond default to False and parse independently of generate_actions")


if __name__ == "__main__":
    print("Running VDT action-generation tests...")
    test_vdt_shapes_and_grads()
    test_gaussian_diffusion_training_losses()
    test_mouse_training_losses()
    test_heun_sample()
    test_heun_sample_with_mouse()
    test_cli_and_defaults()
    print("\n" + "=" * 60)
    print("ALL VDT ACTION-GENERATION TESTS PASSED")
    print("=" * 60)
