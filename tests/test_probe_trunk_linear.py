"""plaicraft-debug#106: the trunk probe sees 13 layers of action-token features, no GT action, and its ridge recovers a planted linear map."""
import torch

from improved_diffusion.script_util import create_vdt_model_and_diffusion
from scripts.probe_trunk_linear import Ridge, extract, predict

B, T, K, P, H = 2, 3, 2, 4, 16


def _model():
    torch.manual_seed(0)
    m, d = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(H, H), in_channels=3, num_frames=T, learn_sigma=False, sigma_small=False,
        diffusion_steps=100, diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False), noise_schedule="linear",
        timestep_respacing="", use_kl=False, predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=P, generate_actions=True, independent_action_t=True,
        pos_encoding="rope3d", attention="full", ticks_per_latent=K)
    for mod in (m.action_head, *(b.adaLN_modulation[-1] for b in m.blocks)):  # undo zero-init so the layers differ
        torch.nn.init.normal_(getattr(mod, "linear", mod).weight, std=0.02)
    return m.eval(), d


def _run(video_t0):
    m, d = _model()
    seen = []
    m.register_forward_pre_hook(lambda mod, a, kw: seen.append((a, dict(kw))), with_kwargs=True)
    x0 = torch.randn(B, T, 3, H, H)
    f, h = extract(m, d, x0, torch.zeros(B, dtype=torch.long), video_t0, torch.Generator().manual_seed(0))
    return d, x0, f, h, seen


def test_hooks_return_embed_plus_every_block_and_the_head_x0():
    _, _, f, h, _ = _run(True)
    assert f.shape == (13, B, (T - 1) * K, 384) and h.shape == (B, (T - 1) * K, P) and torch.isfinite(f).all() and torch.isfinite(h).all()
    assert not torch.allclose(f[0], f[12])


def test_no_gt_action_is_visible_and_actions_are_noise_at_t_max():
    d, x0, _, _, seen = _run(True)
    (x, *_), kw = seen[0]
    assert not {"actions0", "obs_action_mask", "mouse", "mouse0"} & kw.keys()
    t = d.num_timesteps - 1
    assert (kw["timesteps"] == 0).all() and torch.allclose(kw["action_timesteps"], torch.full((B,), d._scale_timesteps(torch.tensor(t)).item()))
    assert torch.equal(kw["obs_mask"], torch.ones(B, T, 1, 1, 1)) and torch.equal(x * 0 + kw["x0"], x0)
    assert abs(kw["actions"].std().item() - d.sqrt_one_minus_alphas_cumprod[t]) < 0.2  # randn * sqrt(1 - abar_max)


def test_video_t_max_gives_the_video_no_information():
    _, _, _, _, seen = _run(False)
    _, kw = seen[0]
    assert "x0" not in kw and "obs_mask" not in kw and torch.equal(kw["timesteps"], kw["action_timesteps"])


def test_ridge_recovers_a_planted_linear_map():
    g = torch.Generator().manual_seed(0)
    X, W = torch.randn(2, 3000, 8, generator=g) * torch.tensor([1.0, 50.0]).view(2, 1, 1) + 5, torch.randn(2, 8, 3, generator=g)
    Y = X[0] @ W[0] + 2.0  # layer 0's target; layer 1 sees a different X
    r = Ridge(2, 8, 3, "cpu")
    r.add(X, Y)
    pred = predict(r.fit(torch.full((2,), 1e-6)), X)
    assert torch.allclose(pred[0].float(), Y, atol=1e-3)  # standardised features, intercept included
    assert pred[1].shape == Y.shape
