"""quantize_keypress snaps a continuous (..., 8) keypress prediction to the
nearest of the 256 valid multi-hot vectors. Since every codebook entry is a
corner of the unit hypercube, nearest-neighbour in L2 reduces to independent
per-dim rounding -- so this is deliberately implemented as a threshold, not a
literal 256-vector table + search (plaicraft-debug#77).

_action_metrics only applies it when the diffusion object's
action_quantization attr is "codebook"; the default "none" preserves the
pre-existing inline >0.5 behavior.
"""
import torch

from improved_diffusion import debug_actions
from improved_diffusion.debug_validation import _action_metrics
from improved_diffusion.script_util import create_vdt_model_and_diffusion


def test_quantize_keypress_all_zeros():
    x = torch.zeros(4, 8)
    assert torch.equal(debug_actions.quantize_keypress(x), torch.zeros(4, 8))


def test_quantize_keypress_all_ones():
    x = torch.ones(4, 8)
    assert torch.equal(debug_actions.quantize_keypress(x), torch.ones(4, 8))


def test_quantize_keypress_mixed():
    x = torch.tensor([0.1, 0.49, 0.51, 0.9, -0.3, 1.2])
    expected = torch.tensor([0., 0., 1., 1., 0., 1.])
    assert torch.equal(debug_actions.quantize_keypress(x), expected)


def test_quantize_keypress_boundary_at_exactly_half():
    # torch's > is strict, so exactly 0.5 rounds down -- matches the pre-existing >0.5 inline check.
    x = torch.tensor([0.5])
    assert torch.equal(debug_actions.quantize_keypress(x), torch.tensor([0.]))


def _keys(seed=0):
    g = torch.Generator().manual_seed(seed)
    p_key = torch.rand(1, 4, 8, generator=g)
    g_key = (torch.rand(1, 4, 8, generator=g) > 0.5).float()
    return p_key, g_key


def test_action_metrics_dispatches_to_quantize_keypress_when_enabled():
    p_key, g_key = _keys()
    calls = []
    orig = debug_actions.quantize_keypress
    debug_actions.quantize_keypress = lambda x: calls.append(x) or orig(x)
    try:
        _action_metrics(p_key, g_key, None, None, slice(None), quantize=True)
    finally:
        debug_actions.quantize_keypress = orig
    assert len(calls) == 2, "quantize_keypress should run on both p_key and g_key"


def test_action_metrics_skips_quantize_keypress_when_disabled():
    p_key, g_key = _keys()
    calls = []
    orig = debug_actions.quantize_keypress
    debug_actions.quantize_keypress = lambda x: calls.append(x) or orig(x)
    try:
        _action_metrics(p_key, g_key, None, None, slice(None), quantize=False)
    finally:
        debug_actions.quantize_keypress = orig
    assert calls == [], "quantize=False must preserve the old inline >0.5 path"


def test_action_quantization_flag_defaults_to_none_and_round_trips():
    _, diffusion_default = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=3,
        num_frames=8, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=8, generate_actions=True,
    )
    assert diffusion_default.action_quantization == "none"

    _, diffusion_codebook = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(32, 32), in_channels=3,
        num_frames=8, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False, action_dim=8, generate_actions=True,
        action_quantization="codebook",
    )
    assert diffusion_codebook.action_quantization == "codebook"
