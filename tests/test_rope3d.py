"""pos_encoding=rope3d in VDT: plaicraft-debug#99."""

import importlib.util
import os

import pytest
import torch

from improved_diffusion.script_util import backfill_pos_encoding
from improved_diffusion.vdt import RopeAttention, VDT_S_2, apply_rope, rope_tables, rope_view

KW = dict(input_size=16, patch_size=4, in_channels=3, num_frames=4, learn_sigma=False, action_dim=5, generate_actions=True)


def test_rope_relative_shift_invariance():
    T, grid, d = 6, (3, 4), 64
    cos, sin = rope_tables(T, grid, [0], d)
    N = 12
    q, k = torch.randn(d), torch.randn(d)
    rot = lambda v, l, n: apply_rope(v.view(1, 1, 1, d), cos[l, n].view(1, 1, d // 2), sin[l, n].view(1, 1, d // 2)).flatten()
    # shift both tokens one frame later: video-video and action-action scores must not change (action-video mixes layouts by design)
    for n_a, n_b in [(0, 5), (7, 7), (N, N)]:
        s0 = rot(q, 1, n_a) @ rot(k, 3, n_b)
        s1 = rot(q, 2, n_a) @ rot(k, 4, n_b)
        assert torch.allclose(s0, s1, atol=1e-4), (n_a, n_b)
    # shifting one grid row/col together also leaves the score unchanged
    assert torch.allclose(rot(q, 0, 0) @ rot(k, 0, 5), rot(q, 0, 4) @ rot(k, 0, 9), atol=1e-4)


def test_rope_split_and_action_tick():
    cos, sin = rope_tables(3, (2, 2), [0, 1], 64, ticks_per_latent=4)
    assert cos.shape == (3, 6, 32)
    # t axis (first 12 freqs) of the frame-1 patch (0,0) uses tick 4; action slot j at frame 1 uses tick 5
    assert torch.allclose(rope_tables(3, (2, 2), [0], 64, 4)[0][1, 4], cos[1, 4])
    assert torch.allclose(cos[0, 0, 12:], torch.ones(20))  # h,w = 0 at the origin


def test_rope_attention_identity_matches_timm_attention():
    from timm.models.vision_transformer import Attention
    torch.manual_seed(0)
    ref, ra = Attention(64, num_heads=4, qkv_bias=True), RopeAttention(64, num_heads=4, qkv_bias=True)
    ra.load_state_dict(ref.state_dict())  # same qkv/proj param names
    x = torch.randn(2, 9, 64)
    ones, zeros = torch.ones(1, 9, 8), torch.zeros(1, 9, 8)
    assert torch.allclose(ref(x), ra(x, rope=(ones, zeros)), atol=1e-5)
    mask = torch.rand(9, 9) > 0.3
    mask |= torch.eye(9, dtype=torch.bool)
    assert torch.allclose(ref(x, attn_mask=mask), ra(x, attn_mask=mask), atol=1e-5)


def test_rope_view_layouts():
    cos, sin = rope_tables(4, (2, 2), [0], 64)
    assert rope_view((cos, sin), 'spatial', 3)[0].shape == (12, 5, 32)
    assert rope_view((cos, sin), 'temporal', 3)[0].shape == (15, 4, 32)
    assert rope_view((cos, sin), 'flat', 3)[0].shape == (1, 20, 32)
    assert torch.equal(rope_view((cos, sin), 'temporal', 3)[0][5 + 2, 3], cos[3, 2])  # (b=1, n=2), t=3


@pytest.mark.parametrize("window", [None, 1])
def test_rope3d_forward_and_sincos_checkpoint_loads(window):
    torch.manual_seed(0)
    sin_model = VDT_S_2(attn_window=window, **KW)
    rope_model = VDT_S_2(attn_window=window, pos_encoding='rope3d', **KW)
    missing = rope_model.load_state_dict(sin_model.state_dict(), strict=False)
    assert not missing.missing_keys  # rope3d only drops params
    x, t, a = torch.randn(2, 4, 3, 16, 16), torch.tensor([10, 200]), torch.randn(2, 4, 5)
    v, (act, _) = rope_model(x, timesteps=t, actions=a)
    assert v.shape == x.shape and act.shape == a.shape and torch.isfinite(v).all()
    # the zero-init output layer makes the first forward trivially zero; perturb it so rope matters
    for m in (rope_model, sin_model):
        torch.nn.init.normal_(m.final_layer.linear.weight, std=0.02)
        torch.nn.init.normal_(m.action_head.linear.weight, std=0.02)
    rope_model.load_state_dict({**sin_model.state_dict()}, strict=False)
    assert not torch.allclose(rope_model(x, timesteps=t, actions=a)[0], sin_model(x, timesteps=t, actions=a)[0], atol=1e-4)


def test_rope3d_skips_position_params():
    m = VDT_S_2(pos_encoding='rope3d', **KW)
    assert m.action_pos_embed is None and m.pos_encoding == 'rope3d'


def test_sincos_forward_unchanged_vs_original_module():
    orig_path = os.environ.get("VDT_ORIG_PATH", "/ubc/cs/research/plai-scratch/ctardy/projects/continual-diffusion/run-scripts/wan_99/vdt_orig.py")
    if not os.path.exists(orig_path):
        pytest.skip("pre-#99 vdt.py snapshot not available")
    spec = importlib.util.spec_from_file_location("vdt_orig", orig_path)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    torch.manual_seed(0)
    old = orig.VDT_S_2(attn_window=1, **KW)
    new = VDT_S_2(attn_window=1, **KW)
    sd = old.state_dict()
    for k in sd:
        if "final_layer" in k or "action_head" in k:
            sd[k] = torch.randn_like(sd[k]) * 0.02
    old.load_state_dict(sd)
    new.load_state_dict(sd)
    x, t, a = torch.randn(2, 4, 3, 16, 16), torch.tensor([10, 200]), torch.randn(2, 4, 5)
    (v0, (a0, _)), (v1, (a1, _)) = old(x, timesteps=t, actions=a), new(x, timesteps=t, actions=a)
    assert torch.equal(v0, v1) and torch.equal(a0, a1)


def test_backfill_pos_encoding():
    import argparse
    ns = argparse.Namespace()
    backfill_pos_encoding(ns)
    assert ns.pos_encoding == "sincos"
    ns.pos_encoding = "rope3d"
    backfill_pos_encoding(ns)
    assert ns.pos_encoding == "rope3d"
