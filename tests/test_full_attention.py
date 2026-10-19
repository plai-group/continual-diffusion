"""attention=full in VDT: plaicraft-debug#99."""

import pytest
import torch

from improved_diffusion.script_util import backfill_attention
from improved_diffusion.vdt import VDT_S_2

KW = dict(input_size=16, patch_size=4, in_channels=3, learn_sigma=False, action_dim=5, generate_actions=True)


def _perturb(m):
    for mod in (m.final_layer, m.action_head):
        torch.nn.init.normal_(mod.linear.weight, std=0.02)


@pytest.mark.parametrize("pos", ["sincos", "rope3d"])
def test_full_shapes_and_no_temporal_sublayer(pos):
    torch.manual_seed(0)
    m = VDT_S_2(num_frames=3, attention='full', pos_encoding=pos, **KW)
    _perturb(m)
    assert all(not hasattr(b, 'temporal_attn') for b in m.blocks)
    x, a = torch.randn(2, 3, 3, 16, 16), torch.randn(2, 3, 5)
    v, (act, _) = m(x, timesteps=torch.tensor([5, 50]), actions=a)
    assert v.shape == x.shape and act.shape == a.shape and torch.isfinite(v).all()


def test_full_attends_across_frames():
    torch.manual_seed(0)
    m = VDT_S_2(num_frames=3, attention='full', **KW).eval()
    _perturb(m)
    for b in m.blocks:  # adaLN-Zero gates start at 0; open them so attention contributes
        torch.nn.init.normal_(b.adaLN_modulation[-1].weight, std=0.02)
    x, t, a = torch.randn(1, 3, 3, 16, 16), torch.tensor([20]), torch.randn(1, 3, 5)
    x2 = x.clone()
    x2[:, 2] += 1.0
    v0, v1 = m(x, timesteps=t, actions=a)[0], m(x2, timesteps=t, actions=a)[0]
    assert not torch.allclose(v0[:, 0], v1[:, 0], atol=1e-6)  # frame 0 output sees frame 2


def test_full_equals_factorized_spatial_at_T1():
    torch.manual_seed(0)
    fact = VDT_S_2(num_frames=1, **KW).eval()
    full = VDT_S_2(num_frames=1, attention='full', **KW).eval()
    sd = {k: v for k, v in fact.state_dict().items()}
    for k in sd:
        if "final_layer" in k or "action_head" in k or "adaLN_modulation" in k:
            sd[k] = torch.randn_like(sd[k]) * 0.02
    fact.load_state_dict(sd)
    full.load_state_dict({k: v for k, v in sd.items() if 'temporal' not in k})
    # at T=1 the temporal sublayer is a residual add of temporal_fc(attn over a single token); zero it for the comparison
    for b in fact.blocks:
        torch.nn.init.zeros_(b.temporal_fc.weight)
        torch.nn.init.zeros_(b.temporal_fc.bias)
    x, t, a = torch.randn(2, 1, 3, 16, 16), torch.tensor([10, 200]), torch.randn(2, 1, 5)
    (v0, (a0, _)), (v1, (a1, _)) = fact(x, timesteps=t, actions=a), full(x, timesteps=t, actions=a)
    assert torch.allclose(v0, v1, atol=1e-5) and torch.allclose(a0, a1, atol=1e-5)


def test_full_rejects_attn_window():
    with pytest.raises(AssertionError):
        VDT_S_2(num_frames=3, attention='full', attn_window=1, **KW)


def test_backfill_attention():
    import argparse
    ns = argparse.Namespace()
    backfill_attention(ns)
    assert ns.attention == "factorized"
