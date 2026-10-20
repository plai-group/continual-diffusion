"""Windowed cross-frame action attention (attn_window) in VDT: plaicraft-debug#97."""

import torch

from improved_diffusion.vdt import VDT, VDT_S_2, window_mask


def test_window_mask_shape_and_rules():
    T, N, k, w = 4, 6, 1, 1
    n_tok = N + k
    mask = window_mask(T, N, n_tok, w)
    assert mask.shape == (T * n_tok, T * n_tok)
    assert mask.dtype == torch.bool

    def idx(t, i):
        return t * n_tok + i

    # same frame: every pair allowed, including patch-patch
    for i in range(n_tok):
        for j in range(n_tok):
            assert mask[idx(0, i), idx(0, j)]

    # patch-patch across frames is blocked even inside the window
    assert not mask[idx(0, 0), idx(1, 0)]
    # action-patch within the window is allowed, both directions
    assert mask[idx(0, N), idx(1, 0)]
    assert mask[idx(1, 0), idx(0, N)]
    # action-patch beyond the window is blocked
    assert not mask[idx(0, N), idx(2, 0)]
    # action-action: allowed inside the window, blocked beyond it
    assert mask[idx(0, N), idx(1, N)]
    assert not mask[idx(0, N), idx(3, N)]
    print("  [PASS] window_mask same-frame/patch/action rules")


def test_attn_window_zero_matches_unwindowed():
    torch.manual_seed(0)
    B, T, C, action_dim = 2, 4, 3, 5
    kwargs = dict(
        input_size=16, patch_size=4, in_channels=C, num_frames=T, learn_sigma=False,
        action_dim=action_dim, generate_actions=True,
    )
    base = VDT_S_2(**kwargs)
    windowed = VDT_S_2(attn_window=0, **kwargs)
    windowed.load_state_dict(base.state_dict())

    x = torch.randn(B, T, C, 16, 16)
    t = torch.tensor([10, 200])
    actions = torch.randn(B, T, action_dim)

    v0, (a0, _) = base(x, timesteps=t, actions=actions)
    v1, (a1, _) = windowed(x, timesteps=t, actions=actions)
    assert torch.allclose(v0, v1, atol=1e-5)
    assert torch.allclose(a0, a1, atol=1e-5)
    print("  [PASS] attn_window=0 forward matches attn_window=None")


def test_attn_window_reaches_within_but_not_beyond():
    torch.manual_seed(0)
    B, T, C, action_dim, w = 1, 8, 3, 4, 2
    model = VDT(
        input_size=16, patch_size=4, in_channels=C, hidden_size=64, depth=1, num_heads=4,
        num_frames=T, action_dim=action_dim, generate_actions=True, attn_window=w,
    )
    model.eval()
    # DiT zero-inits adaLN gates and the action head's linear, so attention is a no-op and the
    # action output is always 0 at init; break that so the test can observe real information flow.
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(model.action_head.adaLN_modulation[-1].weight, std=0.02)
    torch.nn.init.normal_(model.action_head.linear.weight, std=0.02)
    # Single block + zeroed temporal_fc: the action token at t0 can only be reached by a
    # patch token through the windowed spatial attention added in this commit.
    for block in model.blocks:
        torch.nn.init.zeros_(block.temporal_fc.weight)
        torch.nn.init.zeros_(block.temporal_fc.bias)

    x = torch.randn(B, T, C, 16, 16)
    t0 = 3
    actions = torch.randn(B, T, action_dim)
    timesteps = torch.tensor([100])

    def action_out_at(x_in):
        with torch.no_grad():
            _, (a, _) = model(x_in, timesteps=timesteps, actions=actions)
        return a[:, t0]

    base_out = action_out_at(x)

    x_near = x.clone()
    x_near[:, t0 + w] = torch.randn_like(x_near[:, t0 + w])
    assert not torch.allclose(base_out, action_out_at(x_near), atol=1e-6)

    x_far = x.clone()
    x_far[:, t0 + w + 1] = torch.randn_like(x_far[:, t0 + w + 1])
    assert torch.allclose(base_out, action_out_at(x_far), atol=1e-6)
    print("  [PASS] action token sees patches within the window, not beyond it")


if __name__ == "__main__":
    print("Running VDT windowed-attention tests...")
    test_window_mask_shape_and_rules()
    test_attn_window_zero_matches_unwindowed()
    test_attn_window_reaches_within_but_not_beyond()
    print("ALL WINDOW-ATTENTION TESTS PASSED")
