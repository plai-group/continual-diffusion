"""Token conditioning: the action rides in the sequence but is never denoised.

The point of the mode is to separate two things #69 changed at once -- putting
an action token in the sequence, and asking the model to denoise it. These
tests pin that the token is live (it reaches the video output), that nothing
in the loss or the sampler treats it as a generated quantity, and that a
trunk trained with adaLN conditioning can be warm-started into it.
"""
import pytest
import torch

from improved_diffusion.vdt import VDT_S_2
from improved_diffusion.script_util import (
    create_vdt_model_and_diffusion,
    create_gaussian_diffusion,
    vdt_model_and_diffusion_defaults,
)

B, T, C, H, W = 2, 8, 3, 32, 32
ACTION_DIM = 10


def _model(**kw):
    return VDT_S_2(input_size=H, patch_size=4, in_channels=C, num_frames=T,
                   learn_sigma=False, action_dim=ACTION_DIM, **kw)


def test_token_cond_builds_the_token_path_but_no_head():
    m = _model(action_token_cond=True)
    assert m.action_x_embedder is not None
    assert m.action_pos_embed is not None
    assert m.action_head is None, "token-cond must not build a generation head"
    assert m.action_embedder is None, "token-cond must not also adaLN-condition"
    assert m.generate_actions is False


def test_generation_wins_when_both_flags_are_set():
    """generate_actions is the stronger claim; token_cond must not disarm it."""
    m = _model(generate_actions=True, action_token_cond=True)
    assert m.action_token_cond is False
    assert m.action_head is not None


def test_token_cond_returns_no_action_output():
    m = _model(action_token_cond=True)
    v, (a, mouse) = m(torch.randn(B, T, C, H, W), timesteps=torch.tensor([50, 150]),
                       actions=torch.randn(B, T, ACTION_DIM))
    assert v.shape == (B, T, C, H, W)
    assert a is None and mouse is None


def test_the_action_token_actually_reaches_the_video_output():
    """A token that changes nothing is the failure this whole mode is about.

    Every adaLN gate is zero-init, which makes an untrained VDT an exact
    identity map -- the action token would be provably inert for the wrong
    reason. Open the gates first so this measures mixing, not initialisation.
    """
    m = _model(action_token_cond=True).eval()
    for blk in m.blocks:
        torch.nn.init.normal_(blk.adaLN_modulation[-1].bias, std=0.5)
    torch.nn.init.normal_(m.final_layer.adaLN_modulation[-1].bias, std=0.5)
    torch.nn.init.normal_(m.final_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(m.action_x_embedder.weight, std=0.02)
    x, t = torch.randn(B, T, C, H, W), torch.tensor([50, 150])
    with torch.no_grad():
        a = m(x, timesteps=t, actions=torch.zeros(B, T, ACTION_DIM))[0]
        b = m(x, timesteps=t, actions=torch.ones(B, T, ACTION_DIM))[0]
    assert not torch.allclose(a, b), "video output is independent of the action"


def test_token_cond_loss_has_no_action_term():
    """No obs_action_mask/actions0 from the caller => nothing to denoise."""
    kw = vdt_model_and_diffusion_defaults()
    kw.update(model_name="VDT-S", patch_size=4, input_size=H, in_channels=C,
              num_frames=T, learn_sigma=False, action_dim=ACTION_DIM,
              action_token_cond=True, diffusion_steps=100,
              diffusion_space_kwargs=dict(diffusion_space="pixel",
                                          pre_encoded=False,
                                          enable_decoding=False))
    model, diffusion = create_vdt_model_and_diffusion(**kw)
    x = torch.randn(B, T, C, H, W)
    terms = diffusion.training_losses(
        model, x, torch.tensor([10, 20]),
        model_kwargs={"actions": torch.randn(B, T, ACTION_DIM)},
    )
    assert "loss_action" not in terms
    assert torch.allclose(terms["loss"], terms["loss_video"])


def test_warm_start_reuses_the_trunk_and_zeroes_the_token_path():
    """An adaLN-conditioned donor shares everything except the action path."""
    donor = _model()                       # action_embedder, no tokens
    target = _model(action_token_cond=True)  # tokens, no action_embedder
    missing, unexpected = target.load_state_dict(donor.state_dict(), strict=False)

    assert all("action_embedder" in k for k in unexpected), unexpected
    assert all("action_x_embedder" in k or "action_pos_embed" in k
               for k in missing), missing
    shared = len(donor.state_dict()) - len(unexpected)
    assert shared > 200, f"only {shared} tensors carried over"

    # Mirrors TrainLoop._warm_start: the new token starts as a constant, not as random noise in the sequence.
    for name, p in target.named_parameters():
        if name in missing:
            with torch.no_grad():
                p.zero_()
    for name, p in target.named_parameters():
        if name.startswith("blocks.0.attn"):
            assert torch.allclose(p, dict(donor.named_parameters())[name])
    assert target.action_x_embedder.weight.abs().sum() == 0


# ── composing with the issue-85 conditioning path ──────────────────────────────────────────

def test_token_cond_composes_with_concat_ln():
    """The #85 fourth-pass arms depend on this and nothing else pinned it.

    cond_combine != "add" is unsupported alongside ActionEmbedder, because that path adds a
    per-frame (B,T,D) action term straight into c. Token-cond leaves action_embedder None and
    routes the action through the sequence instead, so c stays (B,D) and cond_proj applies.
    """
    m = _model(action_token_cond=True, cond_combine="concat_ln", num_classes=2,
               class_dropout_prob=0.0).eval()
    assert m.action_embedder is None and m.cond_proj is not None
    x, t = torch.randn(B, T, C, H, W), torch.tensor([50, 150])
    acts = torch.randn(B, T, ACTION_DIM)
    with torch.no_grad():
        for param in m.parameters():
            if param.count_nonzero() == 0:
                param.normal_(0.0, 0.05)
        p0 = m(x, timesteps=t, actions=acts, y=torch.zeros(B, dtype=torch.long))[0]
        p1 = m(x, timesteps=t, actions=acts, y=torch.ones(B, dtype=torch.long))[0]
    assert not torch.allclose(p0, p1), "the player label must still reach the output"


def test_adaln_action_conditioning_still_refuses_concat_ln():
    """The other side of the same guard: silently ignoring cond_combine would ship a run
    whose label was folded into c by addition while its config claimed otherwise."""
    m = _model(cond_combine="concat_ln", num_classes=2, class_dropout_prob=0.0)
    assert m.action_embedder is not None, "this test is only meaningful on the adaLN path"
    with pytest.raises(AssertionError, match="cond_combine"):
        m(torch.randn(B, T, C, H, W), timesteps=torch.tensor([50, 150]),
          actions=torch.randn(B, T, ACTION_DIM), y=torch.zeros(B, dtype=torch.long))


# ── independent_action_t: the action token's own noise level (issue #85) ───────────────────

def _action_gen_model(**kw):
    """generate_actions=True with the action_head de-zeroed, or act_out is vacuously constant."""
    m = _model(generate_actions=True, **kw).eval()
    torch.nn.init.normal_(m.action_head.linear.weight, std=0.1)
    torch.nn.init.normal_(m.action_head.adaLN_modulation[-1].weight, std=0.1)
    return m


def test_independent_action_t_changes_act_out_with_action_timesteps():
    m = _action_gen_model(independent_action_t=True)
    x, t = torch.randn(B, T, C, H, W), torch.tensor([50, 150])
    acts = torch.randn(B, T, ACTION_DIM)
    with torch.no_grad():
        a = m(x, timesteps=t, actions=acts, action_timesteps=torch.tensor([10, 20]))[1][0]
        b = m(x, timesteps=t, actions=acts, action_timesteps=torch.tensor([900, 950]))[1][0]
    assert not torch.allclose(a, b)


def test_independent_action_t_none_matches_passing_video_timesteps():
    m = _action_gen_model(independent_action_t=True)
    x, t = torch.randn(B, T, C, H, W), torch.tensor([50, 150])
    acts = torch.randn(B, T, ACTION_DIM)
    with torch.no_grad():
        a = m(x, timesteps=t, actions=acts)[1][0]
        b = m(x, timesteps=t, actions=acts, action_timesteps=t)[1][0]
    assert torch.allclose(a, b)


def test_independent_action_t_off_ignores_action_timesteps():
    m = _action_gen_model()  # independent_action_t defaults False
    x, t = torch.randn(B, T, C, H, W), torch.tensor([50, 150])
    acts = torch.randn(B, T, ACTION_DIM)
    with torch.no_grad():
        a = m(x, timesteps=t, actions=acts)[1][0]
        b = m(x, timesteps=t, actions=acts, action_timesteps=torch.tensor([900, 950]))[1][0]
    assert torch.allclose(a, b)


class _CapturingActionModel:
    """Stub standing in for VDT: records the kwargs training_losses calls it with."""
    def __init__(self, action_dim, independent_action_t):
        self.generate_actions = True
        self.action_dim = action_dim
        self.independent_action_t = independent_action_t
        self.calls = []

    def __call__(self, x_t, timesteps=None, **kwargs):
        self.calls.append(dict(timesteps=timesteps, **kwargs))
        actions = kwargs["actions"]
        return torch.zeros_like(x_t), (torch.zeros_like(actions), None)


def _stub_diffusion():
    return create_gaussian_diffusion(
        steps=100, timestep_respacing="",
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False))


def test_training_losses_passes_independent_action_timesteps():
    diffusion = _stub_diffusion()
    N = 64
    x = torch.randn(N, 1, 3, 4, 4)
    actions = torch.randn(N, 1, 5)
    t = torch.randint(0, 100, (N,))
    model = _CapturingActionModel(action_dim=5, independent_action_t=True)
    diffusion.training_losses(model, x, t, model_kwargs={"actions": actions})
    call = model.calls[0]
    assert "action_timesteps" in call
    assert not torch.equal(call["action_timesteps"], call["timesteps"])


def test_training_losses_omits_action_timesteps_when_flag_off():
    diffusion = _stub_diffusion()
    N = 64
    x = torch.randn(N, 1, 3, 4, 4)
    actions = torch.randn(N, 1, 5)
    t = torch.randint(0, 100, (N,))
    model = _CapturingActionModel(action_dim=5, independent_action_t=False)
    diffusion.training_losses(model, x, t, model_kwargs={"actions": actions})
    assert "action_timesteps" not in model.calls[0]
