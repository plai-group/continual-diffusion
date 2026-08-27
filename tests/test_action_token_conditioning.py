"""Token conditioning: the action rides in the sequence but is never denoised.

The point of the mode is to separate two things #69 changed at once -- putting
an action token in the sequence, and asking the model to denoise it. These
tests pin that the token is live (it reaches the video output), that nothing
in the loss or the sampler treats it as a generated quantity, and that a
trunk trained with adaLN conditioning can be warm-started into it.
"""
import torch

from improved_diffusion.vdt import VDT_S_2
from improved_diffusion.script_util import (
    create_vdt_model_and_diffusion,
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
    v, a = m(torch.randn(B, T, C, H, W), timesteps=torch.tensor([50, 150]),
             actions=torch.randn(B, T, ACTION_DIM))
    assert v.shape == (B, T, C, H, W)
    assert a is None


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
