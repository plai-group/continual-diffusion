"""plaicraft-debug#106: idm=True feeds the model GT video at every frame and noise level 0, and still samples the post-context actions."""
import torch

import improved_diffusion.debug_validation as dv
from improved_diffusion.script_util import create_vdt_model_and_diffusion

B, T, N_OBS, C, H, AD = 2, 6, 3, 3, 32, 8


class _Valset:
    T, n_observed = T, N_OBS
    rows = [{"num": i, "prompt": "p", "test_type": "t"} for i in range(B)]

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.frames = torch.rand(B, T, C, H, H, generator=g) * 2 - 1
        self.keys = (torch.rand(B, T, 8, generator=g) < 0.3).float()

    def load_all(self):
        return self.frames

    def load_all_actions(self):
        return self.keys, torch.zeros(B, T, 2)

    load_all_actions_raw = load_all_actions

    def slug(self, row):
        return str(row["num"])


def _run(monkeypatch, tmp_path, idm):
    monkeypatch.setattr(dv, "_METRICS", lambda p, g: {"psnr": 1.0})
    model, diffusion = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(H, H), in_channels=C, num_frames=T, learn_sigma=False,
        sigma_small=False, diffusion_steps=100, diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False, predict_xstart=False, rescale_timesteps=True,
        rescale_learned_sigmas=True, use_checkpoint=False, use_edm_scaling=False, action_dim=8, action_dropout_prob=0.0,
        generate_actions=True, independent_action_t=True)
    seen = []
    model.register_forward_pre_hook(lambda m, a, kw: seen.append((a, dict(kw))), with_kwargs=True)
    vs = _Valset()
    res = dv.run_debug_validation(model.eval(), diffusion, vs, "cpu", tmp_path, log_videos=False, swap_test=False,
                                  fvd_repeats=0, idm=idm)
    return vs, seen, res


def test_idm_feeds_gt_video_everywhere_and_samples_actions(monkeypatch, tmp_path):
    vs, seen, res = _run(monkeypatch, tmp_path, idm=True)
    assert seen
    for (x, *_), kw in seen:
        assert torch.equal(kw["obs_mask"], torch.ones(B, T, 1, 1, 1))
        assert torch.equal(x * (1 - kw["obs_mask"]) + kw["x0"] * kw["obs_mask"], vs.frames)  # what VDT.forward feeds its patch embedder
        assert (kw["timesteps"] == 0).all() and (kw["action_timesteps"] >= 0).all()
        assert kw["obs_action_mask"][:, :N_OBS + 1].eq(1).all() and kw["obs_action_mask"][:, N_OBS + 1:].eq(0).all()  # context pinned, rest sampled
        assert not torch.equal(kw["actions"][:, N_OBS + 1:], vs.keys[:, N_OBS + 1:])
    assert "val/action/key_jaccard_distance" in res["aggregate"]


def test_normal_mode_still_masks_the_generated_frames(monkeypatch, tmp_path):
    _, seen, _ = _run(monkeypatch, tmp_path, idm=False)
    assert all(kw["obs_mask"][:, N_OBS:].eq(0).all() and kw.get("action_timesteps") is None for _, kw in seen)


def test_idm_shared_t_keeps_the_sampler_timestep_for_the_video(monkeypatch, tmp_path):
    monkeypatch.setattr(dv, "_IDMWrapper", lambda m, t0, _w=dv._IDMWrapper: _w(m, False))
    _, seen, _ = _run(monkeypatch, tmp_path, idm=True)
    assert any((kw["timesteps"] > 0).any() for _, kw in seen) and all(torch.equal(kw["timesteps"], kw["action_timesteps"]) for _, kw in seen)
