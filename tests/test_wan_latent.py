"""Wan2.2 latent backend + ticks_per_latent validation plumbing: plaicraft-debug#99. The VAE is a stub; no weights are downloaded."""
import json

import numpy as np
import pytest
import torch as th
import torch.nn as nn

from improved_diffusion import debug_actions
from improved_diffusion.corpus_validation import CorpusValidationSet
from improved_diffusion.debug_validation import _decode_km_actions, _render_panels
from improved_diffusion.km_tokenizer import ctx14_10hz
from improved_diffusion.script_util import create_gaussian_diffusion

MEAN, STD = 0.5, 2.0


class StubWanVAE:
    """decode(z: (1,48,t,h,w)) -> (1,3,1+4(t-1),4h,4w); frame f carries the value of the latent that produced it."""
    device = th.device("cpu")

    def to(self, *_a, **_k):
        return self

    def decode(self, z):
        self.last_z = z
        t = z.shape[2]
        idx = th.tensor([0] + [1 + j // 4 for j in range(4 * (t - 1))])  # latent index per output frame
        v = z.mean(dim=1, keepdim=True)[:, :, idx]  # (1,1,F,h,w), carries denormalized latent values
        out = v.repeat(1, 3, 1, 1, 1).repeat_interleave(4, -1).repeat_interleave(4, -2)
        return type("O", (), {"sample": out})()


def _wan_diffusion():
    d = create_gaussian_diffusion(diffusion_space_kwargs=dict(diffusion_space="latent", pre_encoded=True, enable_decoding=False, vae="wan22"))
    d.enc_dec_dtype = th.float32
    d.vae, d.wan_mean, d.wan_std = StubWanVAE(), th.full((1, 48, 1, 1, 1), MEAN), th.full((1, 48, 1, 1, 1), STD)
    return d


def test_wan_decode_frame_count_and_gen_slice():
    d = _wan_diffusion()
    T, n_obs, h, w = 6, 3, 2, 3
    # latent j holds the constant j/100 (normalized), so its decoded value is j/100*STD + MEAN
    z = th.stack([th.full((48, h, w), j / 100) for j in range(T)]).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    px = d.decode(z)
    assert d.px_per_latent == 4 and px.shape == (2, 1 + 4 * (T - 1), 3, 4 * h, 4 * w)
    # the stub saw denormalized latents as ONE clip of T latents
    assert d.vae.last_z.shape == (1, 48, T, h, w) and th.allclose(d.vae.last_z[0, 0, :, 0, 0], th.arange(T) / 100 * STD + MEAN)
    gen = px[:, -d.px_per_latent * (T - n_obs):]
    assert gen.shape[1] == 12
    for g in range(T - n_obs):  # each GEN latent owns 4 consecutive frames at the end
        assert th.allclose(gen[:, 4 * g:4 * g + 4], th.full_like(gen[:, :4], (n_obs + g) / 100 * STD + MEAN))


def test_wan_decode_is_clamped():
    d = _wan_diffusion()
    px = d.decode(th.full((1, 3, 48, 2, 2), 5.0))
    assert px.shape[1] == 9 and px.max() == 1.0


def test_sdxl_default_has_one_pixel_frame_per_latent():
    d = create_gaussian_diffusion(diffusion_space_kwargs=dict(diffusion_space="latent", pre_encoded=True, enable_decoding=False))
    assert d.vae_name == "sdxl" and d.px_per_latent == 1


class _FakeTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(th.zeros(1))
        self.config = type("C", (), {"num_tokens": 2, "fsq_dim": 16})()

    def decode_codes(self, codes):
        self.seen = codes.shape
        B, T = codes.shape[:2]
        return th.zeros(B, T, 10, 79), th.zeros(B, T, 10, 2), th.zeros(B, T, 10, 2)


def test_decode_km_actions_unpacks_k_ticks_per_latent():
    tok = _FakeTokenizer()
    keys, mouse, probs = _decode_km_actions(tok, th.zeros(2, 6, 4 * 32), all_keys=True)  # (B, T, k*32)
    assert tuple(tok.seen) == (2, 24, 2, 16) and keys.shape == (2, 24, 79) and mouse.shape == (2, 24, 2)
    _decode_km_actions(tok, th.zeros(2, 6, 32), all_keys=True)
    assert tuple(tok.seen) == (2, 6, 2, 16)  # k=1 unchanged


def test_render_panels_right_aligns_ticks_to_wan_frames(tmp_path, monkeypatch):
    import imageio

    written = []
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: type("W", (), {"append_data": lambda _s, f: written.append(f), "close": lambda _s: None})())
    F, ticks = 9, 12  # 1+4*2 frames, 3 latents * 4 ticks
    frames = np.random.RandomState(0).uniform(-1, 1, (F, 3, 24, 40)).astype(np.float32)
    acts = (np.zeros((ticks, 79), np.float32), np.zeros((ticks, 2), np.float32))
    _render_panels((frames, frames), (acts, acts), ["GT", "GEN"], 5, str(tmp_path / "o.mp4"), ncols=2, fps=10)
    assert len(written) == F


K, T, N_OBS = 4, 6, 3
NAMES = ["a", "b"]


def _fixture(tmp_path):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    ws = np.array([100, 200], dtype=np.int64)
    np.savez(vdir / "validation.npz", frames=np.zeros((n, T, 48, 2, 2), np.float32),
             keypress=np.zeros((n, K * T, 8), np.float32), mouse=np.zeros((n, K * T, 2), np.float32),
             names=np.array(NAMES), session_ids=np.array(["s0", "s1"]),
             window_start_ticks=ws, boundary_ticks=ws + K * N_OBS)
    (vdir / "manifest.json").write_text(json.dumps({
        "schema_version": "1", "corpus_dir": str(tmp_path), "tick_ms": 100, "subbin_ms": 10, "subbin_rule_version": "1",
        "T": T, "n_observed": N_OBS, "n_held_out_sessions": 20, "built_at": "2026-01-01T00:00:00Z",
        "exercises": [{"index": i, "name": nm, "session_id": f"s{i}"} for i, nm in enumerate(NAMES)]}))
    return vdir


def test_corpus_validation_k4_actions_are_per_tick_and_codes_pack_k_per_latent(tmp_path, monkeypatch):
    vdir = _fixture(tmp_path)
    monkeypatch.setattr(ctx14_10hz, "load_ctx14_tokenizer", lambda checkpoint_dir, device: object())
    full = {f"s{i}": np.arange(500 * 32, dtype=np.float32).reshape(500, 32) + 1e5 * i for i in range(2)}
    monkeypatch.setattr(debug_actions, "load_or_build", lambda d, **kw: (full[d.name], None))
    calls = []

    def ctx(session_dir, ws, boundary, tok, dev, builder):
        calls.append((ws, boundary))
        return np.full((boundary - ws, 32), -1.0, np.float32)

    monkeypatch.setattr(debug_actions, "encode_km_ctx14_context", ctx)
    # session_root also reads raw arrays from disk; stub them
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((500, 8)), np.zeros((500, 2))))
    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, action_encoding="km_ctx14", session_root=tmp_path / "c",
                             action_builder="raw79", ticks_per_latent=K)
    key_raw, mouse_raw = vs.load_all_actions_raw()
    assert key_raw.shape == (2, K * T, 8) and mouse_raw.shape == (2, K * T, 2)
    codes, mouse = vs.load_all_actions()
    assert codes.shape == (2, T, K * 32) and mouse.shape == (2, T, 0)
    assert calls == [(100, 100 + K * N_OBS), (200, 200 + K * N_OBS)]
    per_tick = codes[0].reshape(K * T, 32).numpy()  # (T, k*32) -> (T*k, 32) recovers the tick stream
    assert np.all(per_tick[:K * N_OBS] == -1.0)
    assert np.array_equal(per_tick[K * N_OBS:], full["s0"][100 + K * N_OBS:100 + K * T])


def test_corpus_validation_rejects_boundary_not_at_k_times_n_observed(tmp_path):
    vdir = _fixture(tmp_path)
    with pytest.raises(ValueError):
        CorpusValidationSet(vdir, T=T, n_observed=N_OBS)  # k=1 expects T-length action arrays


def test_wan_decode_with_prefix_returns_4T_frames_and_drops_prefix_frame():
    d = _wan_diffusion()
    T, h, w = 6, 2, 3
    z = th.stack([th.full((48, h, w), (j + 1) / 100) for j in range(T)]).unsqueeze(0)
    prefix = th.full((1, 1, 48, h, w), 0.5)
    px = d.decode(z, prefix=prefix)
    assert px.shape == (1, 4 * T, 3, 4 * h, 4 * w) and d.vae.last_z.shape == (1, 48, T + 1, h, w)
    assert th.allclose(d.vae.last_z[0, 0, 0], th.tensor(0.5 * STD + MEAN))  # prefix is latent 0 of the clip
    for j in range(T):  # window latent j owns frames 4j..4j+3, aligned with its ticks
        assert th.allclose(px[:, 4 * j:4 * j + 4], th.full_like(px[:, :4], (j + 1) / 100 * STD + MEAN))
    assert d.decode(z).shape[1] == 1 + 4 * (T - 1)  # no prefix: unchanged


def test_decode_if_latent_forwards_prefix_only_when_given():
    class D:
        diffusion_space = "latent"
        def decode(self, v, chunk_size=20, prefix=None): return (v, prefix)
    from improved_diffusion.debug_validation import _decode_if_latent
    assert _decode_if_latent(D(), 1)[1] is None and _decode_if_latent(D(), 1, 2)[1] == 2


def test_render_panels_with_prefix_aligns_frame_i_with_tick_i(tmp_path, monkeypatch):
    import imageio

    written = []
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: type("W", (), {"append_data": lambda _s, f: written.append(f), "close": lambda _s: None})())
    seen = []
    monkeypatch.setattr("improved_diffusion.debug_validation._action_vec_to_bar", lambda k, m: seen.append(float(k[0])))
    monkeypatch.setattr("improved_diffusion.debug_validation._overlay_frame", lambda f, bar, border: f)
    F = 12  # 3 latents * 4 ticks; prefix decode => frames == ticks
    frames = np.zeros((F, 3, 24, 40), np.float32)
    acts = (np.arange(F, dtype=np.float32)[:, None].repeat(79, 1), np.zeros((F, 2), np.float32))
    _render_panels((frames,), (acts,), ["GT"], 4, str(tmp_path / "o.mp4"), ncols=1, fps=10)
    assert seen[:-1] == list(range(1, F))  # frame i shows tick i's display-shifted bar (cache[i+1]); no extra right-align offset
