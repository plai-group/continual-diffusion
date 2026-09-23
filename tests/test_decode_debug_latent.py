"""plaicraft-debug#93 B4: decode_debug's latent-frame decode gate (maybe_decode_frames)."""
import numpy as np
import torch

from improved_diffusion import decode_debug as dd


class _StubVAEOutput:
    def __init__(self, sample):
        self.sample = sample


class _StubVAE:
    """Mimics diffusers AutoencoderKL's .decode(x).sample interface: returns 3 channels,
    same spatial size, and records how many chunks it was called with."""

    def __init__(self):
        self.calls = []

    def parameters(self):
        return iter([torch.zeros(1)])

    def decode(self, chunk):
        self.calls.append(chunk.shape[0])
        n, _c, h, w = chunk.shape
        return _StubVAEOutput(torch.zeros(n, 3, h, w))


def test_c3_frames_pass_through_unchanged():
    frames = np.random.randn(4, 3, 24, 40).astype(np.float32)
    vae = _StubVAE()
    out = dd.maybe_decode_frames(frames, vae=vae)
    assert out is frames  # no decode call at all
    assert vae.calls == []


def test_c4_frames_are_decoded_and_chunked():
    frames = torch.randn(5, 4, 24, 40)
    vae = _StubVAE()
    out = dd.maybe_decode_frames(frames, vae=vae, chunk_size=2)
    assert out.shape == (5, 3, 24, 40)
    assert vae.calls == [2, 2, 1]  # chunked into ceil(5/2) == 3 calls


def test_unexpected_channel_count_raises():
    frames = np.zeros((2, 5, 24, 40), dtype=np.float32)
    import pytest
    with pytest.raises(ValueError):
        dd.maybe_decode_frames(frames, vae=_StubVAE())


def test_decode_latents_chunked_divides_by_latent_scale():
    """decode_latents_chunked must un-scale by 0.13025 before vae.decode, the same
    convention gaussian_diffusion.decode uses."""

    class _RecordingVAE(_StubVAE):
        def decode(self, chunk):
            self.seen = chunk
            return super().decode(chunk)

    vae = _RecordingVAE()
    latents = torch.ones(1, 4, 2, 2) * dd._LATENT_SCALE
    dd.decode_latents_chunked(vae, latents, chunk_size=10)
    assert torch.allclose(vae.seen, torch.ones(1, 4, 2, 2))


def test_render_clip_decodes_latents_and_draws_one_frame_per_tick(tmp_path, monkeypatch):
    import h5py
    import scripts.render_session_overlay as rso
    sd = tmp_path / "abcd"
    (sd / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(sd / "encoded_video_hdf5" / "abcd.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((12, 4, 24, 40), np.float16))
    monkeypatch.setattr(rso, "load_or_build_raw", lambda d, action_builder: (np.zeros((12, 8)), np.zeros((12, 2))))
    written = []
    monkeypatch.setattr(rso.imageio, "get_writer", lambda *a, **k: type("W", (), {
        "append_data": lambda s, im: written.append(im.shape), "close": lambda s: None})())
    vae = _StubVAE()
    assert rso.render_clip(sd, 2, 5, tmp_path / "o.mp4", vae=vae) == 5
    assert vae.calls and len(written) == 5 and written[0][2] == 3
