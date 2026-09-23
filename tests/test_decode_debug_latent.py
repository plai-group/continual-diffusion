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
