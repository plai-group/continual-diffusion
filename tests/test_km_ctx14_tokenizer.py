"""plaicraft-debug#96: the vendored ctx14 10Hz keyboard/mouse tokenizer."""
import pytest
import torch

from improved_diffusion.km_tokenizer.ctx14_10hz import (
    DEFAULT_CHECKPOINT_DIR,
    KmTokenizerCtx14_10Hz,
    KmTokenizerCtx14_10HzConfig,
    load_ctx14_tokenizer,
)


def test_encode_decode_round_trip_shapes():
    torch.manual_seed(0)
    model = KmTokenizerCtx14_10Hz(KmTokenizerCtx14_10HzConfig())
    model.eval()

    B, T = 2, 20  # 2 ticks of 10 at 100Hz
    key_press = (torch.rand(B, T, 79) > 0.9).float()
    mouse_movement = torch.randint(-50, 51, (B, T, 2)).float()

    with torch.no_grad():
        out = model(key_press, mouse_movement)

    N = T // model.config.block_size
    assert out.token_ids.shape == (B, N, model.config.num_tokens)
    assert out.token_ids.min() >= 0 and out.token_ids.max() < model.config.vocab_size
    assert out.key_logits.shape == (B, N, model.config.block_size, 79)
    assert out.mouse_pred.shape == (B, N, model.config.block_size, 2)

    with torch.no_grad():
        ids = model.encode_tokens(key_press, mouse_movement)
        key_logits2, _mouse_logits2, mouse_pred2 = model.decode_tokens(ids)
    assert torch.equal(ids, out.token_ids)
    assert torch.allclose(key_logits2, out.key_logits, atol=1e-5)
    assert torch.allclose(mouse_pred2, out.mouse_pred, atol=1e-5)


def test_load_ctx14_tokenizer_rejects_corrupted_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text('{"num_keys": 79}')
    (tmp_path / "model.safetensors").write_bytes(b"not a real checkpoint")
    with pytest.raises(RuntimeError, match="sha256"):
        load_ctx14_tokenizer(checkpoint_dir=tmp_path)


def test_load_ctx14_tokenizer_from_staged_checkpoint_encodes_and_decodes():
    tokenizer = load_ctx14_tokenizer(checkpoint_dir=DEFAULT_CHECKPOINT_DIR, device="cpu")
    assert tokenizer.config.num_tokens == 8 and tokenizer.config.fsq_dim == 4

    B, T = 1, 1000  # 10s at 100Hz
    key_press = torch.zeros(B, T, 79)
    mouse_movement = torch.zeros(B, T, 2)
    with torch.no_grad():
        ids = tokenizer.encode_tokens(key_press, mouse_movement)
        key_logits, mouse_logits, mouse_px = tokenizer.decode_tokens(ids)
    assert ids.shape == (B, 100, 8)
    assert key_logits.shape == (B, 100, 10, 79)
    assert mouse_px.shape == (B, 100, 10, 2)
