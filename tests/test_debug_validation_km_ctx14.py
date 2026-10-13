"""plaicraft-debug#96: km_ctx14 wiring in debug_validation.py -- _get_km_tokenizer's dispatch
on encoding, and _decode_km_actions against a fixed code tensor (no run_debug_validation,
this tokenizer's swap-test path is unused: plaicraft_morgan corpora have no swap package)."""
import pytest
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion import debug_validation as dv
from improved_diffusion.km_tokenizer.ctx14_10hz import DEFAULT_CHECKPOINT_DIR, KmTokenizerCtx14_10Hz, KmTokenizerCtx14_10HzConfig


def test_get_km_tokenizer_dispatches_on_encoding():
    dv._KM_TOKENIZERS.clear()
    fsq_tok = dv._get_km_tokenizer("cpu", encoding="km_fsq")
    ctx14_tok = dv._get_km_tokenizer("cpu", encoding="km_ctx14")
    assert type(fsq_tok).__name__ == "KeyboardMouseTokenizer"
    assert isinstance(ctx14_tok, KmTokenizerCtx14_10Hz)
    assert dv._get_km_tokenizer("cpu", encoding="km_ctx14") is ctx14_tok  # cached per encoding


def test_decode_km_actions_on_a_fixed_code_tensor():
    torch.manual_seed(0)
    tokenizer = KmTokenizerCtx14_10Hz(KmTokenizerCtx14_10HzConfig())
    tokenizer.eval()

    B, N = 1, 3
    codes = torch.zeros(B, N, da.KM_CTX14_CODE_DIM)  # the all-zero lattice point: a fixed, reproducible tensor
    keys, mouse, probs = dv._decode_km_actions(tokenizer, codes, all_keys=True)
    assert keys.shape == (B, N, 79)
    assert mouse.shape == (B, N, 2)
    assert probs.shape == (B, N, 79)
    assert torch.equal((probs > 0.5).float(), keys)

    # The same fixed tensor decodes identically every call -- a frozen, eval-mode tokenizer.
    keys2, mouse2, probs2 = dv._decode_km_actions(tokenizer, codes, all_keys=True)
    assert torch.equal(keys, keys2) and torch.equal(mouse, mouse2) and torch.allclose(probs, probs2)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_DIR.exists(), reason="staged km_ctx14 checkpoint not present")
def test_decode_km_actions_with_the_staged_tokenizer_shapes():
    tokenizer = dv._get_km_tokenizer("cpu", encoding="km_ctx14")
    codes = torch.zeros(2, 4, da.KM_CTX14_CODE_DIM)
    keys, mouse, probs = dv._decode_km_actions(tokenizer, codes, all_keys=True)
    assert keys.shape == (2, 4, 79)
    assert mouse.shape == (2, 4, 2)
    assert probs.shape == (2, 4, 79)
