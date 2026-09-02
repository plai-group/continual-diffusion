"""plaicraft-debug#80: the vendored grouped-FSQ keyboard/mouse tokenizer.

_RAW_POSITIONS is pinned against a fixture of real plaicraft-debug key labels
(tests/fixtures/keypress_layout_ground_truth.npz, copied from issue-76's
456b7eb) rather than a round trip through our own scatter/lookup, so a
channel-mapping bug in _RAW_POSITIONS can't cancel itself out the way a round
trip would (see plaicraft-debug#74's cb507a9 for exactly that failure mode).
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from improved_diffusion.km_tokenizer import keypress_lookup
from improved_diffusion.km_tokenizer.keypress_scatter import _RAW_POSITIONS, scatter_keypress
from improved_diffusion.km_tokenizer.model import (
    KeyboardMouseTokenizer,
    KeyboardMouseTokenizerConfig,
    load_tokenizer,
)

FIXTURE = Path(__file__).parent / "fixtures" / "keypress_layout_ground_truth.npz"


def test_encode_decode_round_trip_shapes():
    torch.manual_seed(0)
    model = KeyboardMouseTokenizer(KeyboardMouseTokenizerConfig())
    model.eval()

    B, T = 2, 16  # 2 blocks of 8 at 100Hz
    key_press = (torch.rand(B, T, 79) > 0.9).float()
    mouse_movement = torch.randint(-50, 51, (B, T, 2)).float()

    with torch.no_grad():
        out = model(key_press, mouse_movement)

    N = T // model.config.block_size
    assert out.token_ids.shape == (B, N, model.config.num_tokens)
    assert out.token_ids.min() >= 0 and out.token_ids.max() < model.config.vocab_size
    assert out.key_logits.shape == (B, N, model.config.block_size, 79)
    assert out.mouse_pred.shape == (B, N, model.config.block_size, 2)

    # encode_tokens/decode_tokens must agree with forward()'s own quantize/decode.
    with torch.no_grad():
        ids = model.encode_tokens(key_press, mouse_movement)
        key_logits2, _mouse_logits2, mouse_pred2 = model.decode_tokens(ids)
    assert torch.equal(ids, out.token_ids)
    assert torch.allclose(key_logits2, out.key_logits, atol=1e-5)
    assert torch.allclose(mouse_pred2, out.mouse_pred, atol=1e-5)


def test_fsq_indices_round_trip_is_exact():
    torch.manual_seed(1)
    fsq = KeyboardMouseTokenizer(KeyboardMouseTokenizerConfig()).fsq
    codes = torch.rand(4, 12, 3) * 2 - 1
    quantized, indices = fsq(codes)
    recovered = fsq.indices_to_codes(indices)
    assert torch.allclose(quantized, recovered, atol=1e-5)


def test_no_vector_quantize_pytorch_import():
    import improved_diffusion.km_tokenizer.model as m

    with open(m.__file__) as f:
        lines = f.readlines()
    assert not any(l.strip().startswith(("import vector_quantize_pytorch", "from vector_quantize_pytorch"))
                    for l in lines)


def test_load_tokenizer_rejects_corrupted_checkpoint(tmp_path):
    bad = tmp_path / "pytorch_model.bin"
    bad.write_bytes(b"not a real checkpoint")
    with pytest.raises(RuntimeError, match="sha256"):
        load_tokenizer(checkpoint_path=bad)


def test_raw_positions_length_and_uniqueness():
    assert len(_RAW_POSITIONS) == 8
    assert len(set(_RAW_POSITIONS)) == 8
    assert all(0 <= p < 79 for p in _RAW_POSITIONS)


def test_raw_positions_matches_plaicraft_debug_key_order():
    """Fixture rows are real plaicraft-debug key_press_encodings inputs (plus one
    synthetic multi-key row); labels name which compact dim(s) each row holds."""
    fixture = np.load(FIXTURE)
    raw, labels = fixture["raw"], fixture["labels"]

    label_to_key_id = {
        "w": "87", "a": "65", "s": "83", "d": "68", "space": "32",
        "shift": "340", "left": "left", "right": "right",
    }

    for row, label in zip(raw, labels):
        label = str(label)
        if label not in label_to_key_id:
            continue  # "zero" / "w+shift": covered below
        compact_idx = int(np.flatnonzero(row)[0])
        expected_pos = keypress_lookup.KEYPRESS_LOOKUP[label_to_key_id[label]]
        assert _RAW_POSITIONS[compact_idx] == expected_pos, label


def test_scatter_keypress_all_zero_row():
    fixture = np.load(FIXTURE)
    row = torch.from_numpy(fixture["raw"][list(fixture["labels"]).index("zero")])
    scattered = scatter_keypress(row)
    assert torch.equal(scattered, torch.zeros(79))


def test_scatter_keypress_multi_key_row():
    fixture = np.load(FIXTURE)
    row = torch.from_numpy(fixture["raw"][list(fixture["labels"]).index("w+shift")])
    scattered = scatter_keypress(row)
    w_pos = keypress_lookup.KEYPRESS_LOOKUP["87"]
    shift_pos = keypress_lookup.KEYPRESS_LOOKUP["340"]
    assert scattered[w_pos] == 1.0
    assert scattered[shift_pos] == 1.0
    assert scattered.sum() == 2.0
