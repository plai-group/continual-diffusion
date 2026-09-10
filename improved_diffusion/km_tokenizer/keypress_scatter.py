"""Scatters the debug corpus's compact 8-dim keypress vector into the 79-wide
layout keypress_lookup.py defines, at the keys' REAL trained indices -- not
indices 0..7 (plaicraft-debug#74's cb507a9 fixed the same bug for the old
keypress autoencoder; #80 repeats the fix for the km tokenizer's own 79-wide
input, which uses a different, PLAIOmni-native channel order).

Compact dim order: [w, a, s, d, space, shift, mouse_left, mouse_right] --
matches debug_actions._KEY_IDS / debug_validation._ACTION_KEY_NAMES.
"""
import torch

from .keypress_lookup import KEYPRESS_LOOKUP

_KEY_IDS = ["87", "65", "83", "68", "32", "340"]
_CLICK_IDS = ["left", "right"]

_RAW_POSITIONS = [KEYPRESS_LOOKUP[k] for k in _KEY_IDS] + [KEYPRESS_LOOKUP[k] for k in _CLICK_IDS]


def scatter_keypress(compact: torch.Tensor) -> torch.Tensor:
    """(..., 8) compact keypress -> (..., 79) at _RAW_POSITIONS, zeros elsewhere."""
    if compact.shape[-1] != len(_RAW_POSITIONS):
        raise ValueError(f"Expected last dim {len(_RAW_POSITIONS)}, got {compact.shape[-1]}.")
    out = compact.new_zeros(*compact.shape[:-1], 79)
    out[..., _RAW_POSITIONS] = compact
    return out
