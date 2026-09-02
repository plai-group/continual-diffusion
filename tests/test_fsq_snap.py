"""plaicraft-debug#80: the closed-form FSQ lattice snap for km_fsq action_quantization.

VDT regresses the 36-d continuous post-FSQ codes with MSE; this snap is what
projects a regressed prediction back onto the 8x6x5 lattice at inference,
exactly and without a 240-way search.
"""
import torch

from improved_diffusion.debug_actions import KM_CODE_DIM, quantize_km_fsq
from improved_diffusion.km_tokenizer.model import FSQ


def _fsq():
    return FSQ(levels=[8, 6, 5], dim=3)


def test_snap_is_idempotent_on_exact_lattice_points():
    torch.manual_seed(0)
    raw = torch.rand(5, 12, 3) * 2 - 1
    lattice = quantize_km_fsq(raw.reshape(5, KM_CODE_DIM))
    twice = quantize_km_fsq(lattice)
    assert torch.equal(lattice, twice)


def test_snap_matches_fsq_codes_to_indices_then_indices_to_codes():
    # Comparing against codes_to_indices/indices_to_codes directly, not FSQ.forward: forward's
    # tanh soft-bound exists for training-time gradient flow and is not exactly idempotent at
    # the lattice edges, but codes_to_indices/indices_to_codes is the LOSSLESS index<->code map
    # this snap is meant to match (see the docstring in debug_actions.quantize_km_fsq).
    torch.manual_seed(1)
    fsq = _fsq()
    raw = torch.rand(8, 12, 3) * 2 - 1
    snapped = quantize_km_fsq(raw.reshape(8, KM_CODE_DIM)).reshape(8, 12, 3)

    indices = fsq.codes_to_indices(raw)
    round_trip = fsq.indices_to_codes(indices)
    assert torch.allclose(snapped, round_trip, atol=1e-5)


def test_snap_rejects_wrong_last_dim():
    import pytest
    with pytest.raises(ValueError):
        quantize_km_fsq(torch.zeros(2, 10))
