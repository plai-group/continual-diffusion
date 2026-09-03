"""Issue #81: scripts/build_debug_validation_km_codes.py, against a fabricated
validation.npz fixture and the real (CPU) km tokenizer checkpoint -- same pattern
as tests/test_debug_actions_km_cache.py, so no mocking of the tokenizer itself."""
import json
import sys
from pathlib import Path

import imageio
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_debug_validation_km_codes import build_km_codes, render_gt_overlays  # noqa: E402

from improved_diffusion import debug_actions as da
from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT, _sha256

pytestmark = pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="km tokenizer checkpoint not staged")

NAMES = ["walk-forward", "strafe-left", "walk-backward"]
T, N_OBS, H, W = 8, 4, 6, 8


def _make_fixture(tmp_path):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    rng = np.random.RandomState(0)
    frames = rng.uniform(-1, 1, size=(n, T, 3, H, W)).astype(np.float32)
    keypress = (rng.rand(n, T, 8) > 0.7).astype(np.float32)
    mouse = (rng.randn(n, T, 2) * 20).astype(np.float32)
    window_start_ticks = np.array([1000, 2000, 3000], dtype=np.int64)
    boundary_ticks = window_start_ticks + N_OBS
    np.savez(
        vdir / "validation.npz", frames=frames, keypress=keypress, mouse=mouse,
        names=np.array(NAMES), session_ids=np.array([f"sess{i}" for i in range(n)]),
        window_start_ticks=window_start_ticks, boundary_ticks=boundary_ticks,
    )
    manifest = {
        "schema_version": "1", "corpus_dir": str(tmp_path), "tick_ms": 80, "subbin_ms": 10,
        "subbin_rule_version": "1", "T": T, "n_observed": N_OBS, "n_held_out_sessions": 20,
        "built_at": "2026-01-01T00:00:00Z",
        "exercises": [
            {"index": 0, "name": "walk-forward", "session_id": "sess0",
             "swap_kind": "keypress", "swap_dim": 0, "swap_counterpart_dim": 2},
            {"index": 1, "name": "strafe-left", "session_id": "sess1",
             "swap_kind": "keypress", "swap_dim": 1, "swap_counterpart_dim": 3},
            {"index": 2, "name": "walk-backward", "session_id": "sess2", "swap_kind": "mouse_dy"},
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    return vdir


def test_build_km_codes_shape_and_sidecar(tmp_path):
    vdir = _make_fixture(tmp_path)
    codes_path = build_km_codes(vdir, DEFAULT_CHECKPOINT, device="cpu")
    codes = np.load(codes_path)["km_codes"]
    assert codes.shape == (len(NAMES), T, da.KM_CODE_DIM)
    sidecar_path = vdir / "km_codes_manifest.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["tokenizer_sha256"] == _sha256(DEFAULT_CHECKPOINT)
    assert sidecar["subbin_rule_version"] == da.SUBBIN_RULE_VERSION
    assert "built_at" in sidecar


def test_build_km_codes_skips_rebuild_when_sidecar_matches(tmp_path, monkeypatch):
    vdir = _make_fixture(tmp_path)
    build_km_codes(vdir, DEFAULT_CHECKPOINT, device="cpu")

    import build_debug_validation_km_codes as mod

    def _boom(*a, **k):
        raise AssertionError("should not re-encode when sidecar already matches")
    monkeypatch.setattr(mod, "_encode_km_actions", _boom)
    build_km_codes(vdir, DEFAULT_CHECKPOINT, device="cpu")  # must not raise


def test_build_km_codes_force_rebuilds(tmp_path):
    vdir = _make_fixture(tmp_path)
    build_km_codes(vdir, DEFAULT_CHECKPOINT, device="cpu")
    codes_path = vdir / "km_codes.npz"
    mtime_before = codes_path.stat().st_mtime_ns
    build_km_codes(vdir, DEFAULT_CHECKPOINT, device="cpu", force=True)
    assert codes_path.stat().st_mtime_ns >= mtime_before


def test_render_gt_overlays_writes_one_mp4_per_exercise(tmp_path, monkeypatch):
    vdir = _make_fixture(tmp_path)

    class _NullWriter:
        def __init__(self):
            self.frames = []
        def append_data(self, frame):
            self.frames.append(frame)
        def close(self):
            pass

    writers = []

    def _fake_get_writer(*a, **k):
        w = _NullWriter()
        writers.append(w)
        return w

    monkeypatch.setattr(imageio, "get_writer", _fake_get_writer)
    out_dir = render_gt_overlays(vdir)
    assert out_dir == vdir / "decoded"
    assert len(writers) == len(NAMES)
    for w in writers:
        assert len(w.frames) == T
        # (top action bar + content) stacked, at the DECODE_FINAL_FRAME_SIZE (W, H) upscale.
        assert w.frames[0].ndim == 3 and w.frames[0].shape[2] == 3
