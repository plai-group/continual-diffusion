"""plaicraft-debug#96 B4: CorpusValidationSet's km_ctx14 action loading -- the generated
region comes from the full-session cache (same as km_fsq), but the context region
[window_start, boundary_tick) is re-encoded in isolation so a future change in the session
can never change the context codes. The tokenizer itself is mocked -- only the splice logic
is exercised here, mirroring test_corpus_validation.py's km_fsq session_root tests."""
import json

import numpy as np

from improved_diffusion import debug_actions
from improved_diffusion.corpus_validation import CorpusValidationSet
from improved_diffusion.km_tokenizer import ctx14_10hz

NAMES = ["walk-forward", "strafe-left", "walk-backward"]
T, N_OBS = 6, 3


def _make_fixture_no_swap(tmp_path):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    rng = np.random.RandomState(0)
    frames = rng.uniform(-1, 1, size=(n, T, 4, 4, 4)).astype(np.float32)
    keypress = rng.rand(n, T, 8).astype(np.float32)
    mouse = rng.randn(n, T, 2).astype(np.float32)
    window_start_ticks = np.array([100, 200, 300], dtype=np.int64)
    boundary_ticks = window_start_ticks + N_OBS
    np.savez(
        vdir / "validation.npz", frames=frames, keypress=keypress, mouse=mouse,
        names=np.array(NAMES), session_ids=np.array([f"sess{i}" for i in range(n)]),
        window_start_ticks=window_start_ticks, boundary_ticks=boundary_ticks,
    )
    manifest = {
        "schema_version": "1", "corpus_dir": str(tmp_path), "tick_ms": 100, "subbin_ms": 10,
        "subbin_rule_version": "1", "T": T, "n_observed": N_OBS, "n_held_out_sessions": 20,
        "built_at": "2026-01-01T00:00:00Z",
        "exercises": [{"index": i, "name": name, "session_id": f"sess{i}"} for i, name in enumerate(NAMES)],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    return vdir


def test_km_ctx14_splices_isolated_context_onto_the_full_session_cache(tmp_path, monkeypatch):
    vdir = _make_fixture_no_swap(tmp_path)
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((400, 8)), np.zeros((400, 2))))
    monkeypatch.setattr(ctx14_10hz, "load_ctx14_tokenizer", lambda checkpoint_dir, device: object())

    full = {f"sess{i}": np.full((400, 32), float(i), dtype=np.float32) for i in range(len(NAMES))}
    monkeypatch.setattr(debug_actions, "load_or_build",
                        lambda d, action_encoding, tokenizer_checkpoint, device, action_builder: (full[d.name], None))

    context_calls = []

    def _fake_context(session_dir, window_start, boundary, tokenizer, device, action_builder):
        context_calls.append((session_dir.name, window_start, boundary))
        return np.full((boundary - window_start, 32), -1.0, dtype=np.float32)

    monkeypatch.setattr(debug_actions, "encode_km_ctx14_context", _fake_context)

    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, action_encoding="km_ctx14",
                             session_root=tmp_path / "corpus", action_builder="raw79")
    km_codes, mouse = vs.load_all_actions()

    assert km_codes.shape == (len(NAMES), T, 32) and mouse.shape[-1] == 0
    assert context_calls == [("sess0", 100, 100 + N_OBS), ("sess1", 200, 200 + N_OBS), ("sess2", 300, 300 + N_OBS)]
    for i in range(len(NAMES)):
        row = km_codes[i].numpy()
        assert np.all(row[:N_OBS] == -1.0)  # isolated context, never the full-session cache
        assert np.all(row[N_OBS:] == float(i))  # generated region: the full-session cache slice


def test_km_ctx14_context_is_unaffected_by_a_later_change_to_the_full_session_cache(tmp_path, monkeypatch):
    """The whole point of B4: changing what the full-session cache returns for the generated
    region must never change the context codes a row carries."""
    vdir = _make_fixture_no_swap(tmp_path)
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((400, 8)), np.zeros((400, 2))))
    monkeypatch.setattr(ctx14_10hz, "load_ctx14_tokenizer", lambda checkpoint_dir, device: object())
    monkeypatch.setattr(debug_actions, "encode_km_ctx14_context",
                        lambda session_dir, window_start, boundary, tokenizer, device, action_builder:
                        np.full((boundary - window_start, 32), 7.0, dtype=np.float32))

    def _load_all(session_value):
        monkeypatch.setattr(debug_actions, "load_or_build",
                            lambda d, action_encoding, tokenizer_checkpoint, device, action_builder:
                            (np.full((400, 32), session_value, dtype=np.float32), None))
        vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, action_encoding="km_ctx14",
                                 session_root=tmp_path / "corpus", action_builder="raw79")
        return vs.load_all_actions()[0][0, :N_OBS].numpy()

    context_a = _load_all(1.0)
    context_b = _load_all(999.0)  # "future" changed -- a session-level re-encode would leak it
    assert np.array_equal(context_a, context_b)
