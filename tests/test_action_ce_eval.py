"""issue #76 slice 3: run_action_ce_eval end-to-end on a tiny synthetic corpus + VDT model."""

import pickle
import sqlite3
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from improved_diffusion.debug_actions import ENCODED_KEYPRESS_DIM, keypress_ce_baserate
from improved_diffusion.debug_validation import DebugCorpusWindowSet, run_action_ce_eval
from improved_diffusion.script_util import create_vdt_model_and_diffusion


def _make_session(session_dir, n_frames, C, H, W):
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    hdf5_dir = session_dir / "encoded_video_hdf5"
    hdf5_dir.mkdir()
    frames = np.random.RandomState(0).randn(n_frames, C, H, W).astype(np.float32)
    with h5py.File(hdf5_dir / f"{session_dir.name}_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=frames)

    db_path = session_dir / f"{session_dir.name}.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX REAL, mouseDY REAL)")
    # "w" (key_id 87) held during frame 3's window, so the ground truth isn't degenerate.
    con.execute("INSERT INTO keyboard VALUES ('87', 300, 400)")
    con.execute(
        "CREATE TABLE key_press_encodings "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "start_timestamp INTEGER, end_timestamp INTEGER, encoding BLOB)"
    )
    rng = np.random.RandomState(1)
    for k in range(n_frames):
        start = k * 100
        con.execute(
            "INSERT INTO key_press_encodings (start_timestamp, end_timestamp, encoding) VALUES (?, ?, ?)",
            (start, start + 100, pickle.dumps(rng.randn(16, 5).astype(np.float32))),
        )
    con.commit()
    con.close()


def test_run_action_ce_eval_end_to_end():
    torch.manual_seed(0)
    B_unused, T, C, H, W = None, 8, 3, 32, 32
    n_observed = 4

    model, diffusion = create_vdt_model_and_diffusion(
        model_name="VDT-S", patch_size=4, input_size=(H, W), in_channels=C,
        num_frames=T, learn_sigma=False, sigma_small=False, diffusion_steps=100,
        diffusion_space_kwargs=dict(diffusion_space="pixel", pre_encoded=False),
        noise_schedule="linear", timestep_respacing="", use_kl=False,
        predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True,
        use_checkpoint=False, use_edm_scaling=False,
        action_dim=ENCODED_KEYPRESS_DIM, action_dropout_prob=0.0, generate_actions=True,
    )
    model.eval()

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_session(root / "sess0", T, C, H, W)
        windowset = DebugCorpusWindowSet(root, T=T, n_observed=n_observed, n_windows=10, seed=0)

        out = run_action_ce_eval(model, diffusion, windowset, device="cpu", chunk_size=4)

        # Baserate is measured from the same scored rows, independent of the model -- cross-check
        # by recomputing it directly from the windowset's ground truth.
        first_gen = n_observed + 1
        y_true = windowset.load_all_keypress_raw()[:, first_gen:]

    for key in ("val/action_ce/keys_sample", "val/action_ce/keys_mean", "val/action_ce/keys_baserate"):
        assert key in out
        assert np.isfinite(out[key]), f"{key} is not finite: {out[key]}"
    assert out["val/action_ce/keys_sample"] >= 0.0
    assert out["val/action_ce/keys_mean"] >= 0.0

    expected_baserate = float(keypress_ce_baserate(y_true))
    assert abs(out["val/action_ce/keys_baserate"] - expected_baserate) < 1e-4
