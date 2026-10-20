"""plaicraft-debug#96 C1: debug_toy + km_ctx14 runs through the raw79 builder; other encodings unchanged."""
import numpy as np
import pytest

from improved_diffusion import video_datasets as vd
from improved_diffusion.debug_dataset import ContinuousDebugDataset
from improved_diffusion.km_tokenizer.ctx14_10hz import DEFAULT_CHECKPOINT_DIR
from improved_diffusion.km_tokenizer.keypress_lookup import KEYPRESS_LOOKUP
from test_debug_actions_km_ctx14 import _make_session_79


def test_debug_toy_builder_only_switches_for_km_ctx14():
    assert vd.debug_toy_builder("debug_toy", "km_ctx14") == "raw79"
    assert [vd.debug_toy_builder("debug_toy", e) for e in ("raw", "km_fsq", "raw_fused")] == ["raw8"] * 3
    assert vd.debug_toy_builder("plaicraft_morgan", "km_ctx14") == "raw8"


def test_valset_kwargs_debug_toy_km_ctx14_uses_session_root_and_raw79():
    kw = vd.plaicraft_valset_kwargs("debug_toy", 0, "", "km_ctx14")
    assert kw == dict(session_root=vd.video_data_paths_dict["debug_toy"], action_builder="raw79")
    assert vd.plaicraft_valset_kwargs("debug_toy", 0, "", "km_fsq") == {}
    assert vd.plaicraft_valset_kwargs("debug_toy") == {}


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_DIR.exists(), reason="staged km_ctx14 checkpoint not present")
def test_10hz_debug_dataset_yields_ctx14_codes_and_causal_gt(tmp_path):
    sd = _make_session_79(tmp_path, fps=10.0, n_ticks=20, key_events=[("87", 1000, 1100)])
    ds = ContinuousDebugDataset(tmp_path, window_length=4, action_encoding="km_ctx14",
                                action_builder=vd.debug_toy_builder("debug_toy", "km_ctx14"))
    codes, mouse = ds._get_action_arrays(sd)
    assert codes.shape == (20, 32) and mouse.shape == (20, 0)
    from improved_diffusion import debug_actions as da
    assert np.nonzero(da.load_ticks_79(sd)[:, KEYPRESS_LOOKUP["87"]])[0].tolist() == [11]
