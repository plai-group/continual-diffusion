"""ticks_per_latent=k in VDT and the debug dataset: plaicraft-debug#99."""

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion.debug_dataset import ContinuousDebugDataset
from improved_diffusion.vdt import VDT_S_2
from test_player_dataset import _session

KW = dict(input_size=16, patch_size=4, in_channels=3, num_frames=3, learn_sigma=False)


def _randomize(m):
    for mod in (m.final_layer, m.action_head):
        torch.nn.init.normal_(mod.linear.weight, std=0.02)
    for b in m.blocks:
        torch.nn.init.normal_(b.adaLN_modulation[-1].weight, std=0.02)


@pytest.mark.parametrize("pos", ["sincos", "rope3d"])
@pytest.mark.parametrize("attention", ["factorized", "full"])
def test_k4_shapes_and_per_tick_tokens(pos, attention):
    torch.manual_seed(0)
    k, D, B, T = 4, 6, 2, 3
    m = VDT_S_2(action_dim=D, generate_actions=True, ticks_per_latent=k, pos_encoding=pos, attention=attention, **KW).eval()
    _randomize(m)
    assert m.action_dim == k * D and m.tick_action_dim == D
    x, a = torch.randn(B, T, 3, 16, 16), torch.randn(B, T, k * D)
    v, (act, _) = m(x, timesteps=torch.tensor([5, 50]), actions=a)
    assert v.shape == x.shape and act.shape == a.shape and torch.isfinite(act).all()
    # tick j of frame t only enters through its own token: changing it perturbs the video
    a2 = a.clone()
    a2[:, 1, 2 * D:3 * D] += 1.0
    assert not torch.allclose(m(x, timesteps=torch.tensor([5, 50]), actions=a2)[0], v)


def test_k1_identical_to_pre_change_module():
    import importlib.util
    import os
    path = "/ubc/cs/research/plai-scratch/ctardy/projects/continual-diffusion/run-scripts/wan_99/vdt_orig.py"
    if not os.path.exists(path):
        pytest.skip("pre-#99 vdt.py snapshot not available")
    spec = importlib.util.spec_from_file_location("vdt_orig", path)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    torch.manual_seed(0)
    kw = dict(action_dim=5, generate_actions=True, **KW)
    old, new = orig.VDT_S_2(**kw), VDT_S_2(ticks_per_latent=1, **kw)
    sd = old.state_dict()
    for key in sd:
        if "final_layer" in key or "action_head" in key or "adaLN" in key:
            sd[key] = torch.randn_like(sd[key]) * 0.02
    old.load_state_dict(sd)
    new.load_state_dict(sd)
    x, t, a = torch.randn(2, 3, 3, 16, 16), torch.tensor([10, 200]), torch.randn(2, 3, 5)
    (v0, (a0, _)), (v1, (a1, _)) = old(x, timesteps=t, actions=a), new(x, timesteps=t, actions=a)
    assert torch.equal(v0, v1) and torch.equal(a0, a1)


def test_k_gt1_rejects_mouse():
    with pytest.raises(AssertionError):
        VDT_S_2(action_dim=4, mouse_dim=2, generate_actions=True, generate_mouse=True, ticks_per_latent=4, **KW)


def _dataset(tmp_path, k, off, n_lat=10, T=3, D=2):
    d = _session(tmp_path, "s00", n_ticks=n_lat, fps=10.0)
    with h5py.File(d / "encoded_video_hdf5" / "s00_encoded_video.hdf5", "r+") as f:
        if (k, off) != (1, 0):
            f["frames"].attrs["ticks_per_latent"], f["frames"].attrs["tick_offset"] = k, off
    ds = ContinuousDebugDataset(tmp_path, window_length=T, action_encoding="km_ctx14")
    n_ticks = off + k * n_lat
    kp = np.arange(n_ticks * D, dtype=np.float32).reshape(n_ticks, D)
    mo = -kp[:, :1]
    ds._get_action_arrays = lambda _d: (kp, mo)
    return ds, kp, mo


def test_k4_window_slices_ticks_and_round_trips(tmp_path):
    k, off, T, D = 4, 1, 3, 2
    ds, kp, mo = _dataset(tmp_path, k, off, T=T, D=D)
    assert (ds.ticks_per_latent, ds.tick_offset) == (k, off)
    frames, _idx, keypress, mouse, _p = ds[2]
    assert frames.shape[0] == T and keypress.shape == (T, k * D) and mouse.shape == (T, k)
    expect = kp[off + k * 2:off + k * (2 + T)]
    assert torch.equal(keypress, torch.from_numpy(expect).reshape(T, k * D))
    # (T, k*D) -> (T*k, D) recovers the per-tick stream
    assert torch.equal(keypress.reshape(T * k, D), torch.from_numpy(expect))


def test_default_attrs_keep_per_frame_slicing(tmp_path):
    ds, kp, mo = _dataset(tmp_path, 1, 0, n_lat=10, T=3)
    _f, _i, keypress, mouse, _p = ds[4]
    assert torch.equal(keypress, torch.from_numpy(kp[4:7])) and torch.equal(mouse, torch.from_numpy(mo[4:7]))
