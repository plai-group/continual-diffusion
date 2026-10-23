"""plaicraft-debug#100: uint8 pixel corpora map to [-1,1] and set the model grid from the corpus."""
import h5py
import numpy as np
import torch

from improved_diffusion import video_datasets as vd
from improved_diffusion.plaicraft_morgan_dataset import PlaicraftDataset
from test_plaicraft_morgan_dataset import _make_global_db, _make_session, _make_validation_db


def _corpus(tmp_path, dtype, shape, fill):
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    sd = _make_session(root, "trainA", n_ticks=4)
    with h5py.File(sd / "encoded_video_hdf5" / "trainA.hdf5", "w") as f:
        f.create_dataset("frames", data=np.full((4, *shape), fill, dtype=dtype))
        f["frames"].attrs["fps"] = 12.5  # default raw encoding wants 12.5
    _make_global_db(tmp_path / "g.db", [("trainA", "Morgan")])
    _make_validation_db(tmp_path / "v.db", [])
    return root, dict(global_database_path=tmp_path / "g.db", validation_db_path=tmp_path / "v.db")


def test_uint8_frames_map_to_unit_range(tmp_path):
    root, kw = _corpus(tmp_path, np.uint8, (3, 96, 160), 0)
    with h5py.File(root / "trainA/encoded_video_hdf5/trainA.hdf5", "r+") as f:
        f["frames"][1] = 255
        f["frames"][2] = 128
    ds = PlaicraftDataset(root, window_length=3, num_classes=0, **kw)
    frames = ds[0][0]
    assert frames.shape == (3, 3, 96, 160) and frames.dtype == torch.float32
    assert frames[0].min() == -1 and frames[1].max() == 1 and torch.allclose(frames[2], torch.tensor(128 / 127.5 - 1), atol=1e-6)


def test_float_latent_corpus_unchanged(tmp_path):
    root, kw = _corpus(tmp_path, np.float16, (4, 24, 40), 0.5)
    ds = PlaicraftDataset(root, window_length=3, num_classes=0, **kw)
    assert torch.equal(ds[0][0], torch.full((3, 4, 24, 40), 0.5))


def test_full_image_size_reads_corpus_shape(tmp_path, monkeypatch):
    for name, shape, want in (("pix", (3, 96, 160), (96, 160)), ("lat", (4, 24, 40), (24, 40))):
        root, _ = _corpus(tmp_path / name, np.uint8, shape, 0)
        monkeypatch.setitem(vd.video_data_paths_dict, "plaicraft_morgan", str(root))
        assert vd.full_image_size("plaicraft_morgan") == want
    assert vd.full_image_size("debug_toy") == (24, 40)


def test_full_image_size_falls_back_without_corpus(tmp_path, monkeypatch):
    monkeypatch.setitem(vd.video_data_paths_dict, "plaicraft_morgan", str(tmp_path / "missing"))
    assert vd.full_image_size("plaicraft_morgan") == (24, 40)
