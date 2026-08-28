import os
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from improved_diffusion.debug_actions import load_or_build as load_or_build_actions


class ContinuousDebugDataset(Dataset):
    """
    Sliding-window dataset over the plaicraft-debug HDF5 corpus.

    Layout: <dataset_path>/<session_id>/encoded_video_hdf5/<session_id>_encoded_video.hdf5
    Each file has dataset key "frames", shape (N, 3, H, W), float32, already in [-1, 1].

    Sessions are discovered by globbing (no global_database.db dependency). The last
    (up to) 20 sessions by sorted session id are held out as the test split.

    An h5py.File handle is never shared across a fork: handles are cached lazily,
    keyed by (pid, path), and dropped whenever os.getpid() changes.
    """

    N_TEST_SESSIONS = 20

    def __init__(self, dataset_path, window_length=20, frame_range=(0, None)):
        self.dataset_path = Path(dataset_path)
        self.window_length = self.T = window_length
        self.is_test = False
        self.original_frame_range = frame_range

        self._h5_handles = {}
        self._keypress_arrays = {}
        self._mouse_arrays = {}
        self._handle_pid = None

        self._validate_parameters()
        self._initialize_file_index_mapping()

    def _validate_parameters(self):
        assert isinstance(self.window_length, int) and self.window_length > 0, \
            f"window_length must be a positive integer, but got {self.window_length}."

    # ------------------------------------------------------------------ #
    # session discovery / index mapping
    # ------------------------------------------------------------------ #

    def _discover_sessions(self):
        files = sorted(
            self.dataset_path.glob("*/encoded_video_hdf5/*.hdf5"),
            key=lambda p: p.parent.parent.name,
        )
        if not files:
            raise ValueError(f"No hdf5 files found under {self.dataset_path}")

        n_test = min(self.N_TEST_SESSIONS, len(files))
        if n_test >= len(files):
            # Fine for a single-session local fixture, but on a real corpus it
            # means test == train, which silently invalidates FID/JEDi. Be loud:
            # a partially-failed generation run is the likely cause.
            warnings.warn(
                f"debug_toy: only {len(files)} session(s) under {self.dataset_path}; "
                f"cannot hold out {self.N_TEST_SESSIONS}. Test split is IDENTICAL to "
                f"train — held-out metrics are meaningless. Expected 200 sessions.",
                RuntimeWarning,
                stacklevel=2,
            )
            train_files, test_files = files, files
        else:
            train_files, test_files = files[:-n_test], files[-n_test:]
        return test_files if self.is_test else train_files

    def _initialize_file_index_mapping(self):
        self._close_handles()
        session_files = self._discover_sessions()

        self.file_boundaries = []  # list of (frame_start, frame_end, path)
        total_frames = 0
        frame_shape = None  # (C, H, W), must agree across every session
        for path in session_files:
            with h5py.File(path, "r") as f:
                n, *chw = f["frames"].shape
            chw = tuple(chw)
            if frame_shape is None:
                frame_shape = chw
            elif chw != frame_shape:
                # Mixing resolutions yields batches VDT cannot patchify, and the
                # failure surfaces far away inside PatchEmbed. Fail here instead.
                raise ValueError(
                    f"debug_toy: inconsistent frame shapes under {self.dataset_path}. "
                    f"{session_files[0].parent.parent.name} is {frame_shape} but "
                    f"{path.parent.parent.name} is {chw}. Point dataset_path at a "
                    f"single-resolution corpus root."
                )
            self.file_boundaries.append((total_frames, total_frames + n, path))
            total_frames += n
        self.total_frames = total_frames
        self.frame_shape = frame_shape

        self.frame_range = self.original_frame_range
        if self.frame_range[1] is None or self.frame_range[1] > total_frames:
            self.frame_range = (self.frame_range[0], total_frames)

        self.window_starts = self._build_window_starts(step=1)

    def _build_window_starts(self, step):
        """Valid window start positions, never straddling a session boundary."""
        starts = []
        for (fs, fe, _path) in self.file_boundaries:
            lo = max(fs, self.frame_range[0])
            hi = min(fe, self.frame_range[1])
            s = lo
            while s + self.T <= hi:
                starts.append(s)
                s += step
        return starts

    def _get_start_frame_index(self, idx):
        return self.window_starts[idx]

    def __len__(self):
        return len(self.window_starts)

    # ------------------------------------------------------------------ #
    # data access
    # ------------------------------------------------------------------ #

    def _get_h5_handle(self, path):
        pid = os.getpid()
        if self._handle_pid != pid:
            # New process (e.g. a forked DataLoader worker): never reuse a
            # handle that may have been inherited from the parent process.
            self._h5_handles = {}
            self._keypress_arrays = {}
            self._mouse_arrays = {}
            self._handle_pid = pid
        path = str(path)
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def _get_action_arrays(self, session_dir):
        pid = os.getpid()
        if self._handle_pid != pid:
            self._h5_handles = {}
            self._keypress_arrays = {}
            self._mouse_arrays = {}
            self._handle_pid = pid
        session_dir = str(session_dir)
        if session_dir not in self._keypress_arrays:
            keypress, mouse = load_or_build_actions(session_dir)
            self._keypress_arrays[session_dir] = keypress
            self._mouse_arrays[session_dir] = mouse
        return self._keypress_arrays[session_dir], self._mouse_arrays[session_dir]

    def _close_handles(self):
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_handles = {}
        self._keypress_arrays = {}
        self._mouse_arrays = {}
        self._handle_pid = None

    def __getitem__(self, idx):
        start_frame = self._get_start_frame_index(idx)
        end_frame = start_frame + self.T

        for (fs, fe, path) in self.file_boundaries:
            if fs <= start_frame < fe:
                break
        else:
            raise IndexError(f"Index {idx} (start frame {start_frame}) not found in any session.")
        assert end_frame <= fe, f"Window at index {idx} straddles a session boundary."

        local_start = start_frame - fs
        handle = self._get_h5_handle(path)
        frames = torch.from_numpy(handle["frames"][local_start:local_start + self.T]).float()

        if frames.shape[0] != self.T:
            raise IndexError(f"Incomplete window at index {idx}, should have been discarded.")

        session_dir = path.parent.parent
        keypress_array, mouse_array = self._get_action_arrays(session_dir)
        keypress = torch.from_numpy(
            np.asarray(keypress_array[local_start:local_start + self.T])
        ).float()
        mouse = torch.from_numpy(
            np.asarray(mouse_array[local_start:local_start + self.T])
        ).float()

        absolute_index_map = torch.arange(start_frame, end_frame, dtype=torch.int64)
        return frames, absolute_index_map, keypress, mouse

    def set_train(self):
        self.is_test = False
        self._initialize_file_index_mapping()
        print("setting train mode")

    def set_test(self):
        self.is_test = True
        self._initialize_file_index_mapping()
        print("setting test mode")

    def __del__(self):
        self._close_handles()


class ChunkedDebugDataset(ContinuousDebugDataset):
    def _initialize_file_index_mapping(self):
        super()._initialize_file_index_mapping()
        self.chunked_starts = self._build_window_starts(step=self.T)

    def _get_start_frame_index(self, idx):
        return self.chunked_starts[idx]

    def __len__(self):
        return len(self.chunked_starts)


class SpacedDebugDataset(ContinuousDebugDataset):
    def __init__(self, n_data, *args, **kwargs):
        self.n_data = n_data
        super().__init__(*args, **kwargs)

    def _initialize_file_index_mapping(self):
        super()._initialize_file_index_mapping()
        chunk_starts = self._build_window_starts(step=self.T)
        assert len(chunk_starts) > 0, "No non-overlapping windows available to space over."
        spacing = max(1, len(chunk_starts) // self.n_data)
        self.spaced_starts = chunk_starts[::spacing][:self.n_data]
        while len(self.spaced_starts) < self.n_data:
            self.spaced_starts.append(chunk_starts[-1])
        print(f"Total windows: {len(chunk_starts)}, Spacing: {spacing}, # Data: {self.n_data}")

    def _get_start_frame_index(self, idx):
        return self.spaced_starts[idx]

    def __len__(self):
        return self.n_data
