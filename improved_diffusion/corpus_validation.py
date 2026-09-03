"""Issue #81: validation against plaicraft-debug's held-out-policy exercise package.

plaicraft-debug freezes 13 fixed "exercises" into
<corpus_dir>/validation/validation.npz + manifest.json (a frozen contract owned by
that repo). This class is the read side of that contract -- a duck-typed sibling of
DebugValidationSet (see debug_validation.py) so run_debug_validation can drive
either without a shared base class.
"""
import json
from pathlib import Path

import numpy as np
import torch as th


class CorpusValidationSet:
    """The 13 frozen exercises plus the frames/actions each one needs."""

    def __init__(self, validation_dir, T=None, n_observed=None):
        self.validation_dir = Path(validation_dir)
        npz_path = self.validation_dir / "validation.npz"
        manifest_path = self.validation_dir / "manifest.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"validation package not found: {npz_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"validation manifest not found: {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text())
        if T is not None and self.manifest["T"] != T:
            raise ValueError(f"manifest T={self.manifest['T']} != requested T={T}")
        if n_observed is not None and self.manifest["n_observed"] != n_observed:
            raise ValueError(
                f"manifest n_observed={self.manifest['n_observed']} != requested n_observed={n_observed}"
            )
        self.T = self.manifest["T"]
        self.n_observed = self.manifest["n_observed"]
        self.n_generated = self.T - self.n_observed

        npz = np.load(npz_path, allow_pickle=False)
        self._frames = np.asarray(npz["frames"], dtype=np.float32)
        self._keypress = np.asarray(npz["keypress"], dtype=np.float32)
        self._mouse = np.asarray(npz["mouse"], dtype=np.float32)
        names = [str(n) for n in npz["names"]]
        session_ids = [str(s) for s in npz["session_ids"]]
        window_start_ticks = npz["window_start_ticks"]
        boundary_ticks = npz["boundary_ticks"]

        by_name = {e["name"]: e for e in self.manifest["exercises"]}
        self.rows = []
        for i, name in enumerate(names):
            ex = by_name[name]
            self.rows.append(dict(
                num=int(ex["index"]),
                name=name,
                # prompt/test_type: run_debug_validation reads these directly (its
                # per-row loop is shared with DebugValidationSet, which is prose-prompt
                # driven -- an exercise's name stands in for both here).
                prompt=name,
                test_type="exercise",
                session_id=session_ids[i],
                swap_kind=ex["swap_kind"],
                swap_dim=ex.get("swap_dim"),
                swap_counterpart_dim=ex.get("swap_counterpart_dim"),
                window_start=int(window_start_ticks[i]),
                boundary_tick=int(boundary_ticks[i]),
            ))

    def slug(self, row):
        return f"{row['num']:02d}_{row['name']}"

    def load_all(self):
        """(13, T, 3, H, W) float32 in [-1, 1], straight from the npz."""
        return th.from_numpy(self._frames.copy())

    def load_all_actions(self):
        """(keypress: (13, T, 8), mouse: (13, T, 2)) float32, straight from the npz.

        Always the raw tick-resolution arrays -- the frozen contract carries no
        other encoding, so this doubles as load_all_actions_raw for duck typing
        with run_debug_validation (overlays/interventions always want raw anyway).
        """
        return th.from_numpy(self._keypress.copy()), th.from_numpy(self._mouse.copy())

    load_all_actions_raw = load_all_actions
