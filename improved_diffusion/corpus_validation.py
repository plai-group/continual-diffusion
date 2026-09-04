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

from . import debug_actions


class CorpusValidationSet:
    """The 13 frozen exercises plus the frames/actions each one needs."""

    def __init__(self, validation_dir, T=None, n_observed=None, action_encoding="raw",
                 tokenizer_checkpoint=None, device=None):
        self.validation_dir = Path(validation_dir)
        self.action_encoding = action_encoding
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.device = device
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

        self._validate_shapes(names, session_ids, window_start_ticks, boundary_ticks)

        by_name = {e["name"]: e for e in self.manifest["exercises"]}
        self.rows = []
        for i, name in enumerate(names):
            ex = by_name[name]
            window_start = int(window_start_ticks[i])
            boundary_tick = int(boundary_ticks[i])
            offset = boundary_tick - window_start
            # Load-bearing contract (plaicraft-debug#81): the swap intervention hardcodes
            # boundary_idx=n_observed (debug_validation.py), which is only correct if the
            # producer put the boundary this many ticks after window_start. A silent
            # mismatch here means every metric still computes but the swap test (AC2)
            # is intervening on the wrong tick with no error -- so this must raise, not warn.
            if offset != self.n_observed:
                raise ValueError(
                    f"exercise {name!r}: boundary_tick - window_start = {offset}, "
                    f"expected n_observed={self.n_observed} "
                    f"(window_start={window_start}, boundary_tick={boundary_tick})"
                )
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
                window_start=window_start,
                boundary_tick=boundary_tick,
            ))

    def _validate_shapes(self, names, session_ids, window_start_ticks, boundary_ticks):
        """A producer that emits a mismatched package must fail loudly here, not
        misbehave later -- the manifest-vs-constructor checks above don't touch the
        npz arrays themselves."""
        n_exercises = len(self.manifest["exercises"])
        row_counts = {
            "frames": self._frames.shape[0], "keypress": self._keypress.shape[0],
            "mouse": self._mouse.shape[0], "names": len(names), "session_ids": len(session_ids),
            "window_start_ticks": window_start_ticks.shape[0], "boundary_ticks": boundary_ticks.shape[0],
        }
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"row-count mismatch across validation.npz arrays: {row_counts}")
        n_rows = next(iter(row_counts.values()))
        if n_rows != n_exercises:
            raise ValueError(
                f"validation.npz has {n_rows} rows but manifest lists {n_exercises} exercises"
            )
        for name, arr in (("frames", self._frames), ("keypress", self._keypress), ("mouse", self._mouse)):
            if arr.shape[1] != self.T:
                raise ValueError(
                    f"{name}.shape={arr.shape} has T={arr.shape[1]}, expected manifest T={self.T}"
                )
        if self._keypress.shape[-1] != 8:
            raise ValueError(f"keypress.shape={self._keypress.shape} must have 8 columns (raw keypress dim)")
        if self._mouse.shape[-1] != 2:
            raise ValueError(f"mouse.shape={self._mouse.shape} must have 2 columns (raw mouse dim)")

    def slug(self, row):
        return f"{row['num']:02d}_{row['name']}"

    def load_all(self):
        """(13, T, 3, H, W) float32 in [-1, 1], straight from the npz."""
        return th.from_numpy(self._frames.copy())

    def load_all_actions(self):
        """(keypress, mouse) float32, in the model's NATIVE conditioning encoding.

        raw -> (13, T, 8) + (13, T, 2), straight from the npz.
        km_fsq -> (13, T, 36) codes + (13, T, 0) empty mouse (folded into the codes,
        same convention as debug_actions.load_or_build). Built lazily on first use via
        scripts/build_debug_validation_km_codes.build_km_codes and cached in
        validation_dir/km_codes.npz; a stale or missing cache is rebuilt automatically.
        raw_fused -> (13, T, 10) [keypress, symlog(mouse)] + (13, T, 0) empty mouse; the
        npz's mouse array is raw pixels, so this symlog-compresses it, no tokenizer involved.
        """
        if self.action_encoding == "raw":
            return self.load_all_actions_raw()
        if self.action_encoding == "raw_fused":
            fused = np.concatenate([self._keypress, debug_actions._symlog(self._mouse)], axis=-1).astype(np.float32)
            empty_mouse = np.zeros((*fused.shape[:-1], 0), dtype=np.float32)
            return th.from_numpy(fused), th.from_numpy(empty_mouse)
        if self.action_encoding == "km_fsq":
            from scripts.build_debug_validation_km_codes import build_km_codes
            from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT

            checkpoint = self.tokenizer_checkpoint or DEFAULT_CHECKPOINT
            codes_path = build_km_codes(self.validation_dir, checkpoint, self.device)
            km_codes = np.asarray(np.load(codes_path)["km_codes"], dtype=np.float32)
            empty_mouse = np.zeros((km_codes.shape[0], km_codes.shape[1], 0), dtype=np.float32)
            return th.from_numpy(km_codes), th.from_numpy(empty_mouse)
        raise ValueError(f"unknown action_encoding {self.action_encoding!r}, expected 'raw', 'km_fsq', or 'raw_fused'")

    def load_all_actions_raw(self):
        """(keypress: (13, T, 8), mouse: (13, T, 2)) float32, straight from the npz.

        Always the raw tick-resolution arrays, independent of action_encoding --
        overlays and interventions read this, never the model's native conditioning
        tensor (mirrors DebugValidationSet.load_all_actions_raw, plaicraft-debug#80/81)."""
        return th.from_numpy(self._keypress.copy()), th.from_numpy(self._mouse.copy())
