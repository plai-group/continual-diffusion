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
                 tokenizer_checkpoint=None, device=None, session_root=None, action_builder="raw8",
                 player_index_fn=None, ticks_per_latent=1):
        self.validation_dir = Path(validation_dir)
        self.session_root = Path(session_root) if session_root else None  # set: actions come from the sessions' own caches
        self.action_builder = action_builder
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
        # k action ticks per latent frame (Wan: 4): npz frames are (N, T) latents, actions are (N, k*T) ticks (plaicraft-debug#99).
        self.k = ticks_per_latent

        npz = np.load(npz_path, allow_pickle=False)
        self._frames = np.asarray(npz["frames"], dtype=np.float32)
        self._keypress = np.asarray(npz["keypress"], dtype=np.float32) if "keypress" in npz.files else None
        self._mouse = np.asarray(npz["mouse"], dtype=np.float32) if "mouse" in npz.files else None
        self._wan_prefix = np.asarray(npz["wan_prefix"], dtype=np.float32) if "wan_prefix" in npz.files else None  # (N, 48, h, w) image latent before each window
        names = [str(n) for n in npz["names"]]
        session_ids = [str(s) for s in npz["session_ids"]]
        window_start_ticks = npz["window_start_ticks"]
        boundary_ticks = npz["boundary_ticks"]
        # Each row's own player label (the one it trained under); None leaves the model's default y.
        self.player_labels = (th.tensor([player_index_fn(self.session_root / sid) for sid in session_ids])
                              if player_index_fn is not None else None)
        if self.session_root is not None:
            self._keypress, self._mouse = map(np.stack, zip(*(
                (np.asarray(k[w:w + self.k * self.T], dtype=np.float32), np.asarray(m[w:w + self.k * self.T], dtype=np.float32))
                for sid, w in zip(session_ids, window_start_ticks)
                for k, m in [debug_actions.load_or_build_raw(self.session_root / sid, action_builder=self.action_builder)])))

        self._validate_shapes(names, session_ids, window_start_ticks, boundary_ticks)

        by_name = {e["name"]: e for e in self.manifest["exercises"]}
        self.rows = []
        for i, name in enumerate(names):
            ex = by_name[name]
            window_start = int(window_start_ticks[i])
            boundary_tick = int(boundary_ticks[i])
            offset = boundary_tick - window_start  # in ticks
            # Raises, not warns: swap hardcodes boundary_idx=n_observed; mismatch would intervene on the wrong tick silently (plaicraft-debug#81).
            if offset != self.k * self.n_observed:
                raise ValueError(
                    f"exercise {name!r}: boundary_tick - window_start = {offset}, "
                    f"expected ticks_per_latent*n_observed={self.k * self.n_observed} "
                    f"(window_start={window_start}, boundary_tick={boundary_tick})"
                )
            self.rows.append(dict(
                num=int(ex["index"]),
                name=name,
                # prompt/test_type: run_debug_validation's per-row loop is shared with the prose-prompt DebugValidationSet, so the exercise name stands in for both.
                prompt=name,
                test_type="exercise",
                session_id=session_ids[i],
                # None (plaicraft-debug#93): a manifest with no swap package at all, e.g.
                # plaicraft_morgan's -- run_debug_validation skips the swap test per row.
                swap_kind=ex.get("swap_kind"),
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
        for name, arr, t in (("frames", self._frames, self.T), ("keypress", self._keypress, self.k * self.T),
                             ("mouse", self._mouse, self.k * self.T)):
            if arr.shape[1] != t:
                raise ValueError(
                    f"{name}.shape={arr.shape} has T={arr.shape[1]}, expected {t} (manifest T={self.T}, ticks_per_latent={self.k})"
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

    def load_all_wan_prefix(self):
        """(N, 1, 48, h, w) float32 image latent of the tick before each window, or None when the package has none."""
        return None if self._wan_prefix is None else th.from_numpy(self._wan_prefix.copy()).unsqueeze(1)

    def load_all_actions(self):
        """(keypress, mouse) float32, in the model's NATIVE conditioning encoding.

        raw -> (13, T, 8) + (13, T, 2), straight from the npz.
        km_fsq -> (13, T, 36) codes + (13, T, 0) empty mouse (folded into the codes,
        same convention as debug_actions.load_or_build). Built lazily on first use via
        scripts/build_debug_validation_km_codes.build_km_codes and cached in
        validation_dir/km_codes.npz; a stale or missing cache is rebuilt automatically.
        raw_fused -> (13, T, 10) [keypress, symlog(mouse)] + (13, T, 0) empty mouse; the
        npz's mouse array is raw pixels, so this symlog-compresses it, no tokenizer involved.
        km_ctx14 -> (N, T, 32) codes + (N, T, 0) empty mouse, session_root only: the generated
        region is a slice of the full-session cache (same as km_fsq), but the context region
        [window_start, boundary_tick) is re-encoded in isolation (plaicraft-debug#96) -- ctx14's
        encoder is bidirectional, so slicing it out of the full-session cache would let context
        ticks near the boundary see the generated future through their +-14 tick reach.
        """
        if self.action_encoding == "raw":
            return self.load_all_actions_raw()
        if self.action_encoding == "raw_fused":
            fused = np.concatenate([self._keypress, debug_actions._symlog(self._mouse)], axis=-1).astype(np.float32)
            empty_mouse = np.zeros((*fused.shape[:-1], 0), dtype=np.float32)
            return th.from_numpy(fused), th.from_numpy(empty_mouse)
        assert self.k == 1 or self.action_encoding == "km_ctx14", "ticks_per_latent > 1 is only wired for km_ctx14"
        if self.action_encoding == "km_fsq" and self.session_root is not None:
            km_codes = np.stack([np.asarray(debug_actions.load_or_build(
                self.session_root / r["session_id"], action_encoding="km_fsq", tokenizer_checkpoint=self.tokenizer_checkpoint,
                device=self.device, action_builder=self.action_builder)[0][r["window_start"]:r["window_start"] + self.T],
                dtype=np.float32) for r in self.rows])
            return th.from_numpy(km_codes), th.from_numpy(np.zeros((*km_codes.shape[:2], 0), dtype=np.float32))
        if self.action_encoding == "km_fsq":
            from scripts.build_debug_validation_km_codes import build_km_codes
            from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT

            checkpoint = self.tokenizer_checkpoint or DEFAULT_CHECKPOINT
            codes_path = build_km_codes(self.validation_dir, checkpoint, self.device)
            km_codes = np.asarray(np.load(codes_path)["km_codes"], dtype=np.float32)
            empty_mouse = np.zeros((km_codes.shape[0], km_codes.shape[1], 0), dtype=np.float32)
            return th.from_numpy(km_codes), th.from_numpy(empty_mouse)
        if self.action_encoding == "km_ctx14" and self.session_root is not None:
            from .km_tokenizer.ctx14_10hz import DEFAULT_CHECKPOINT_DIR, load_ctx14_tokenizer

            checkpoint = self.tokenizer_checkpoint or DEFAULT_CHECKPOINT_DIR
            tokenizer = load_ctx14_tokenizer(checkpoint_dir=checkpoint, device=self.device)
            rows = []
            for r in self.rows:
                session_dir = self.session_root / r["session_id"]
                full = np.asarray(debug_actions.load_or_build(
                    session_dir, action_encoding="km_ctx14", tokenizer_checkpoint=checkpoint,
                    device=self.device, action_builder=self.action_builder)[0], dtype=np.float32)
                window = full[r["window_start"]:r["window_start"] + self.k * self.T].copy()  # per-tick codes
                n_ctx = r["boundary_tick"] - r["window_start"]
                # The non-causal encoder would otherwise leak the generated future into context
                # codes near the boundary -- re-encode [window_start, boundary_tick) alone.
                window[:n_ctx] = debug_actions.encode_km_ctx14_context(
                    session_dir, r["window_start"], r["boundary_tick"], tokenizer, self.device, self.action_builder)
                rows.append(window)
            km_codes = np.stack(rows).reshape(len(rows), self.T, -1)  # (N, k*T, 32) -> (N, T, k*32)
            return th.from_numpy(km_codes), th.from_numpy(np.zeros((*km_codes.shape[:2], 0), dtype=np.float32))
        raise ValueError(
            f"unknown action_encoding {self.action_encoding!r}, expected 'raw', 'km_fsq', 'raw_fused', or 'km_ctx14'"
        )

    def load_all_actions_raw(self):
        """(keypress: (13, T, 8), mouse: (13, T, 2)) float32, straight from the npz.

        Always the raw tick-resolution arrays, independent of action_encoding --
        overlays and interventions read this, never the model's native conditioning
        tensor (mirrors DebugValidationSet.load_all_actions_raw, plaicraft-debug#80/81)."""
        return th.from_numpy(self._keypress.copy()), th.from_numpy(self._mouse.copy())

    def load_all_keys79(self):
        """(rows, T, 79) keys over PLAICraft's whole vocabulary (plaicraft-debug#93), or None when
        the corpus has only the 8 debug keys. Built from each session's DB on first call."""
        if self.session_root is None or self.action_builder != "raw79":
            return None
        if getattr(self, "_keys79", None) is None:
            ticks = {sid: debug_actions.load_ticks_79(self.session_root / sid) for sid in {r["session_id"] for r in self.rows}}
            self._keys79 = np.stack([ticks[r["session_id"]][r["window_start"]:r["window_start"] + self.k * self.T] for r in self.rows])
        return th.from_numpy(self._keys79.copy())


class PlayerValidationSet(CorpusValidationSet):
    """The issue-85 paired package: the same exercises, but two frame stacks per row.

    Both arms replay one action trace under each player, so they differ only in the cue block
    and the click tint. `frames` is player 0's arm, `frames_p2` player 1's, and the actions are
    shared. The package keeps the name validation.npz in its own directory so every base-class
    behaviour -- manifest checks, the boundary assert, the raw/raw_fused/km_fsq branches --
    applies unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        npz = np.load(self.validation_dir / "validation.npz", allow_pickle=False)
        if "frames_p2" not in npz:
            raise ValueError(
                f"{self.validation_dir / 'validation.npz'} has no frames_p2: this is a plain "
                f"issue-81 package, not a paired player one. Point --player_validation_dir at "
                f"a directory built by plaicraft-debug's build_player_validation.py."
            )
        self._frames_p2 = np.asarray(npz["frames_p2"], dtype=np.float32)
        if self._frames_p2.shape != self._frames.shape:
            raise ValueError(
                f"frames_p2.shape={self._frames_p2.shape} != frames.shape={self._frames.shape}; "
                f"the two arms must be the same trace rendered twice"
            )
        self.player_indices = [int(p) for p in npz["player_indices"]] if "player_indices" in npz else [0, 1]
        if len(self.player_indices) != 2:
            raise ValueError(f"expected exactly 2 player_indices, got {self.player_indices}")

    def load_all_p2(self):
        """(N, T, H, W) float32 in [-1, 1]: the SECOND player's arm of every row."""
        return th.from_numpy(self._frames_p2.copy())
