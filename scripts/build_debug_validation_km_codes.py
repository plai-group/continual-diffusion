"""Issue #81: derived artifacts for the frozen validation.npz package.

  python scripts/build_debug_validation_km_codes.py \
      --validation-dir <corpus_dir>/validation \
      --tokenizer-checkpoint <path, default km_tokenizer's DEFAULT_CHECKPOINT> \
      --device cuda

Writes <validation-dir>/km_codes.npz (key "km_codes", (13, T, 36)) + a sidecar
manifest, and renders 13 GT-only overlay mp4s to <validation-dir>/decoded/. Both
steps are idempotent: a matching sidecar skips the km_codes rebuild unless
--force; overlays are cheap enough to always re-render.
"""
import argparse
import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import imageio
import numpy as np
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.corpus_validation import CorpusValidationSet
from improved_diffusion.debug_validation import _action_vec_to_bar, _encode_km_actions, _to_display_actions
from improved_diffusion.decode_debug import DECODE_VIDEO_FPS, _overlay_frame, _to_uint8_frame
from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT, _sha256, load_tokenizer


def _km_codes_paths(validation_dir):
    return validation_dir / "km_codes.npz", validation_dir / "km_codes_manifest.json"


def build_km_codes(validation_dir, tokenizer_checkpoint, device, force=False):
    """Scatter+encode the frozen raw keypress/mouse through the km tokenizer once,
    caching the (13, T, 36) codes with a sidecar recording the checkpoint used --
    mirrors debug_actions._load_or_build_km_codes's per-session cache pattern."""
    validation_dir = Path(validation_dir)
    codes_path, sidecar_path = _km_codes_paths(validation_dir)
    checkpoint_path = Path(tokenizer_checkpoint)
    expected = {"tokenizer_sha256": _sha256(checkpoint_path), "subbin_rule_version": da.SUBBIN_RULE_VERSION}

    if not force and codes_path.exists() and sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            sidecar = None
        if sidecar and all(sidecar.get(k) == v for k, v in expected.items()):
            print(f"km_codes up to date at {codes_path}, skipping (--force to rebuild)")
            return codes_path

    valset = CorpusValidationSet(validation_dir)
    keypress, mouse = valset.load_all_actions_raw()  # never load_all_actions: km_fsq would recurse into this builder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(checkpoint_path=checkpoint_path, device=device)
    with torch.no_grad():
        km_codes = _encode_km_actions(tokenizer, keypress.to(device), mouse.to(device))
    km_codes = km_codes.cpu().numpy().astype(np.float32)
    sidecar = dict(expected, built_at=datetime.now(timezone.utc).isoformat())

    try:
        tmp_codes = codes_path.with_name(f".{codes_path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp_codes, km_codes=km_codes)
        os.replace(tmp_codes, codes_path)
        tmp_sidecar = sidecar_path.with_name(f".{sidecar_path.stem}.{os.getpid()}.tmp.json")
        tmp_sidecar.write_text(json.dumps(sidecar, indent=2))
        os.replace(tmp_sidecar, sidecar_path)
    except OSError as e:
        # validation_dir may be read-only/shared; cache in a private tmp dir instead of crashing.
        fallback_dir = Path(tempfile.mkdtemp(prefix="km_codes_"))
        codes_path = fallback_dir / "km_codes.npz"
        np.savez(codes_path, km_codes=km_codes)
        (fallback_dir / "km_codes_manifest.json").write_text(json.dumps(sidecar, indent=2))
        warnings.warn(f"cannot write km_codes cache under {validation_dir} ({e}); using {fallback_dir}")

    print(f"wrote {codes_path} ({km_codes.shape})")
    return codes_path


def render_gt_overlays(validation_dir, out_dir=None):
    """GT-only overlay mp4s built straight from the npz's own baked actions -- no
    session db read, unlike decode_debug.render_overlay (the frozen npz is self-
    contained and the held-out sessions may not even be staged locally)."""
    validation_dir = Path(validation_dir)
    valset = CorpusValidationSet(validation_dir)
    frames = valset.load_all().numpy()
    keypress, mouse = valset.load_all_actions()
    dk, dm = _to_display_actions(keypress), _to_display_actions(mouse)  # boundary action -> frame n_observed-1
    out_dir = Path(out_dir) if out_dir else validation_dir / "decoded"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(valset.rows):
        out_path = out_dir / f"{valset.slug(row)}.mp4"
        writer = imageio.get_writer(
            str(out_path), fps=DECODE_VIDEO_FPS, codec="libx264",
            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        for t in range(valset.T):
            frame = _to_uint8_frame(frames[i, t])
            bar = _action_vec_to_bar(dk[i, t], dm[i, t])
            writer.append_data(_overlay_frame(frame, bar, border=t < valset.n_observed))
        writer.close()
        print(f"wrote {out_path}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validation-dir", required=True)
    p.add_argument("--tokenizer-checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-km-codes", action="store_true")
    p.add_argument("--skip-overlays", action="store_true")
    args = p.parse_args()

    validation_dir = Path(args.validation_dir)
    if not args.skip_km_codes:
        build_km_codes(validation_dir, args.tokenizer_checkpoint, args.device, force=args.force)
    if not args.skip_overlays:
        render_gt_overlays(validation_dir)


if __name__ == "__main__":
    main()
