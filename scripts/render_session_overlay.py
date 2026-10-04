"""Render the debug overlay over a corpus session's own frames (plaicraft-debug#93 reconstruction check)."""
import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np

from improved_diffusion.debug_actions import load_or_build_raw, load_ticks_79
from improved_diffusion.debug_validation import _action_bars
from improved_diffusion.decode_debug import DECODE_VIDEO_FPS, _overlay_frame, _to_uint8_frame, maybe_decode_frames


def render_clip(session_dir, start, length, out_path, action_builder="raw79", vae=None):
    session_dir = Path(session_dir)
    with h5py.File(next((session_dir / "encoded_video_hdf5").glob("*.hdf5")), "r") as f:
        frames = f["frames"][start:start + length]
    pixels = np.asarray(maybe_decode_frames(np.asarray(frames, dtype=np.float32), vae=vae), dtype=np.float32)
    keypress, mouse = load_or_build_raw(session_dir, action_builder=action_builder)
    if action_builder == "raw79":
        keypress = load_ticks_79(session_dir)  # all 79 keys, causally shifted like the cache
    bars = _action_bars(keypress[start:start + len(pixels)], mouse[start:start + len(pixels)])
    writer = imageio.get_writer(str(out_path), fps=DECODE_VIDEO_FPS)
    for px, bar in zip(pixels, bars):
        writer.append_data(_overlay_frame(_to_uint8_frame(px), bar, border=False))
    writer.close()
    return len(pixels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session_dir", required=True)
    p.add_argument("--starts", type=int, nargs="+", required=True, help="clip start ticks")
    p.add_argument("--length", type=int, default=250, help="ticks per clip (250 = 20 s)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--action_builder", default="raw79")
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    sid = Path(a.session_dir).name
    for s in a.starts:
        n = render_clip(a.session_dir, s, a.length, Path(a.out_dir) / f"{sid}_{s:06d}.mp4", a.action_builder)
        print(f"{sid} @ {s}: {n} frames")


if __name__ == "__main__":
    main()
