# improved_diffusion/decode_debug.py
#
# Self-contained overlay-video renderer for the plaicraft-debug pixel-space
# world. Visually reproduces plaicraft-model-pi0's inference/decode_debug.py
# (top action bar + GT-over-Pred video), but with none of that repo's
# coupling: no VAE, no audio, no keypress-autoencoder. Actions are read
# straight from the raw sqlite session DB (keyboard / mouse_click /
# mouse_movement tables) since both the GT and prediction rows show the
# SAME ground-truth action bar.
import sqlite3
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ #
#  Layout constants — copied verbatim from plaicraft-model-pi0's
#  src/utils/constants.py so rendered frames are pixel-comparable.
# ------------------------------------------------------------------ #
DECODE_FINAL_FRAME_SIZE = (1280, 768)  # (W, H)
DECODE_TOP_BAR_HEIGHT = 100
DECODE_KEY_BOX_HEIGHT = 90
DECODE_KEY_BOX_PADDING_X = 10
DECODE_KEY_ROW_GAP = 5
DECODE_LEFT_SECTION_MAX_W = 0.45
DECODE_KEY_FONT_SCALE = 1.5
DECODE_KEY_FONT_THICKNESS = 2
DECODE_MOUSE_LINE_COLOR = (0, 255, 255)
DECODE_MOUSE_LINE_THICKNESS = 3
DECODE_MOUSE_ARROW_TIP_LEN = 0.2
DECODE_VIDEO_FPS = 10

FRAME_DURATION_MS = 1000.0 / DECODE_VIDEO_FPS  # 100 ms per video frame

# GLFW key_id (as stored, stringified) -> label drawn on the key box.
KEY_ID_TO_NAME = {
    "87": "w", "65": "a", "83": "s", "68": "d",
    "32": "space", "340": "Shift_L",
}


# ------------------------------------------------------------------ #
#  Action lookup: read straight from the raw sqlite session DB.
# ------------------------------------------------------------------ #
def get_frame_actions(session_db_path, start_frame_idx, n_frames):
    """
    Return a list of length n_frames of per-frame action dicts:
      {"keys": [key_name, ...], "clicks": [button, ...], "mouseDX": float, "mouseDY": float}
    Frame i (absolute index start_frame_idx + i) covers
    [ (start_frame_idx+i)*100, (start_frame_idx+i+1)*100 ) ms.
    """
    con = sqlite3.connect(str(session_db_path))
    cur = con.cursor()
    cur.execute("SELECT key_id, start_timestamp, end_timestamp FROM keyboard")
    key_rows = cur.fetchall()
    cur.execute("SELECT mouse_key_type, start_timestamp, end_timestamp FROM mouse_click")
    click_rows = cur.fetchall()
    cur.execute("SELECT timestamp, mouseDX, mouseDY FROM mouse_movement")
    mouse_by_ts = {int(ts): (dx, dy) for ts, dx, dy in cur.fetchall()}
    con.close()

    actions = []
    for i in range(n_frames):
        abs_idx = start_frame_idx + i
        win_start = abs_idx * FRAME_DURATION_MS
        win_end = win_start + FRAME_DURATION_MS

        keys = [
            KEY_ID_TO_NAME.get(key_id, f"Key_{key_id}")
            for key_id, s, e in key_rows
            if s < win_end and e > win_start
        ]
        clicks = [
            btn for btn, s, e in click_rows
            if s < win_end and e > win_start
        ]
        dx, dy = mouse_by_ts.get(int(win_start), (0.0, 0.0))

        actions.append({"keys": keys, "clicks": clicks, "mouseDX": dx, "mouseDY": dy})
    return actions


# ------------------------------------------------------------------ #
#  Drawing helpers (ported from plaicraft-model-pi0's decode_debug.py)
# ------------------------------------------------------------------ #
def _draw_keyboard_keys(bar, pressed_names, frame_w):
    if not pressed_names:
        return
    key_x, key_y = 10, 10
    max_w = int(frame_w * DECODE_LEFT_SECTION_MAX_W)
    for key_name in pressed_names:
        (text_w, text_h), _ = cv2.getTextSize(
            key_name, cv2.FONT_HERSHEY_SIMPLEX,
            DECODE_KEY_FONT_SCALE, DECODE_KEY_FONT_THICKNESS
        )
        box_w = text_w + 2 * DECODE_KEY_BOX_PADDING_X
        if key_x + box_w > max_w:
            key_x = 10
            key_y += DECODE_KEY_BOX_HEIGHT + DECODE_KEY_ROW_GAP
            if key_y + DECODE_KEY_BOX_HEIGHT > DECODE_TOP_BAR_HEIGHT:
                break
        cv2.rectangle(bar, (key_x, key_y),
                      (key_x + box_w, key_y + DECODE_KEY_BOX_HEIGHT), (0, 0, 0), 2)
        text_x = key_x + (box_w - text_w) // 2
        text_y = key_y + (DECODE_KEY_BOX_HEIGHT + text_h) // 2
        cv2.putText(bar, key_name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, DECODE_KEY_FONT_SCALE,
                    (0, 0, 0), DECODE_KEY_FONT_THICKNESS, cv2.LINE_AA)
        key_x += box_w + DECODE_KEY_ROW_GAP


def _draw_mouse_clicks(bar, pressed_clicks, frame_w):
    mouse_x0 = frame_w - 120
    mouse_y0 = 10
    mouse_w, mouse_h = 100, 60

    cv2.rectangle(bar, (mouse_x0, mouse_y0), (mouse_x0 + mouse_w, mouse_y0 + mouse_h), (0, 0, 0), 2)
    cv2.line(bar, (mouse_x0 + mouse_w // 2, mouse_y0),
             (mouse_x0 + mouse_w // 2, mouse_y0 + mouse_h // 2), (0, 0, 0), 2)

    if "left" in pressed_clicks:
        cv2.rectangle(bar, (mouse_x0 + 2, mouse_y0 + 2),
                      (mouse_x0 + mouse_w // 2 - 2, mouse_y0 + mouse_h // 2 - 2), (0, 255, 0), -1)
    if "right" in pressed_clicks:
        cv2.rectangle(bar, (mouse_x0 + mouse_w // 2 + 2, mouse_y0 + 2),
                      (mouse_x0 + mouse_w - 2, mouse_y0 + mouse_h // 2 - 2), (0, 255, 0), -1)


def _draw_mouse_arrow(content, dx, dy):
    if dx == 0 and dy == 0:
        return
    c_h, c_w = content.shape[:2]
    cx, cy = c_w // 2, c_h // 2
    dx_scaled = int(dx * (c_w / 1920) * 2)
    dy_scaled = int(dy * (c_h / 1080) * 2)
    nx = int(np.clip(cx + dx_scaled, 0, c_w - 1))
    ny = int(np.clip(cy + dy_scaled, 0, c_h - 1))
    cv2.arrowedLine(content, (cx, cy), (nx, ny),
                    DECODE_MOUSE_LINE_COLOR, DECODE_MOUSE_LINE_THICKNESS,
                    cv2.LINE_AA, tipLength=DECODE_MOUSE_ARROW_TIP_LEN)


def _overlay_frame(frame_uint8, action, border):
    """frame_uint8: (H, W, 3) uint8 RGB content frame, already upscaled."""
    content = frame_uint8.copy()
    _draw_mouse_arrow(content, action["mouseDX"], action["mouseDY"])

    c_h, c_w = content.shape[:2]
    top_bar = np.full((DECODE_TOP_BAR_HEIGHT, c_w, 3), 255, dtype=np.uint8)
    _draw_keyboard_keys(top_bar, action["keys"], c_w)
    _draw_mouse_clicks(top_bar, action["clicks"], c_w)

    frame = np.vstack((top_bar, content))
    if border:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 0, 0), 4)
    return frame


def _to_uint8_frame(frame_chw):
    """(3, H, W) float array in [-1, 1] -> (H, W, 3) uint8 RGB, nearest-neighbour upscaled."""
    arr = np.asarray(frame_chw, dtype=np.float32)
    arr = np.clip((arr + 1) * 127.5, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    arr = cv2.resize(arr, DECODE_FINAL_FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
    return arr


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #
def render_overlay(gt_frames, pred_frames, session_db_path, start_frame_idx,
                   out_path, n_observed=10, title=None, pred_actions=None):
    """
    gt_frames, pred_frames: (T, 3, 24, 40) float arrays/tensors in [-1, 1]
    start_frame_idx: index of frame 0 of this window within the session (for action lookup)
    n_observed: first N frames are context; drawn with a red border
    pred_actions: optional list of T action-bar dicts to draw on the PREDICTED
        row, in the same shape get_frame_actions returns. Pass this when the
        model generates its own actions; without it both rows are painted with
        the recorded actions and the generated ones are never visible.
    Writes an mp4 to out_path at DECODE_VIDEO_FPS. Returns out_path.
    """
    gt_frames = np.asarray(gt_frames)
    pred_frames = np.asarray(pred_frames)
    T = gt_frames.shape[0]
    actions = get_frame_actions(session_db_path, start_frame_idx, T)
    if pred_actions is None:
        pred_actions = actions

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode H.264 via imageio/ffmpeg, NOT cv2.VideoWriter. cv2's "mp4v" is
    # MPEG-4 Part 2, which browsers cannot decode in an HTML5 <video> element --
    # so the file uploads to wandb fine and then will not play. cv2's "avc1" is
    # not an option: this container's OpenCV has no H.264 encoder built in
    # ("Could not find encoder for codec_id=27").
    import imageio

    writer = imageio.get_writer(
        str(out_path), fps=DECODE_VIDEO_FPS, codec="libx264",
        macro_block_size=1,  # frame is 1280x1736; don't let ffmpeg resize it
        ffmpeg_params=["-pix_fmt", "yuv420p"],  # required for browser playback
    )
    for t in range(T):
        border = t < n_observed
        gt_content = _to_uint8_frame(gt_frames[t])
        pred_content = _to_uint8_frame(pred_frames[t])
        gt_overlay = _overlay_frame(gt_content, actions[t], border=True)
        pred_overlay = _overlay_frame(pred_content, pred_actions[t], border=border)
        combined = cv2.vconcat([gt_overlay, pred_overlay])
        writer.append_data(combined)  # imageio expects RGB, which is what we have

    writer.close()
    return out_path
