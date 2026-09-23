# improved_diffusion/decode_debug.py
#
# Drawing primitives (action bar + frame upscaling) for plaicraft-debug's
# validation overlays, ported from plaicraft-model-pi0's decode_debug.py
# but with none of that repo's coupling: no VAE, no audio, no keypress-autoencoder.

import cv2
import numpy as np
import torch

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
DECODE_VIDEO_FPS = 12.5  # plaicraft-debug#80: the corpus's session.fps, not the old 10

FRAME_DURATION_MS = 1000.0 / DECODE_VIDEO_FPS  # 80 ms per video frame

# GLFW key_id (as stored, stringified) -> label drawn on the key box.
KEY_ID_TO_NAME = {
    "87": "w", "65": "a", "83": "s", "68": "d",
    "32": "space", "340": "Shift_L",
}


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


# ------------------------------------------------------------------ #
#  plaicraft-debug#93 B4: VAE-decode when a session's hdf5 stores latents
#  (4 channels) instead of pixels (3 channels). Same VAE/scale as
#  gaussian_diffusion.decode -- see that module for why.
# ------------------------------------------------------------------ #
_LATENT_SCALE = 0.13025
_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
_default_vae = None


def _load_default_vae():
    global _default_vae
    if _default_vae is None:
        from diffusers import AutoencoderKL
        _default_vae = AutoencoderKL.from_pretrained(_VAE_ID, torch_dtype=torch.float16)
        _default_vae.to("cuda" if torch.cuda.is_available() else "cpu").eval()  # fp16 on CPU crawls
        for p in _default_vae.parameters():
            p.requires_grad = False
    return _default_vae


@torch.no_grad()
def decode_latents_chunked(vae, latents, chunk_size=20):
    """(N, 4, H, W) scaled latents -> (N, 3, H', W') pixels in [-1, 1], chunked to bound
    peak memory over a full session."""
    outs = [vae.decode(latents[i:i + chunk_size] / _LATENT_SCALE).sample
           for i in range(0, latents.shape[0], chunk_size)]
    return torch.cat(outs, dim=0)


def maybe_decode_frames(frames, vae=None, chunk_size=20):
    """frames: (N, C, H, W). C==3 passes through unchanged (debug-world sessions render
    exactly as before); C==4 VAE-decodes to pixel space first, so the overlay below it
    (_to_uint8_frame/_overlay_frame) never has to know the corpus was latent."""
    c = frames.shape[1]
    if c == 3:
        return frames
    if c != 4:
        raise ValueError(f"decode_debug: unexpected frame channel count {c}, expected 3 or 4")
    vae = vae or _load_default_vae()
    t = frames if torch.is_tensor(frames) else torch.from_numpy(np.asarray(frames))
    param = next(vae.parameters(), None)
    if param is not None:
        t = t.to(device=param.device, dtype=param.dtype)
    return decode_latents_chunked(vae, t, chunk_size=chunk_size).cpu()


def _to_uint8_frame(frame_chw):
    """(3, H, W) float array in [-1, 1] -> (H, W, 3) uint8 RGB, nearest-neighbour upscaled."""
    arr = np.asarray(frame_chw, dtype=np.float32)
    arr = np.clip((arr + 1) * 127.5, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    arr = cv2.resize(arr, DECODE_FINAL_FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
    return arr
