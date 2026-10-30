"""Draw crop (green), weeds (magenta), the in-row band (gray), and boxes around plants that need a check.

Weeds are magenta, not red: red over green leaves turns olive for red-green color-blind viewers, the same as the crop.
"""
import numpy as np
from PIL import Image, ImageDraw

from scout.models import Box

CROP_RGB = (99, 153, 34)
WEED_RGB = (230, 0, 170)
BAND_RGB = (136, 135, 128)
FLAG_RGB = (255, 214, 0)


def render_overlay(image: Image.Image, crop: np.ndarray, weed: np.ndarray,
                   band: np.ndarray | None = None, flagged: list[Box] = ()) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, rgb, alpha in ((band, BAND_RGB, 0.25), (crop, CROP_RGB, 0.55), (weed, WEED_RGB, 0.65)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    width = max(3, round(max(result.size) / 200))  # 5 px on a 1024 px photo: still visible when shown phone-width
    for left, top, right, bottom in flagged:
        if right > left and bottom > top:  # an empty mask's bbox is (0, 0, 0, 0)
            draw.rectangle((left, top, right - 1, bottom - 1), outline=FLAG_RGB, width=width)
    return result
