"""Draw crop (green), weeds (red), the in-row band (gray), and boxes around plants that need a check."""
import numpy as np
from PIL import Image, ImageDraw

from scout.models import Box

CROP_RGB = (99, 153, 34)
WEED_RGB = (226, 75, 74)
BAND_RGB = (136, 135, 128)
FLAG_RGB = (163, 45, 45)


def render_overlay(image: Image.Image, crop: np.ndarray, weed: np.ndarray,
                   band: np.ndarray | None = None, flagged: list[Box] = ()) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, rgb, alpha in ((band, BAND_RGB, 0.25), (crop, CROP_RGB, 0.55), (weed, WEED_RGB, 0.65)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for left, top, right, bottom in flagged:
        draw.rectangle((left, top, right - 1, bottom - 1), outline=FLAG_RGB, width=3)
    return result
