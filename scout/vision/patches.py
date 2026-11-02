"""Weed areas measured from pixels: green, inside the box the agent drew, and not crop.

For a carpet of shoots too dense or small to box one by one. SAM fills a box around a thin shoot with soil; green
pixels don't: soil, residue and shadow are never green.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from scout.models import Box

GREEN_EXG = 0.08  # excess green 2g - r - b on chromatic coordinates; soil sits near 0, straw near 0.04, leaves 0.3+
CROP_MARGIN_PX = 3  # crop outlines grown this much, so leaf edges the crop mask missed don't count as weed
MIN_PIECE_PX = 20  # green specks smaller than this are noise
_EIGHT = np.ones((3, 3), dtype=bool)


def green_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    total = rgb.sum(axis=2) + 1e-6
    r, g, b = (rgb[..., i] / total for i in range(3))
    return (2 * g - r - b) > GREEN_EXG


def patch_mask(green: np.ndarray, box: Box, crop: np.ndarray) -> tuple[np.ndarray, int]:
    """Green pixels inside the box and away from crop, without specks; and how many separate pieces they form."""
    left, top, right, bottom = box
    inside = np.zeros_like(green)
    inside[top:bottom, left:right] = True
    near_crop = ndimage.binary_dilation(crop, structure=_EIGHT, iterations=CROP_MARGIN_PX)
    labels, count = ndimage.label(green & inside & ~near_crop, structure=_EIGHT)
    keep = np.bincount(labels.ravel(), minlength=count + 1) >= MIN_PIECE_PX
    keep[0] = False
    return keep[labels], int(keep.sum())
