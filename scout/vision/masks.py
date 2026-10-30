"""Boolean mask helpers. Masks are numpy bool arrays shaped (height, width); sizes are (width, height)."""
import numpy as np
from scipy import ndimage

from scout.models import Box


def paste(local: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """Place a box-local mask onto a full canvas, clipping anything outside it."""
    width, height = size
    canvas = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    l, t = max(left, 0), max(top, 0)
    r, b = min(right, width), min(bottom, height)
    if r <= l or b <= t:
        return canvas
    canvas[t:b, l:r] = local[t - top:b - top, l - left:r - left]
    return canvas


def union(masks: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    out = np.zeros((height, width), dtype=bool)
    for mask in masks:
        out |= mask
    return out


def pieces(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Split a mask into connected pieces, after an opening that removes slivers along crop edges."""
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(cleaned)
    if count == 0:
        return []
    areas = ndimage.sum(cleaned, labels, index=range(1, count + 1))
    return [labels == index + 1 for index, area in enumerate(areas) if area >= min_area]


def bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pad_box(box: Box, frac: float, size: tuple[int, int]) -> Box:
    left, top, right, bottom = box
    pad_x, pad_y = int((right - left) * frac), int((bottom - top) * frac)
    width, height = size
    return (max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y))
