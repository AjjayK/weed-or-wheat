"""Boolean mask helpers. Masks are numpy bool arrays shaped (height, width); sizes are (width, height)."""
import numpy as np
from scipy import ndimage

from scout.models import Box

_EIGHT = np.ones((3, 3), dtype=bool)


def paste(local: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """Place a box-local mask onto a full canvas, clipping anything outside it."""
    width, height = size
    canvas = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    if local.shape != (bottom - top, right - left):
        raise ValueError(f"mask shape {local.shape} does not match box {box}")
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


def pieces(mask: np.ndarray, min_area: float, exclude: np.ndarray | None = None, margin: int = 2) -> list[np.ndarray]:
    """Split a mask into 8-connected pieces of at least min_area pixels.

    Pixels within `margin` px of `exclude` are dropped first. That removes the slivers left where the plant and
    crop masks disagree slightly, without the erosion that would also erase thin grass blades elsewhere.
    """
    if exclude is not None:
        mask = mask & ~ndimage.binary_dilation(exclude, structure=_EIGHT, iterations=margin)
    labels, count = ndimage.label(mask, structure=_EIGHT)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    return [labels == index for index in range(1, count + 1) if areas[index] >= min_area]


def bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pad_box(box: Box, frac: float, size: tuple[int, int]) -> Box:
    left, top, right, bottom = box
    pad = int(max(right - left, bottom - top) * frac)  # same pad on both axes, so a thin blade's box still widens
    width, height = size
    return (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad))
