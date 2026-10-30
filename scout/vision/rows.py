"""Fit crop rows as lines through crop-plant centers (RANSAC), and build an in-row band mask."""
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RowFit:
    orientation: Literal["vertical", "horizontal"]
    lines: list[tuple[float, float]]  # v = a*u + b; vertical rows: u=y, v=x; horizontal rows: u=x, v=y
    half_width: float


def _fit(points, tol, max_lines, min_inliers, rng):
    remaining = list(points)
    lines, total = [], 0
    while len(remaining) >= min_inliers and len(lines) < max_lines:
        best, best_line = [], None
        for _ in range(200):
            (u1, v1), (u2, v2) = rng.sample(remaining, 2)
            if abs(u2 - u1) < 1e-6:
                continue
            a = (v2 - v1) / (u2 - u1)
            b = v1 - a * u1
            inliers = [(u, v) for u, v in remaining if abs(v - (a * u + b)) <= tol]
            if len(inliers) > len(best):
                best, best_line = inliers, (a, b)
        if len(best) < min_inliers:
            break
        us = np.array([p[0] for p in best], dtype=float)
        vs = np.array([p[1] for p in best], dtype=float)
        a, b = np.polyfit(us, vs, 1) if np.ptp(us) > 0 else best_line
        lines.append((float(a), float(b)))
        total += len(best)
        taken = set(best)
        remaining = [p for p in remaining if p not in taken]
    return lines, total


def fit_rows(centroids: list[tuple[float, float]], widths: list[float], max_lines: int = 6,
             min_inliers: int = 3, seed: int = 0) -> RowFit | None:
    """Return rows only when at least two lines each hold at least `min_inliers` crop plants."""
    if len(centroids) < 2 * min_inliers or not widths:
        return None
    half = float(np.median(widths)) / 2
    tol = max(half * 1.2, 4.0)
    best = None
    for orientation in ("vertical", "horizontal"):
        points = [(y, x) if orientation == "vertical" else (x, y) for x, y in centroids]
        lines, total = _fit(points, tol, max_lines, min_inliers, random.Random(seed))
        if len(lines) >= 2 and (best is None or total > best[2]):
            best = (orientation, lines, total)
    if best is None:
        return None
    return RowFit(orientation=best[0], lines=best[1], half_width=half)


def band_mask(fit: RowFit, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width]
    u, v = (ys, xs) if fit.orientation == "vertical" else (xs, ys)
    band = np.zeros((height, width), dtype=bool)
    for a, b in fit.lines:
        band |= np.abs(v - (a * u + b)) <= fit.half_width
    return band
