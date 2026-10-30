"""Per-stop measurements, computed only from masks and classifications (never by the model)."""
import math
from dataclasses import dataclass

import numpy as np

from scout.models import Instance

APPROX_PIECE_FRAC = 0.02  # one weed piece over 2% of the ground likely merges several plants
CAPTURE_HEIGHT_M = 1.2
PHONE_HFOV_DEG = 69.0


@dataclass
class Metrics:
    weed_cover_pct: float
    crop_cover_pct: float
    weed_count: int
    count_approximate: bool
    density_per_m2: float | None
    in_row_weed_share: float | None
    species: dict[str, int]
    species_cover_pct: dict[str, float]
    species_density: dict[str, float] | None
    species_max_height_in: dict[str, float]
    max_height_in: float | None


def footprint_m2(size: tuple[int, int], view_type: str, capture_height_m: float = CAPTURE_HEIGHT_M,
                 hfov_deg: float = PHONE_HFOV_DEG) -> float | None:
    """Ground area of a straight-down photo; None for any other view."""
    if view_type != "top_down":
        return None
    width, height = size
    ground_width = 2 * capture_height_m * math.tan(math.radians(hfov_deg / 2))
    return ground_width * ground_width * height / width


def species_key(instance: Instance) -> str:
    if instance.species_id:
        return instance.species_id
    return "unidentified_grass" if "grass" in instance.prompt.lower() else "unidentified_broadleaf"


def compute_metrics(instances: list[Instance], masks: dict[str, np.ndarray], size: tuple[int, int],
                    view_type: str, sky: np.ndarray | None = None, band: np.ndarray | None = None) -> Metrics:
    width, height = size
    ground = ~sky if sky is not None else np.ones((height, width), dtype=bool)
    ground_px = max(int(ground.sum()), 1)
    crop = np.zeros((height, width), dtype=bool)
    for inst in instances:
        if inst.role == "crop":
            crop |= masks[inst.id]
    crop &= ground
    weed_mask = np.zeros((height, width), dtype=bool)
    species_masks: dict[str, np.ndarray] = {}
    species_counts: dict[str, int] = {}
    species_heights: dict[str, float] = {}
    heights: list[float] = []
    approximate = False
    for inst in instances:
        if inst.role != "weed":
            continue
        mask = masks[inst.id] & ground & ~crop
        if not mask.any():
            continue
        key = species_key(inst)
        weed_mask |= mask
        species_masks[key] = species_masks.get(key, np.zeros_like(mask)) | mask
        species_counts[key] = species_counts.get(key, 0) + 1
        if inst.height_in is not None:
            heights.append(inst.height_in)
            species_heights[key] = max(species_heights.get(key, 0.0), inst.height_in)
        if mask.sum() > APPROX_PIECE_FRAC * ground_px:
            approximate = True
    weed_px = int(weed_mask.sum())
    count = sum(species_counts.values())
    area = footprint_m2(size, view_type)
    return Metrics(
        weed_cover_pct=round(100 * weed_px / ground_px, 2),
        crop_cover_pct=round(100 * int(crop.sum()) / ground_px, 2),
        weed_count=count,
        count_approximate=approximate,
        density_per_m2=round(count / area, 2) if area else None,
        in_row_weed_share=round(float((weed_mask & band).sum()) / weed_px, 3) if band is not None and weed_px else None,
        species=species_counts,
        species_cover_pct={key: round(100 * int(m.sum()) / ground_px, 2) for key, m in species_masks.items()},
        species_density={key: round(c / area, 2) for key, c in species_counts.items()} if area else None,
        species_max_height_in=species_heights,
        max_height_in=max(heights) if heights else None,
    )
