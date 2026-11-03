"""Per-stop measurements, computed only from masks and classifications (never by the model)."""
import math
from dataclasses import dataclass

import numpy as np

from scout.models import Instance

APPROX_PIECE_FRAC = 0.02  # one weed piece over 2% of the ground likely merges several plants
CAPTURE_HEIGHT_M = 1.2
GRASS_WORDS = ("grass", "foxtail", "sedge", "barnyard", "panicum", "shattercane")
PHONE_HFOV_DEG = 69.0  # spans the sensor's long side, whichever way the phone was held
MIN_GROUND_FRAC = 0.2  # a sky mask leaving less ground than this is a bad SAM answer; ignore it
MIN_OWN_FRAC = 0.5  # a weed must own at least half of its on-ground pixels to count as a plant


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
    """Ground area of a straight-down photo; None for any other view. The FOV spans the photo's long side."""
    long_px, short_px = max(size), min(size)
    if view_type != "top_down" or short_px <= 0:
        return None
    ground_long = 2 * capture_height_m * math.tan(math.radians(hfov_deg / 2))
    return ground_long * ground_long * short_px / long_px


def species_key(instance: Instance) -> str:
    if instance.species_id:
        return instance.species_id
    label = instance.prompt.lower()
    return "unidentified_grass" if any(word in label for word in GRASS_WORDS) else "unidentified_broadleaf"


def compute_metrics(instances: list[Instance], masks: dict[str, np.ndarray], size: tuple[int, int],
                    view_type: str, sky: np.ndarray | None = None, band: np.ndarray | None = None,
                    capture_height_m: float = CAPTURE_HEIGHT_M) -> Metrics:
    """Each pixel counts once. Weeds win over crop (the tools already dropped weed masks that were mostly crop).
    Plain weeds are measured before weed areas, and within each group the weed registered first owns overlapping
    pixels."""
    width, height = size
    ground = np.ones((height, width), dtype=bool)
    if sky is not None and view_type in ("row", "wide") and (~sky).mean() >= MIN_GROUND_FRAC:
        ground = ~np.asarray(sky, dtype=bool)
    ground_px = max(int(ground.sum()), 1)
    crop = np.zeros((height, width), dtype=bool)
    for inst in instances:
        if inst.role == "crop" and inst.id in masks:
            crop |= np.asarray(masks[inst.id], dtype=bool)
    crop &= ground
    weed_mask = np.zeros((height, width), dtype=bool)
    species_masks: dict[str, np.ndarray] = {}
    species_counts: dict[str, int] = {}
    species_heights: dict[str, float] = {}
    heights: list[float] = []
    approximate = False
    # a box inside a weed area is its own plant, so plain weeds are measured before the areas claim their pixels
    for inst in sorted(instances, key=lambda inst: inst.source == "patch"):
        if inst.role != "weed" or inst.id not in masks:
            continue
        on_ground = np.asarray(masks[inst.id], dtype=bool) & ground
        own = on_ground & ~weed_mask
        own_px, mask_ground_px = int(own.sum()), int(on_ground.sum())
        if own_px == 0:
            continue
        if inst.source == "patch":  # a weed area is many plants: count the share of its pieces no box counted yet
            plants = max(1, round(inst.plant_count * own_px / mask_ground_px))
        elif own_px < MIN_OWN_FRAC * mask_ground_px:
            continue  # a second box on a plant already counted
        else:
            plants = inst.plant_count
        key = species_key(inst)
        weed_mask |= own
        if key in species_masks:
            species_masks[key] |= own
        else:
            species_masks[key] = own
        species_counts[key] = species_counts.get(key, 0) + plants  # a weed area holds several plants
        if inst.height_in is not None:
            heights.append(inst.height_in)
            species_heights[key] = max(species_heights.get(key, 0.0), inst.height_in)
        if inst.source == "patch" or (view_type != "close_up" and own_px > APPROX_PIECE_FRAC * ground_px):
            approximate = True  # a weed area's pieces are not exactly its plants
    crop &= ~weed_mask
    weed_px = int(weed_mask.sum())
    count = sum(species_counts.values())
    area = footprint_m2(size, view_type, capture_height_m)
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
