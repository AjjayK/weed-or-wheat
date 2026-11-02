"""Everything known about one stop while the agent works on it.

The agent sees and gives boxes as [x1, y1, x2, y2] from 0 to 1000 on both axes (x from the left edge, y from the
top). Code works in work-image pixels; to_pixels and to_thousandths convert between the two.
"""
import time
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from scout.app.ingest import StopImage
from scout.models import AgentStep, Box, Instance
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.patches import green_mask, patch_mask
from scout.vision.rows import band_mask, canopy_band, fit_rows

SCALE = 1000
MIN_BOX_PX = 2
LOW_SAM_CONFIDENCE = 0.5
CANOPY_SPAN_FRAC = 1 / 3  # a crop mask this long (share of the image's long side) is a row canopy, not one plant
MOSTLY_CROP = 0.6  # a weed mask this much on crop plants is a crop hit and is dropped
CHECK_OVERLAP = 0.3  # a weed mask this much on any crop is kept but flagged for a closer look
PATCH_FRAC = 0.12  # a weed mask over 12% of the frame is a patch of plants: kept as one, flagged, count approximate
PATCH_MOSTLY_CROP = 0.5  # a patch this much on any crop (canopy too) is crop foliage, not a weed patch: dropped
MAX_BOXES = 16  # SAM takes at most 16 objects per request
MAX_MARK_BOXES = 120  # one mark call; code splits it into SAM requests of MAX_BOXES
PATCH_BOX_MOSTLY_CROP = 0.5  # a weed-area box this much on crop is a box around the row: refused
UNMARKED_SHARE = 0.8  # marking fewer plants than this share of the agent's own estimate suggests weeds were missed


class StopFailed(Exception):
    """The stop cannot continue (e.g. SAM rejected the same tool twice)."""


def _span(a: float, b: float, extent: int) -> tuple[int, int]:
    lo, hi = sorted((a * extent / SCALE, b * extent / SCALE))
    lo = round(min(max(lo, 0.0), extent - MIN_BOX_PX))
    hi = round(min(max(hi, lo + MIN_BOX_PX), extent))
    return lo, hi


def to_pixels(box: Sequence[float], size: tuple[int, int]) -> Box:
    """A 0-1000 box from the agent -> work-image pixels: rounded, clamped to the image, at least 2 px a side."""
    if len(box) != 4:
        raise ValueError(f"a box is [x1, y1, x2, y2], got {list(box)}")
    x1, y1, x2, y2 = (float(v) for v in box)
    (left, right), (top, bottom) = _span(x1, x2, size[0]), _span(y1, y2, size[1])
    return (left, top, right, bottom)


def to_thousandths(box: Box, size: tuple[int, int]) -> list[int]:
    """A work-image pixel box -> [x1, y1, x2, y2] from 0 to 1000, for the agent."""
    width, height = size
    left, top, right, bottom = box
    return [round(left * SCALE / width), round(top * SCALE / height),
            round(right * SCALE / width), round(bottom * SCALE / height)]


class StopState:
    def __init__(self, image: StopImage, crop: str, on_step: Callable[[AgentStep], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.image = image
        self.crop = crop
        self.instances: dict[str, Instance] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.sky: np.ndarray | None = None
        self._sky_tried = False
        self.view_type: str | None = None
        self.crop_stage: str | None = None
        self.notes = ""
        self.steps: list[AgentStep] = []
        self.calls = 0
        self.bad_requests: dict[str, int] = {}
        self.box_cap_hit = False  # a weed mark used all 16 boxes, so the weed count is a floor
        self.patch_boxes: dict[str, Box] = {}  # weed areas the agent drew; their masks follow the crop
        self.crop_seen: str | None = None  # from survey
        self.weeds_estimate: int | None = None  # from survey: weed plants the agent sees
        self.all_weeds_marked: bool | None = None  # from finish; None when the stop never finished
        self._green: np.ndarray | None = None
        self.finished = False
        self._next = 0
        self._on_step = on_step
        self._clock = clock
        self.started = clock()

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def log(self, tool: str, summary: str, instance_ids=(), ok: bool = True) -> AgentStep:
        step = AgentStep(t_ms=int((self._clock() - self.started) * 1000), tool=tool, summary=summary,
                         instance_ids=list(instance_ids), ok=ok)
        self.steps.append(step)
        if self._on_step:
            self._on_step(step)
        return step

    def add_mask(self, mask: np.ndarray, source: str, prompt: str, role: str = "unknown",
                 confidence: float | None = None) -> Instance:
        self._next += 1
        inst = Instance(id=f"s{self.image.index}-i{self._next}", source=source, prompt=prompt, box=mk.bbox(mask),
                        sam_confidence=confidence, area_px=int(mask.sum()), role=role,
                        check_this=confidence is not None and confidence < LOW_SAM_CONFIDENCE)
        self.instances[inst.id] = inst
        self.masks[inst.id] = mask
        return inst

    def union_of(self, predicate: Callable[[Instance], bool]) -> np.ndarray:
        return mk.union([self.masks[i.id] for i in self.instances.values() if predicate(i)], self.size)

    def is_canopy(self, inst: Instance) -> bool:
        """A crop mask spanning a third of the image's long side is a row canopy, not one plant."""
        left, top, right, bottom = inst.box
        return max(right - left, bottom - top) >= CANOPY_SPAN_FRAC * max(self.size)

    def _patch_sized(self, area_px: int) -> bool:
        width, height = self.size
        return area_px > PATCH_FRAC * width * height

    def is_patch(self, inst: Instance) -> bool:
        return self._patch_sized(inst.area_px)

    def add_weed(self, mask: np.ndarray, source: str, prompt: str, confidence: float | None = None) -> Instance | None:
        """Register a weed mask, or return None when it mostly lies on crop (a box drawn on crop).

        Only separate crop plants can veto a plant-sized weed: a weed growing inside a row sits almost wholly in the
        row-canopy mask, so canopy overlap only flags it. A patch-sized mask is different: a box that big, mostly on
        any crop, canopy included, is crop foliage near the camera, so it is dropped. Weeds overlapping any crop,
        and the patches that stay, get check_this.
        """
        area = max(int(mask.sum()), 1)
        plants = self.union_of(lambda i: i.role == "crop" and not self.is_canopy(i))
        if float((mask & plants).sum()) / area >= MOSTLY_CROP:
            return None
        crop = self.union_of(lambda i: i.role == "crop")
        crop_share = float((mask & crop).sum()) / area
        patch = self._patch_sized(area)
        if patch and crop_share >= PATCH_MOSTLY_CROP:
            return None
        inst = self.add_mask(mask, source, prompt, role="weed", confidence=confidence)
        inst.check_this = bool(inst.check_this or crop_share > CHECK_OVERLAP or patch)
        return inst

    def unclassified(self) -> list[str]:
        """Weeds that were marked but never classified."""
        return [i.id for i in self.instances.values() if i.role == "weed" and i.id_confidence is None]

    def has_patch(self) -> bool:
        return any(i.role == "weed" and self.is_patch(i) for i in self.instances.values())

    @property
    def green(self) -> np.ndarray:
        if self._green is None:
            self._green = green_mask(self.image.work)
        return self._green

    def _crop_for_areas(self) -> np.ndarray:
        """Crop that weed areas leave out. A weed area later classified as crop is not crop for the others."""
        return self.union_of(lambda i: i.role == "crop" and i.source != "patch")

    def add_patch(self, box: Box, label: str) -> tuple[Instance | None, str | None]:
        """Register a weed area measured from green pixels, or say why not."""
        left, top, right, bottom = box
        crop = self._crop_for_areas()
        if crop[top:bottom, left:right].mean() > PATCH_BOX_MOSTLY_CROP:
            return None, "mostly crop: box the weeds between the rows, not the row"
        mask, pieces = patch_mask(self.green, box, crop)
        if not mask.any():
            return None, "no green weed pixels in this area"
        self._next += 1
        inst = Instance(id=f"s{self.image.index}-i{self._next}", source="patch", prompt=label, box=box,
                        area_px=int(mask.sum()), role="weed", plant_count=pieces)
        self.instances[inst.id], self.masks[inst.id], self.patch_boxes[inst.id] = inst, mask, box
        return inst, None

    def refresh_patches(self) -> None:
        """Recompute every weed area against the current crop, so a crop leaf marked later leaves the area."""
        crop = self._crop_for_areas()
        for iid, box in self.patch_boxes.items():
            mask, pieces = patch_mask(self.green, box, crop)
            self.masks[iid] = mask
            self.instances[iid].area_px, self.instances[iid].plant_count = int(mask.sum()), pieces

    def _weed_areas(self) -> list[Instance]:
        """Weed areas that still hold weed pixels. One that crop marked later covers completely stays registered (a
        later change to the crop can refill it) but no longer counts."""
        return [i for i in self.instances.values() if i.role == "weed" and i.source == "patch" and i.area_px > 0]

    def has_weed_area(self) -> bool:
        return bool(self._weed_areas())

    def weeds_marked(self) -> tuple[int, int]:
        """(weeds boxed one by one, weed areas)."""
        weeds = sum(1 for i in self.instances.values() if i.role == "weed" and i.source != "patch")
        return weeds, len(self._weed_areas())

    def plants_marked(self) -> int:
        """Every weed's plant count added up. An upper bound: it doesn't apply the measuring step's duplicate rules,
        so boxes and weed areas that overlap are counted twice."""
        return sum(i.plant_count for i in self.instances.values() if i.role == "weed")

    def maybe_unmarked(self, plants: int | None = None) -> bool:
        """The agent didn't confirm every weed is marked, or, without weed areas, marked well under its estimate.

        plants is the measured weed count when the caller has it; otherwise the raw sum from plants_marked."""
        if self.all_weeds_marked is not True:
            return True
        if self.weeds_estimate and not self.has_weed_area():
            marked = plants if plants is not None else self.plants_marked()
            return marked < UNMARKED_SHARE * self.weeds_estimate
        return False

    def count_approximate(self, view_type: str | None = None) -> bool:
        """A patch counts as one weed (but a big plant in a close-up is just a plant), a weed area's pieces are not
        exactly its plants, and a capped mark may have left weeds unmarked."""
        return (self.has_patch() and view_type != "close_up") or self.has_weed_area() or self.box_cap_hit

    def ensure_sky(self, sam) -> None:
        """Sky for row and wide views, asked once. Best effort: without it, cover is measured over the whole photo."""
        if self.sky is not None or self._sky_tried:
            return
        self._sky_tried = True
        try:
            records = sam.segment_concept(self.image, "sky")
        except Exception as error:
            self.log("sky", f"Sky not found ({type(error).__name__}); cover uses the whole photo", ok=False)
            return
        self.sky = mk.union([r.full_mask(self.size) for r in records], self.size)

    def row_band(self) -> np.ndarray | None:
        """In-row zone: none for alfalfa (solid-seeded) or without crop; the row canopies when any crop mask is
        canopy-sized, else lines fitted through crop-plant centers. Sky is never in the band."""
        crops = [i for i in self.instances.values() if i.role == "crop"]
        if self.crop == "alfalfa" or not crops:
            return None
        if any(self.is_canopy(i) for i in crops):
            band = canopy_band(self.union_of(lambda i: i.role == "crop"))
        else:
            centers = [((i.box[0] + i.box[2]) / 2, (i.box[1] + i.box[3]) / 2) for i in crops]
            widths = [((i.box[2] - i.box[0]) + (i.box[3] - i.box[1])) / 2 for i in crops]
            fit = fit_rows(centers, widths)
            band = band_mask(fit, self.size) if fit else None
        if band is not None and self.sky is not None:
            band = band & ~self.sky
        return band

    def overlay(self) -> Image.Image:
        weed = self.union_of(lambda i: i.role == "weed")
        crop = self.union_of(lambda i: i.role == "crop") & ~weed  # weeds win: a weed in the row keeps its pixels
        flagged = [i.box for i in self.instances.values() if i.role == "weed" and i.check_this]
        areas = [self.patch_boxes[i.id] for i in self._weed_areas()]
        return render_overlay(self.image.work, crop, weed, self.row_band(), flagged, areas)
