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
from scout.vision.rows import band_mask, canopy_band, fit_rows

SCALE = 1000
MIN_BOX_PX = 2
LOW_SAM_CONFIDENCE = 0.5
CANOPY_SPAN_FRAC = 1 / 3  # a crop mask this long (share of the image's long side) is a row canopy, not one plant
MOSTLY_CROP = 0.6  # a weed mask this much on crop plants is a crop hit and is dropped
CHECK_OVERLAP = 0.3  # a weed mask this much on any crop is kept but flagged for a closer look
PATCH_FRAC = 0.12  # a weed mask over 12% of the frame is a patch of plants: kept as one, flagged, count approximate
MAX_BOXES = 16  # SAM takes at most 16 objects per request


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
        self.view_type: str | None = None
        self.crop_stage: str | None = None
        self.notes = ""
        self.steps: list[AgentStep] = []
        self.calls = 0
        self.bad_requests: dict[str, int] = {}
        self.box_cap_hit = False  # a weed mark used all 16 boxes, so the weed count is a floor
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

    def is_patch(self, inst: Instance) -> bool:
        width, height = self.size
        return inst.area_px > PATCH_FRAC * width * height

    def add_weed(self, mask: np.ndarray, source: str, prompt: str, confidence: float | None = None) -> Instance | None:
        """Register a weed mask, or return None when it mostly lies on crop plants (a box drawn on crop).

        Only separate crop plants can veto a weed: a weed growing inside a row sits almost wholly in the row-canopy
        mask, so canopy overlap only flags it. Weeds overlapping any crop, and large patches, get check_this.
        """
        area = max(int(mask.sum()), 1)
        plants = self.union_of(lambda i: i.role == "crop" and not self.is_canopy(i))
        if float((mask & plants).sum()) / area >= MOSTLY_CROP:
            return None
        crop = self.union_of(lambda i: i.role == "crop")
        inst = self.add_mask(mask, source, prompt, role="weed", confidence=confidence)
        on_crop = float((mask & crop).sum()) / area > CHECK_OVERLAP
        inst.check_this = bool(inst.check_this or on_crop or self.is_patch(inst))
        return inst

    def unclassified(self) -> list[str]:
        """Weeds that were marked but never classified."""
        return [i.id for i in self.instances.values() if i.role == "weed" and i.id_confidence is None]

    def has_patch(self) -> bool:
        return any(i.role == "weed" and self.is_patch(i) for i in self.instances.values())

    def count_approximate(self) -> bool:
        """A patch counts as one weed, and a capped mark may have left weeds unmarked."""
        return self.has_patch() or self.box_cap_hit

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
        return render_overlay(self.image.work, crop, weed, self.row_band(), flagged)
