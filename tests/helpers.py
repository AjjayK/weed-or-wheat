"""Shared test helpers."""
import numpy as np
from meta_sam_parser._mask_codec import _encode_raster  # private encoder; only used to build realistic SAM text


def sam_record_text(object_id, box, mask, frame_size, confidence=None):
    left, top, right, bottom = box
    width, height = frame_size
    payload = _encode_raster(mask.astype(np.uint8).ravel().tolist(), mask.shape[1], mask.shape[0])
    conf = f";c={confidence}" if confidence is not None else ""
    return (f"{object_id}<|box;x1={left};y1={top};x2={right - 1};y2={bottom - 1};w={width};h={height}{conf}|>"
            f"<|mask;x=0;y=0{conf};data={mask.shape[0]},{mask.shape[1]},{payload}|>")


def sam_text(records, frame_size):
    """records: list of (object_id, box, mask) or (object_id, box, mask, confidence)."""
    return "<0f>" + ",".join(sam_record_text(r[0], r[1], r[2], frame_size, *r[3:]) for r in records)


import io

from PIL import Image

from scout.app.ingest import load_image


def png_bytes(width=100, height=100, color=(90, 140, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_image(width=100, height=100, index=0, color=(90, 140, 60)):
    return load_image(index, png_bytes(width, height, color))
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.sam_parse import SamRecord


def rect(left, top, right, bottom, size=(100, 100)):
    mask = np.zeros((size[1], size[0]), bool)
    mask[top:bottom, left:right] = True
    return mask


def mask_record(object_id, mask, confidence=0.9):
    """A SAM record for a full-frame mask: its bounding box plus the box-local raster."""
    left, top, right, bottom = mk.bbox(mask)
    return SamRecord(object_id, (left, top, right, bottom), confidence, mask[top:bottom, left:right].copy())


def field_scene():
    """What a 100x100 soybean photo really holds. Test boxes are 0-1000, so 10x these pixel numbers."""
    return {
        "row": rect(10, 0, 30, 100),  # a crop row seen as one canopy, full height
        "plant": rect(40, 30, 52, 42),  # one separate crop plant
        "weed_a": rect(60, 60, 70, 70),
        "weed_b": rect(80, 10, 90, 20),
        "edge_weed": rect(26, 40, 36, 50),  # 40% of it inside the row canopy
        "patch": rect(40, 75, 100, 100),  # 15% of the frame: a weed patch
    }


class FakeSam:
    """Concept prompts answer from a dict. A box prompt returns the scene object that fills most of the box, cut to
    the box as SAM does; a box over bare ground returns nothing."""

    def __init__(self, scene=None, concepts=None, fail=None, confidence=0.9):
        self.scene = field_scene() if scene is None else scene
        self.concepts = concepts or {}
        self.fail = fail or set()
        self.confidence = confidence
        self.calls = []

    def segment_concept(self, image, phrase):
        self.calls.append(("concept", phrase))
        if phrase in self.fail:
            raise BadRequest(f"rejected {phrase}")
        return list(self.concepts.get(phrase, []))

    def segment_boxes(self, image, boxes):
        self.calls.append(("boxes", tuple(boxes)))
        if "boxes" in self.fail:
            raise BadRequest("rejected boxes")
        records = []
        for n, (left, top, right, bottom) in enumerate(boxes, start=1):
            inside = {name: int(mask[top:bottom, left:right].sum()) for name, mask in self.scene.items()}
            best = max(inside, key=inside.get, default=None)
            if best is not None and inside[best]:
                local = self.scene[best][top:bottom, left:right].copy()
                records.append(SamRecord(str(n), (left, top, right, bottom), self.confidence, local))
        return records


def small_weedbook():
    return WeedBook({
        "waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="Amaranthus tuberculatus",
                              type="broadleaf", synonyms=["common waterhemp"]),
        "horseweed": WeedInfo(id="horseweed", common_name="horseweed", scientific_name="Conyza canadensis",
                              type="broadleaf", synonyms=["marestail"]),
        "giant_foxtail": WeedInfo(id="giant_foxtail", common_name="giant foxtail", scientific_name="Setaria faberi",
                                  type="grass"),
        "unidentified_broadleaf": WeedInfo(id="unidentified_broadleaf", common_name="unidentified broadleaf",
                                           scientific_name="", type="broadleaf"),
        "unidentified_grass": WeedInfo(id="unidentified_grass", common_name="unidentified grass",
                                       scientific_name="", type="grass"),
    })
