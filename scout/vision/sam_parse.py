"""Parse SAM 3.1 output text into records with box-local boolean masks.

Record grammar: ID<|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|><|mask;x=0;y=0[;c=];data=H,W,<payload>|>
x2/y2 are inclusive. The mask raster covers the box. Fields are read by name, and the base85 payload never
contains , ; < > or |, so ';' always separates fields.
"""
import logging
import re
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import InvalidSegmentationMaskError, SegmentationMask, decode_mask_to_raster
from PIL import Image

from scout.models import Box
from scout.vision.masks import paste

log = logging.getLogger(__name__)

_RECORD = re.compile(r"([A-Za-z0-9_-]+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_DATA = re.compile(r"(\d+),(\d+),([!~][^|]+)")


@dataclass
class SamRecord:
    object_id: str
    box: Box
    confidence: float | None
    mask: np.ndarray  # bool, shaped (box height, box width)
    resized: bool = False  # True when SAM's raster size differed from the box and was stretched to fit

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        return paste(self.mask, self.box, size)


def _fields(body: str) -> dict[str, str]:
    out = {}
    for part in body.split(";"):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _record(object_id: str, box_body: str, mask_body: str) -> SamRecord:
    box, mask = _fields(box_body), _fields(mask_body)
    parsed = _DATA.fullmatch(mask.get("data", ""))
    if parsed is None:
        raise ValueError("mask has no H,W,payload data")
    height, width, payload = int(parsed[1]), int(parsed[2]), parsed[3]
    encoding = "lossless" if payload.startswith("~") else "one_bit"
    raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))
    local = np.frombuffer(raster, dtype=np.uint8).reshape(height, width).astype(bool)
    left, top = int(box["x1"]), int(box["y1"])
    right, bottom = int(box["x2"]) + 1, int(box["y2"]) + 1
    resized = local.shape != (bottom - top, right - left)
    if resized:
        local = np.array(Image.fromarray(local).resize((right - left, bottom - top), Image.NEAREST)).astype(bool)
    confidence = box.get("c") or mask.get("c")
    return SamRecord(object_id, (left, top, right, bottom), float(confidence) if confidence else None, local, resized)


def parse_sam_output(text: str) -> list[SamRecord]:
    """Records that fail to parse or decode are skipped and logged, so one bad record cannot sink the call."""
    records, errors = [], []
    for match in _RECORD.finditer(text):
        try:
            records.append(_record(*match.groups()))
        except (InvalidSegmentationMaskError, KeyError, ValueError) as error:
            errors.append(f"record {match[1]}: {type(error).__name__}: {error}")
    unmatched = text.count("<|box") - len(records) - len(errors)
    if unmatched > 0:
        errors.append(f"{unmatched} record(s) not in ID<|box;...|><|mask;...|> form")
    if errors:
        log.warning("SAM output: kept %d records, skipped: %s", len(records), errors[:3])
    return records
