"""Parse SAM 3.1 output text into records with box-local boolean masks.

Record grammar: ID<|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|><|mask;x=0;y=0[;c=];data=H,W,<payload>|>
x2/y2 are inclusive. The mask raster covers the box. The base85 payload never contains '|'.
"""
import re
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from PIL import Image

from scout.models import Box
from scout.vision.masks import paste

_RECORD = re.compile(r"([A-Za-z0-9_-]+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_DATA = re.compile(r"(\d+),(\d+),([!~][^|]+)")


@dataclass
class SamRecord:
    object_id: str
    box: Box
    confidence: float | None
    mask: np.ndarray  # bool, shaped (box height, box width)

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        return paste(self.mask, self.box, size)


def _fields(body: str) -> dict[str, str]:
    out = {}
    for part in body.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def parse_sam_output(text: str) -> list[SamRecord]:
    records = []
    for match in _RECORD.finditer(text):
        object_id, box_body, mask_body = match.groups()
        box = _fields(box_body)
        head, _, data = mask_body.partition("data=")  # data is last, and its payload may contain ';'
        parsed = _DATA.fullmatch(data)
        if parsed is None:
            continue
        height, width, payload = int(parsed[1]), int(parsed[2]), parsed[3]
        encoding = "lossless" if payload.startswith("~") else "one_bit"
        raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))
        local = np.frombuffer(raster, dtype=np.uint8).reshape(height, width).astype(bool)
        left, top = int(box["x1"]), int(box["y1"])
        right, bottom = int(box["x2"]) + 1, int(box["y2"]) + 1
        if local.shape != (bottom - top, right - left):
            local = np.array(Image.fromarray(local).resize((right - left, bottom - top), Image.NEAREST)).astype(bool)
        confidence = box.get("c") or _fields(head).get("c")
        records.append(SamRecord(object_id, (left, top, right, bottom), float(confidence) if confidence else None, local))
    return records
