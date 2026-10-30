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
