"""Probe which concept phrases SAM 3.1 finds on field photos, and what an empty reply looks like."""
import json
import sys
from pathlib import Path

from openai import OpenAI

from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision.sam_parse import parse_sam_output

settings = get_settings()
client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=0, timeout=120)


def ask(image, text):
    return client.responses.create(model=settings.sam_model, input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": text}, {"type": "input_image", "image_url": image.work_data_url}]}],
        metadata={"include_confidence": "true", "mask_encoding": "one_bit"})


if sys.argv[1] == "--raw":
    image = load_image(0, Path(sys.argv[2]).read_bytes())
    dump = ask(image, sys.argv[3]).model_dump()
    print(json.dumps(dump, indent=1, default=str)[:2500])
    sys.exit()

photo, phrases = sys.argv[1], sys.argv[2:]
image = load_image(0, Path(photo).read_bytes())
w, h = image.size
for phrase in phrases:
    records = parse_sam_output(ask(image, phrase).output_text)
    cover = sum(r.mask.sum() for r in records) / (w * h) * 100
    confs = sorted((round(r.confidence, 2) for r in records if r.confidence is not None), reverse=True)
    print(f"{Path(photo).stem} | {phrase!r}: {len(records)} found, ~{cover:.1f}% of frame, conf {confs[:6]}")
