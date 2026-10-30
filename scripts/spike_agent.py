"""Day-1 test: Muse Spark tool calling, an image in a follow-up user message, structured output, and pointing.

Usage: python scripts/spike_agent.py field-images/IMG_4215.JPG [photos to test pointing on ...]
Pointing writes spike_out/<photo>__pointing_<pixel|norm1000>.jpg: SAM masks (magenta) from Muse Spark's points (yellow).
"""
import json
import sys
import time
from pathlib import Path

from PIL import ImageDraw

from scout.agent.llm import Llm, user_message
from scout.app.ingest import load_image, to_data_url
from scout.config import get_settings
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.sam_client import SamClient

TOOL = {
    "type": "function",
    "name": "report_rows",
    "description": "Report how many crop rows are visible.",
    "parameters": {"type": "object", "properties": {"count": {"type": "integer"}, "notes": {"type": "string"}},
                   "required": ["count"], "additionalProperties": False},
}
SCHEMA = {
    "type": "json_schema", "name": "plants", "strict": True,
    "schema": {"type": "object", "properties": {"crop": {"type": "string"}, "weeds_visible": {"type": "boolean"}},
               "required": ["crop", "weeds_visible"], "additionalProperties": False},
}
POINTS = {
    "type": "json_schema", "name": "weed_points", "strict": True,
    "schema": {"type": "object", "properties": {"weeds": {"type": "array", "items": {
        "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "what": {"type": "string"}},
        "required": ["x", "y", "what"], "additionalProperties": False}}},
        "required": ["weeds"], "additionalProperties": False},
}


def pointing(llm: Llm, sam: SamClient, photo: str) -> None:
    """Ask for one point per weed in two coordinate conventions, segment each point with SAM, and save overlays."""
    image = load_image(0, Path(photo).read_bytes())
    width, height = image.size
    asks = {
        "pixel": (f"This field photo is {width}x{height} pixels. Give one point (x, y in pixels from the top-left "
                  "corner) on each weed plant you can see, not on the crop. Put each point on the plant's leaves.",
                  1.0, 1.0),
        "norm1000": ("Give one point on each weed plant you can see, not on the crop, as x and y from 0 to 1000 "
                     "(0,0 is the top-left corner, 1000,1000 the bottom-right). Put each point on the plant's leaves.",
                     width / 1000, height / 1000),
    }
    for name, (prompt, sx, sy) in asks.items():
        started = time.time()
        weeds = json.loads(llm.turn([user_message(prompt, [image.work_data_url])], text_format=POINTS).text)["weeds"]
        points = [(min(width - 1, max(0, round(w["x"] * sx))), min(height - 1, max(0, round(w["y"] * sy))))
                  for w in weeds]
        masks = [r.full_mask(image.size) for x, y in points for r in sam.segment_at(image, points=[(x, y, True)])]
        weed = mk.union(masks, image.size)
        out = render_overlay(image.work, weed & False, weed)
        draw = ImageDraw.Draw(out)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 214, 0), width=3)
        Path("spike_out").mkdir(exist_ok=True)
        path = f"spike_out/{Path(photo).stem}__pointing_{name}.jpg"
        out.save(path, quality=85)
        print(f"4-{name}) {time.time() - started:.1f}s {len(points)} points {points[:12]} "
              f"what={[w['what'] for w in weeds][:6]} -> {path}")


def main(photo: str, pointing_photos: list[str]) -> None:
    llm = Llm.from_settings(get_settings())
    image = load_image(0, Path(photo).read_bytes())
    started = time.time()
    first = llm.turn([user_message("How many crop rows are visible? Use the tool.", [image.work_data_url])], tools=[TOOL])
    print(f"1) {time.time() - started:.1f}s calls={[(c.name, c.arguments) for c in first.calls]} text={first.text[:200]!r}")
    if not first.calls:
        sys.exit("No tool call came back. Check the model id and tool format before continuing.")
    outputs = [{"type": "function_call_output", "call_id": c.call_id, "output": "recorded"} for c in first.calls]
    corner = image.full.crop((0, 0, image.full.width // 2, image.full.height // 2))
    started = time.time()
    second = llm.turn(outputs + [user_message("Close-up of the top-left quarter. Describe the plants in one sentence.",
                                              [to_data_url(corner)])],
                      tools=[TOOL], previous_response_id=first.response_id)
    print(f"2) {time.time() - started:.1f}s follow-up with image: {second.text[:300]!r}")
    third = llm.turn([user_message("What crop is this and are weeds visible?", [image.work_data_url])], text_format=SCHEMA)
    print(f"3) structured output: {json.loads(third.text)}")
    sam = SamClient.from_settings(get_settings())
    for other in pointing_photos:
        pointing(llm, sam, other)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
