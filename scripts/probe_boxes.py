"""Ask Muse Spark for weed boxes (0-1000), then segment all boxes in ONE multi-object SAM request."""
import json
import sys
import time
from pathlib import Path

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt
from PIL import ImageDraw

from scout.agent.llm import Llm, user_message
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.sam_client import SamClient

BOXES = {"type": "json_schema", "name": "weed_boxes", "strict": True, "schema": {
    "type": "object", "properties": {"weeds": {"type": "array", "items": {"type": "object", "properties": {
        "box": {"type": "array", "items": {"type": "integer"}}, "what": {"type": "string"}},
        "required": ["box", "what"], "additionalProperties": False}}},
    "required": ["weeds"], "additionalProperties": False}}
PROMPT = ("Find every weed plant in this crop field photo; ignore the crop plants. For each weed give a tight box "
          "[x1, y1, x2, y2] with coordinates from 0 to 1000 (0,0 top-left, 1000,1000 bottom-right) around just that "
          "weed, and say what it is in two or three words. If weeds form a dense patch, box each distinct plant up to "
          "16 boxes, largest first.")

settings = get_settings()
llm, sam = Llm.from_settings(settings), SamClient.from_settings(settings)
for photo in sys.argv[1:]:
    image = load_image(0, Path(photo).read_bytes())
    w, h = image.size
    t0 = time.time()
    weeds = json.loads(llm.turn([user_message(PROMPT, [image.work_data_url])], text_format=BOXES).text)["weeds"][:16]
    boxes = []
    for wd in weeds:
        x1, y1, x2, y2 = (wd["box"] + [0, 0, 0, 0])[:4]
        l, t, r, b = round(x1 * w / 1000), round(y1 * h / 1000), round(x2 * w / 1000), round(y2 * h / 1000)
        l, r = sorted((max(0, min(w - 1, l)), max(1, min(w, r))))
        t, b = sorted((max(0, min(h - 1, t)), max(1, min(h, b))))
        if r - l >= 2 and b - t >= 2:
            boxes.append((l, t, r, b))
    t1 = time.time()
    text = build_object_prompt(size=image.size, objects=[PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=PromptBox(*bx))])
                                                       for i, bx in enumerate(boxes)])
    records = sam._records(image, text)
    t2 = time.time()
    weed = mk.union([r.full_mask(image.size) for r in records], image.size)
    out = render_overlay(image.work, weed & False, weed)
    d = ImageDraw.Draw(out)
    for l, t, r, b in boxes:
        d.rectangle((l, t, r - 1, b - 1), outline=(255, 214, 0), width=2)
    out.save(f"spike_out/{Path(photo).stem}__boxes.jpg", quality=85)
    print(f"{Path(photo).stem}: {len(boxes)} boxes ({t1 - t0:.1f}s), SAM one request -> {len(records)} records "
          f"ids {[r.object_id for r in records][:16]} ({t2 - t1:.1f}s), weed cover {weed.mean() * 100:.1f}%, "
          f"what={[wd['what'] for wd in weeds][:8]}")
