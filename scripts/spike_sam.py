"""Day-1 test: run SAM 3.1 concept prompts on photos and save overlays.

Usage:
  python scripts/spike_sam.py --crop-prompt "soybean plant" field-images/IMG_4219.JPG field-images/IMG_3926.JPG
Writes spike_out/<photo>__<prompt>.png and spike_out/<photo>__candidates.png, and prints timings.
"""
import argparse
import re
import time
from pathlib import Path

from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.sam_client import SamClient
from scout.vision.sam_parse import parse_sam_output

OUT = Path("spike_out")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop-prompt", required=True)
    parser.add_argument("--extra", nargs="*", default=["grass", "sky"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    settings = get_settings()
    sam = SamClient.from_settings(settings)
    for photo in args.photos:
        image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
        stem = Path(photo).stem
        unions = {}
        for prompt in [args.crop_prompt, "plant", *args.extra]:
            started = time.time()
            raw = sam.fetch_raw(image, prompt)
            fetched = time.time()
            records = parse_sam_output(raw)
            decoded = time.time()
            unions[prompt] = mk.union([r.full_mask(image.size) for r in records], image.size)
            mismatched = sum(r.resized for r in records)
            confs = [round(r.confidence, 2) for r in records if r.confidence is not None]
            print(f"{stem} | {prompt!r}: {len(records)} found, api {fetched - started:.1f}s, "
                  f"decode {decoded - fetched:.1f}s, conf {confs[:8]}, mask/box size mismatches {mismatched}")
            render_overlay(image.work, unions[prompt], unions[prompt] & False).save(OUT / f"{stem}__{slug(prompt)}.png")
        crop = unions[args.crop_prompt]
        candidates = mk.pieces(unions["plant"], 0.0005 * image.size[0] * image.size[1], exclude=crop)
        weed = mk.union(candidates, image.size)
        print(f"{stem} | plant minus crop: {len(candidates)} weed candidates")
        render_overlay(image.work, crop, weed).save(OUT / f"{stem}__candidates.png")


if __name__ == "__main__":
    main()
