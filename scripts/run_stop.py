"""Run the stop agent on photos, in parallel like a walk, and save overlays plus JSON results.

Usage: python scripts/run_stop.py --crop soybean --out run_out_v2 field-images/IMG_4219.JPG field-images/IMG_4220.JPG
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision.sam_client import SamClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    parser.add_argument("--out", default="run_out")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()

    def one(photo: str) -> str:
        stem, started = Path(photo).stem, time.monotonic()
        try:
            image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
            result, state = run_stop(image, args.crop, llm, sam, weeds, settings)
            state.overlay().save(out / f"{stem}.png")
        except Exception as error:  # an unreadable photo or a dropped connection must not end the batch
            return f"!! {stem} failed: {type(error).__name__}: {error}"
        (out / f"{stem}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        return (f"{stem}: {result.status} in {time.monotonic() - started:.0f} s, cover {result.weed_cover_pct}%, "
                f"{result.weed_count} weeds, species {result.species}, crop seen {result.crop_seen}, "
                f"flags {result.flags}")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(one, args.photos):
            print(line, flush=True)


if __name__ == "__main__":
    main()
