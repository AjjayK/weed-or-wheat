"""Run the stop agent on photos and save overlays plus JSON results.

Usage: python scripts/run_stop.py --crop soybean field-images/IMG_4219.JPG field-images/IMG_4220.JPG
"""
import argparse
from pathlib import Path

from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision.sam_client import SamClient

OUT = Path("run_out")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()
    for photo in args.photos:
        stem = Path(photo).stem
        print(f"== {stem}")
        try:
            image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
            result, state = run_stop(image, args.crop, llm, sam, weeds, settings,
                                     on_step=lambda s: print(f"  [{s.tool}] {s.summary}"))
            state.overlay().save(OUT / f"{stem}.png")
        except Exception as error:  # an unreadable photo or a dropped connection must not end the batch
            print(f"  !! {stem} failed: {type(error).__name__}: {error}")
            continue
        (OUT / f"{stem}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        print(f"  -> {result.status}: cover {result.weed_cover_pct}%, {result.weed_count} weeds, "
              f"species {result.species}, flags {len(result.flags)}")


if __name__ == "__main__":
    main()
