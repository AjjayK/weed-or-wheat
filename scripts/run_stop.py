"""Run the stop agent on photos, in parallel like a walk, and save overlays plus JSON results.

Steps print live; each photo's summary prints when it finishes; the run ends with the total time and the crop warning
the report would show. Nothing here is cached: every run calls the live API.

Usage (from the repo root; PYTHONPATH=. is needed so the scout package is found):
  PYTHONPATH=. .venv/Scripts/python scripts/run_stop.py --crop soybean --out run_out_v2 \\
      field-images/IMG_4219.JPG field-images/IMG_4220.JPG
"""
import argparse
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scout.agent.field_agent import crop_mismatch
from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.labels.lookup import adopt_label_species, load_records
from scout.models import StopResult
from scout.vision.sam_client import SamClient


def describe(stem: str, seconds: float, result: StopResult, state) -> str:
    """Everything the live check needs to judge one stop, as lines for one print."""
    plants, areas = state.weeds_marked()
    lines = [f"{stem}: {result.status} in {seconds:.0f} s, cover {result.weed_cover_pct}%, "
             f"{result.weed_count} weeds, species {result.species}",
             f"  crop seen {result.crop_seen}, weeds estimated {result.weeds_estimate}, "
             f"marked {plants} boxed + {areas} weed areas, agent says all marked: {result.all_weeds_marked}",
             f"  tool calls {state.calls}, finished {state.finished}, flags {result.flags}"]
    if result.status == "failed":
        lines.append(f"  last step: {result.steps[-1].summary if result.steps else 'none'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    parser.add_argument("--out", default="run_out")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--steps", action="store_true",
                        help="save every step's picture (marks, zooms, identify's close-ups) to OUT/STEM-steps/")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()
    adopt_label_species(load_records(), weeds)  # the agent sees the labels' weeds, as in the web app
    started = time.monotonic()

    def one(photo: str):
        stem, began = Path(photo).stem, time.monotonic()
        try:
            image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
            sink = None
            if args.steps:
                folder, numbers = out / f"{stem}-steps", itertools.count()
                folder.mkdir(exist_ok=True)

                def sink(picture, folder=folder, numbers=numbers):
                    name = f"{next(numbers):02d}.jpg"
                    picture.convert("RGB").save(folder / name, quality=85)
                    return name
            result, state = run_stop(
                image, args.crop, llm, sam, weeds, settings, picture_sink=sink,
                on_step=lambda s, stem=stem: print(f"  [{stem}] {s.tool}: {s.summary}"
                                                   + (f" ({s.picture})" if s.picture else ""), flush=True))
            state.overlay().save(out / f"{stem}.png")
            (out / f"{stem}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        except Exception as error:  # an unreadable photo or a dropped connection must not end the batch
            return stem, time.monotonic() - began, None, None, f"{type(error).__name__}: {error}"
        return stem, time.monotonic() - began, result, state, None

    results: list[StopResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(one, photo) for photo in args.photos]):
            stem, seconds, result, state, error = future.result()
            if result is None:
                print(f"!! {stem} failed after {seconds:.0f} s: {error}", flush=True)
                continue
            results.append(result)
            print(describe(stem, seconds, result, state), flush=True)
    print(f"{len(results)} of {len(args.photos)} photos finished in {time.monotonic() - started:.0f} s", flush=True)
    print("Report banner: " + (crop_mismatch(args.crop, results) or "no crop warning"), flush=True)


if __name__ == "__main__":
    main()
