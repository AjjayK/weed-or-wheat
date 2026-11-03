"""Run a walk: every stop through the agent (in threads), then the field step. Progress goes to Walk.events."""
import hashlib
import itertools
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from scout.agent.field_agent import build_report
from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.labels.lookup import load_records
from scout.models import FieldReport, StopResult
from scout.report.render import OUTCOME_TEXT
from scout.vision.sam_client import SamClient

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
_WALK_ID = re.compile(r"[0-9a-f]{16}")
STEP_NAME = re.compile(r"s[0-9]{1,2}-[0-9]{2,3}\.jpg")  # step picture file names; the route serves nothing else
STEP_MAX_PX = 1024
STEP_QUALITY = 80
WALK_VERSION = "2"  # bump when the agent or rules change, so old cached reports aren't served for new runs
log = logging.getLogger(__name__)


@dataclass
class Walk:
    id: str
    crop: str
    status: str = "running"  # running | done | failed
    events: list[dict] = field(default_factory=list)
    report: FieldReport | None = None
    error: str | None = None
    started: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def emit(self, event: dict) -> None:
        """Stops run in threads: stamping and appending together keep the list in the order of its times."""
        with self._lock:
            self.events.append({**event, "t": round(time.monotonic() - self.started, 1)})


def walk_id_for(crop: str, images: list[StopImage]) -> str:
    key = f"{WALK_VERSION}|{crop}|" + ",".join(image.hash for image in images)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _write_atomic(path: Path, text: str) -> None:
    """find_dir and replays treat these files as complete, so they appear whole or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Pipeline:
    def __init__(self, llm, sam, weeds: WeedBook, records, settings: Settings, store_dir: Path,
                 demo_dir: Path = DEMO_DIR, stop_runner=run_stop):
        self.llm = llm
        self.sam = sam
        self.weeds = weeds
        self.records = records
        self.settings = settings
        self.store_dir = Path(store_dir)
        self.demo_dir = Path(demo_dir)
        self._stop_runner = stop_runner

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pipeline":
        records = load_records()
        if not records:
            log.warning("The label table is empty: every report will say no product matches. "
                        "Build it with scout.labels.build_labels.")
        return cls(Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load(), records,
                   settings, Path(os.environ.get("SCOUT_STORE", "store")))

    def find_dir(self, walk_id: str) -> Path | None:
        if not _WALK_ID.fullmatch(walk_id):
            return None
        for base in (self.store_dir, self.demo_dir):
            folder = base / walk_id
            if (folder / "report.json").exists():
                return folder
        return None

    def cached(self, walk_id: str) -> FieldReport | None:
        folder = self.find_dir(walk_id)
        if folder is None:
            return None
        return FieldReport.model_validate_json((folder / "report.json").read_text(encoding="utf-8"))

    def saved_events(self, walk_id: str) -> list[dict] | None:
        folder = self.find_dir(walk_id)
        path = folder / "events.json" if folder else None
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def step_picture(self, walk_id: str, name: str) -> Path | None:
        """A running walk's pictures are in the store before its report exists, so this doesn't use find_dir."""
        if not _WALK_ID.fullmatch(walk_id) or not STEP_NAME.fullmatch(name):
            return None
        for base in (self.store_dir, self.demo_dir):
            path = base / walk_id / "steps" / name
            if path.is_file():
                return path
        return None

    def samples(self) -> list[dict]:
        index = self.demo_dir / "samples.json"
        if not index.exists():
            return []
        out = []
        for sample in json.loads(index.read_text(encoding="utf-8")):
            report = self.cached(sample["walk_id"])
            events = self.saved_events(sample["walk_id"]) or []
            thumbnail = next((e.get("picture") for e in events if e.get("type") == "stop_start"), None)
            decision = None
            if report is not None:
                decision = OUTCOME_TEXT[report.decision] + (f" ({report.treat_mode})" if report.treat_mode else "")
            out.append({**sample, "thumbnail": thumbnail, "decision": decision,
                        "stops": len(report.stops) if report else None})
        return out

    def run(self, walk: Walk, images: list[StopImage], names: list[str] | None = None) -> None:
        """Events are emitted before status changes, so a reader that sees a final status has every event."""
        try:
            report = self.cached(walk.id)
            fresh = report is None
            if fresh:
                out_dir = self.store_dir / walk.id
                (out_dir / "steps").mkdir(parents=True, exist_ok=True)
                names = names or [f"Photo {image.index + 1}" for image in images]
                walk.emit({"type": "walk_start", "crop": walk.crop,
                           "budget": {"calls": self.settings.max_tool_calls, "seconds": self.settings.stop_timeout_s},
                           "stops": [{"stop": image.index, "name": name} for image, name in zip(images, names)]})
                with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_stops) as pool:
                    stops = list(pool.map(lambda image: self._stop(walk, image, out_dir), images))
                if all(stop.status == "failed" for stop in stops):  # an outage or a bad key: don't cache it
                    raise RuntimeError("no stop could be analyzed")
                walk.emit({"type": "report_building"})
                report = build_report(walk.id, walk.crop, stops, self.weeds, self.records, self.llm)
                _write_atomic(out_dir / "report.json", report.model_dump_json(indent=1))
            walk.report = report
            walk.emit({"type": "report_ready"})
            if fresh:  # the saved feed is what a replay plays back
                try:
                    _write_atomic(self.store_dir / walk.id / "events.json", json.dumps(walk.events))
                except Exception:  # the report is cached: losing the replay must not fail the walk
                    log.warning("replay events could not be saved", exc_info=True)
            walk.status = "done"
        except Exception as error:
            log.exception("walk %s failed", walk.id)
            walk.error = str(error)
            walk.emit({"type": "error", "message": "The walk couldn't be analyzed. Try again."})
            walk.status = "failed"

    def _stop(self, walk: Walk, image: StopImage, out_dir: Path) -> StopResult:
        numbers = itertools.count()

        def save(picture: Image.Image) -> str:
            name = f"s{image.index}-{next(numbers):02d}.jpg"
            copy = picture.convert("RGB")
            copy.thumbnail((STEP_MAX_PX, STEP_MAX_PX))
            copy.save(out_dir / "steps" / name, format="JPEG", quality=STEP_QUALITY)
            return name

        def url(name: str | None) -> str | None:
            return f"/walks/{walk.id}/steps/{name}" if name else None

        def try_save(picture: Image.Image) -> str | None:
            try:
                return save(picture)
            except Exception:  # pictures are for people watching; a failed save must not end the stop
                log.warning("step picture could not be saved", exc_info=True)
                return None

        def on_step(step):
            if step.tool == "think":
                walk.emit({"type": "thought", "stop": image.index, "text": step.summary, "detail": step.detail})
            else:
                walk.emit({"type": "step", "stop": image.index, "tool": step.tool, "summary": step.summary,
                           "ok": step.ok, "picture": url(step.picture)})

        walk.emit({"type": "stop_start", "stop": image.index, "picture": url(try_save(image.work))})
        final = None
        try:
            result, state = self._stop_runner(image, walk.crop, self.llm, self.sam, self.weeds, self.settings,
                                              on_step=on_step, picture_sink=save)
            overlay = state.overlay()
            overlay.save(out_dir / f"stop_{image.index}.png")
            final = url(try_save(overlay))
        except Exception:
            log.exception("stop %s of walk %s failed", image.index, walk.id)
            result = StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, status="failed",
                                flags=["This stop could not be analyzed"])
            image.work.save(out_dir / f"stop_{image.index}.png")
        walk.emit({"type": "stop_done", "stop": image.index, "status": result.status,
                   "cover": result.weed_cover_pct, "weeds": result.weed_count, "picture": final})
        return result
