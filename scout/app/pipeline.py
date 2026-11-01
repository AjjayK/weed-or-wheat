"""Run a walk: every stop through the agent (in threads), then the field step. Progress goes to Walk.events."""
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from scout.agent.field_agent import build_report
from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.labels.lookup import load_records
from scout.models import FieldReport, StopResult
from scout.vision.sam_client import SamClient

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
_WALK_ID = re.compile(r"[0-9a-f]{16}")
log = logging.getLogger(__name__)


@dataclass
class Walk:
    id: str
    crop: str
    status: str = "running"  # running | done | failed
    events: list[dict] = field(default_factory=list)
    report: FieldReport | None = None
    error: str | None = None

    def emit(self, event: dict) -> None:
        self.events.append(event)


def walk_id_for(crop: str, images: list[StopImage]) -> str:
    key = crop + "|" + ",".join(image.hash for image in images)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


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
        return cls(Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load(), load_records(),
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

    def samples(self) -> list[dict]:
        index = self.demo_dir / "samples.json"
        return json.loads(index.read_text(encoding="utf-8")) if index.exists() else []

    def run(self, walk: Walk, images: list[StopImage]) -> None:
        """Events are emitted before status changes, so a reader that sees a final status has every event."""
        try:
            report = self.cached(walk.id)
            if report is None:
                out_dir = self.store_dir / walk.id
                out_dir.mkdir(parents=True, exist_ok=True)
                with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_stops) as pool:
                    stops = list(pool.map(lambda image: self._stop(walk, image, out_dir), images))
                report = build_report(walk.id, walk.crop, stops, self.weeds, self.records, self.llm)
                (out_dir / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
            walk.report = report
            walk.emit({"type": "report_ready"})
            walk.status = "done"
        except Exception as error:
            log.exception("walk %s failed", walk.id)
            walk.error = str(error)
            walk.emit({"type": "error", "message": "The walk couldn't be analyzed. Try again."})
            walk.status = "failed"

    def _stop(self, walk: Walk, image: StopImage, out_dir: Path) -> StopResult:
        def on_step(step):
            walk.emit({"type": "step", "stop": image.index, "tool": step.tool, "summary": step.summary, "ok": step.ok})

        try:
            result, state = self._stop_runner(image, walk.crop, self.llm, self.sam, self.weeds, self.settings,
                                              on_step=on_step)
            state.overlay().save(out_dir / f"stop_{image.index}.png")
        except Exception:
            log.exception("stop %s of walk %s failed", image.index, walk.id)
            result = StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, status="failed",
                                flags=["This stop could not be analyzed"])
            image.work.save(out_dir / f"stop_{image.index}.png")
        walk.emit({"type": "stop_done", "stop": image.index, "status": result.status})
        return result
