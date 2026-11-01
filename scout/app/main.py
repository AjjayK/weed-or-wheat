"""HTTP app. Run locally: uvicorn scout.app.main:create_app --factory --port 8000"""
import asyncio
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scout.app.ingest import load_image
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings, get_settings
from scout.models import FieldReport
from scout.report.render import report_csv, report_html

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CROPS = {"corn", "soybean", "alfalfa"}


def create_app(pipeline=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipeline = pipeline or Pipeline.from_settings(settings)
    app = FastAPI(title="Weed scout")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    walks: dict[str, Walk] = {}
    recent: dict[str, list[float]] = defaultdict(list)

    def report_for(walk_id: str) -> FieldReport:
        walk = walks.get(walk_id)
        report = walk.report if walk and walk.report else pipeline.cached(walk_id)
        if report is None:
            raise HTTPException(404, "No report for that walk.")
        return report

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/samples")
    def samples():
        return pipeline.samples()

    @app.post("/walks")
    async def create_walk(request: Request, crop: str = Form(...), files: list[UploadFile] = File(...)):
        if crop not in CROPS:
            raise HTTPException(400, "Pick corn, soybean, or alfalfa.")
        if not 1 <= len(files) <= settings.max_photos:
            raise HTTPException(400, f"Upload 1 to {settings.max_photos} photos.")
        images = []
        for index, upload in enumerate(files):
            data = await upload.read()
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(400, f"{upload.filename} is over {settings.max_upload_bytes // 1_000_000} MB.")
            try:
                # In a thread: decoding takes 0.5-1.5 s per photo and would otherwise stall every SSE stream.
                images.append(await asyncio.to_thread(load_image, index, data, settings.full_max_px, settings.work_max_px))
            except Exception:
                raise HTTPException(400, f"{upload.filename} isn't a photo we can read.") from None
        walk_id = walk_id_for(crop, images)
        if walk_id in walks and walks[walk_id].status != "failed":
            return {"walk_id": walk_id}
        if pipeline.cached(walk_id) is None:
            ip = request.client.host if request.client else "unknown"
            now = time.time()
            recent[ip] = [t for t in recent[ip] if now - t < 3600]
            if len(recent[ip]) >= settings.walks_per_ip_per_hour:
                raise HTTPException(429, "You've reached the hourly limit. Open a sample walk instead.")
            recent[ip].append(now)
        walk = Walk(id=walk_id, crop=crop)
        walks[walk_id] = walk
        threading.Thread(target=pipeline.run, args=(walk, images), daemon=True).start()
        return {"walk_id": walk_id}

    @app.get("/walks/{walk_id}/events")
    async def events(walk_id: str):
        walk = walks.get(walk_id)
        if walk is None:
            if pipeline.cached(walk_id) is None:
                raise HTTPException(404, "Unknown walk.")
            walk = Walk(id=walk_id, crop="", status="done", events=[{"type": "report_ready"}])

        async def stream():
            sent = 0
            while True:
                done = walk.status != "running"  # read first: events are appended before status changes
                while sent < len(walk.events):
                    yield f"data: {json.dumps(walk.events[sent])}\n\n"
                    sent += 1
                if done:
                    break
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/walks/{walk_id}/report", response_class=HTMLResponse)
    def report_page(walk_id: str):
        return report_html(report_for(walk_id), pipeline.weeds, lambda i: f"/walks/{walk_id}/overlay/{i}.png")

    @app.get("/walks/{walk_id}/report.json")
    def report_json(walk_id: str):
        return report_for(walk_id).model_dump(mode="json")

    @app.get("/walks/{walk_id}/stops.csv")
    def stops_csv(walk_id: str):
        return Response(report_csv(report_for(walk_id)), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="walk-{walk_id}.csv"'})

    @app.get("/walks/{walk_id}/overlay/{index}.png")
    def overlay(walk_id: str, index: int):
        folder = pipeline.find_dir(walk_id)
        path = folder / f"stop_{index}.png" if folder else None
        if path is None or not path.exists():
            raise HTTPException(404, "No such overlay.")
        return FileResponse(path, media_type="image/png")

    return app
