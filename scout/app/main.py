"""HTTP app. Run locally: uvicorn scout.app.main:create_app --factory --port 8000"""
import asyncio
import json
import random
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scout.app.ingest import load_image
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings, get_settings
from scout.models import FieldReport
from scout.report.render import report_csv, report_html

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CROPS = {"corn", "soybean", "alfalfa"}
POLL_S = 0.3  # how often an event stream looks for new events
KEEP_ALIVE_S = 15  # a Muse Spark turn can take 15+ s; a comment this often keeps proxies from closing a quiet stream
REPLAY_SPEED = 4.0  # a saved walk replays this many times faster than it ran
REPLAY_MAX_GAP_S = 3.0  # and never waits longer than this between two events


def create_app(pipeline=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipeline = pipeline or Pipeline.from_settings(settings)
    app = FastAPI(title="Weed scout")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    walks: dict[str, Walk] = {}
    recent: dict[str, list[float]] = defaultdict(list)
    recent_all: list[float] = []
    body_limit = settings.max_photos * settings.max_upload_bytes + 1_000_000

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        # Runs before Starlette spools the multipart body to disk; browsers always send Content-Length.
        if request.method == "POST" and int(request.headers.get("content-length") or 0) > body_limit:
            return JSONResponse({"detail": "That upload is too large."}, status_code=413)
        return await call_next(request)

    def client_ip(request: Request) -> str:
        # On a Hugging Face Space request.client is the proxy; the visitor is the first X-Forwarded-For entry.
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        return forwarded or (request.client.host if request.client else "unknown")

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
        for upload in files:  # sizes are known once the form is parsed: check them all before reading any photo
            if (upload.size or 0) > settings.max_upload_bytes:
                raise HTTPException(400, f"{upload.filename} is over {settings.max_upload_bytes // 1_000_000} MB.")
        images = []
        for index, upload in enumerate(files):
            data = await upload.read()
            try:
                # In a thread: decoding takes 0.5-1.5 s per photo and would otherwise stall every SSE stream.
                images.append(await asyncio.to_thread(load_image, index, data, settings.full_max_px, settings.work_max_px))
            except Exception:
                raise HTTPException(400, f"{upload.filename} isn't a photo we can read.") from None
        if settings.offset_gps:  # public site: one random shift per walk hides the farm and keeps the pin layout
            d_lat, d_lon = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            for image in images:
                if image.gps:
                    image.gps = (round(image.gps[0] + d_lat, 6), round(image.gps[1] + d_lon, 6))
        walk_id = walk_id_for(crop, images)
        if walk_id in walks and walks[walk_id].status != "failed":
            return {"walk_id": walk_id}
        if pipeline.cached(walk_id) is not None:  # no Walk: an in-memory one would hide the saved replay for good
            return {"walk_id": walk_id}
        ip = client_ip(request)
        now = time.time()
        for key in list(recent):  # forget visitors whose last walk is over an hour old
            recent[key] = [t for t in recent[key] if now - t < 3600]
            if not recent[key]:
                del recent[key]
        recent_all[:] = [t for t in recent_all if now - t < 3600]
        if len(recent.get(ip, ())) >= settings.walks_per_ip_per_hour or len(recent_all) >= settings.walks_per_hour:
            raise HTTPException(429, "You've reached the hourly limit. Open a sample walk instead.")
        recent[ip].append(now)
        recent_all.append(now)
        walk = Walk(id=walk_id, crop=crop)
        walks[walk_id] = walk
        names = [upload.filename or f"Photo {index + 1}" for index, upload in enumerate(files)]
        threading.Thread(target=pipeline.run, args=(walk, images), kwargs={"names": names}, daemon=True).start()
        return {"walk_id": walk_id}

    @app.get("/walks/{walk_id}/events")
    async def events(walk_id: str, request: Request):
        walk = walks.get(walk_id)
        replaying = False
        if walk is None:
            if pipeline.cached(walk_id) is None:
                raise HTTPException(404, "Unknown walk.")
            saved = pipeline.saved_events(walk_id)
            replaying = bool(saved)
            events = [{**event, "replay": True} for event in saved] if saved else [{"type": "report_ready"}]
            walk = Walk(id=walk_id, crop="", status="done", events=events)
        last = request.headers.get("last-event-id", "")
        start = int(last) + 1 if last.isdigit() else 0  # EventSource sends it when it reconnects: resume, don't replay

        async def stream():
            sent, quiet = start, 0.0
            while True:
                done = walk.status != "running"  # read first: events are appended before status changes
                while sent < len(walk.events):
                    if replaying and sent > start:  # a saved walk plays back at REPLAY_SPEED, gaps capped
                        gap = walk.events[sent].get("t", 0) - walk.events[sent - 1].get("t", 0)
                        await asyncio.sleep(min(max(gap, 0) / REPLAY_SPEED, REPLAY_MAX_GAP_S))
                    yield f"id: {sent}\ndata: {json.dumps(walk.events[sent])}\n\n"
                    sent += 1
                    quiet = 0.0
                if done:
                    break
                await asyncio.sleep(POLL_S)
                quiet += POLL_S
                if quiet >= KEEP_ALIVE_S:
                    yield ": keep-alive\n\n"
                    quiet = 0.0

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/walks/{walk_id}/report", response_class=HTMLResponse)
    def report_page(walk_id: str):
        return report_html(report_for(walk_id), pipeline.weeds, lambda i: f"/walks/{walk_id}/overlay/{i}.png",
                           lambda name: f"/walks/{walk_id}/steps/{name}")

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

    @app.get("/walks/{walk_id}/steps/{name}")
    def step_picture(walk_id: str, name: str):
        path = pipeline.step_picture(walk_id, name)
        if path is None:
            raise HTTPException(404, "No such picture.")
        return FileResponse(path, media_type="image/jpeg")

    return app
