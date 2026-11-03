import io
import json
import threading

from fastapi.testclient import TestClient
from PIL import Image

from scout.app import main as main_module
from scout.app.main import create_app
from scout.app.pipeline import STEP_NAME
from scout.config import Settings
from scout.models import FieldReport
from tests.helpers import png_bytes, small_weedbook


class FakePipeline:
    def __init__(self, root):
        self.weeds = small_weedbook()
        self.root = root

    def find_dir(self, walk_id):
        folder = self.root / walk_id
        return folder if (folder / "report.json").exists() else None

    def cached(self, walk_id):
        folder = self.find_dir(walk_id)
        return FieldReport.model_validate_json((folder / "report.json").read_text()) if folder else None

    def samples(self):
        return [{"name": "Sample", "walk_id": "0123456789abcdef"}]

    def saved_events(self, walk_id):
        folder = self.find_dir(walk_id)
        path = folder / "events.json" if folder else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def step_picture(self, walk_id, name):
        path = self.root / walk_id / "steps" / name
        return path if STEP_NAME.fullmatch(name) and path.is_file() else None

    def run(self, walk, images, names=None):
        self.names = names
        report = FieldReport(walk_id=walk.id, crop=walk.crop, stops=[], species_summary=[], decision="no_action",
                             rule_trace=["No weeds found at any stop -> no action"],
                             explanation="Nothing to do. Read the label before use.", options=[])
        folder = self.root / walk.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(report.model_dump_json())
        walk.emit({"type": "step", "stop": 0, "tool": "segment", "summary": "fake", "ok": True})
        walk.report = report
        walk.emit({"type": "report_ready"})
        walk.status = "done"


def _client(tmp_path, pipeline=None, **settings):
    return TestClient(create_app(pipeline or FakePipeline(tmp_path), Settings(api_key="t", **settings)))


def _upload(client, n=1, crop="soybean", shade=0, headers=None):
    files = [("files", (f"p{i}.png", png_bytes(40, 40, (i, 100, shade)), "image/png")) for i in range(n)]
    return client.post("/walks", data={"crop": crop}, files=files, headers=headers)


def _jpeg_with_gps(lat_minutes, color):
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (41.0, lat_minutes, 0.0), 3: "W", 4: (93.0, 36.0, 0.0)}  # 41 + m/60 N, 93.6 W
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), color).save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _upload_gps_photos(client):
    files = [("files", (f"p{i}.jpg", _jpeg_with_gps(30.0 + i, (i * 40, 90, 60)), "image/jpeg")) for i in range(2)]
    walk_id = client.post("/walks", data={"crop": "soybean"}, files=files).json()["walk_id"]
    client.get(f"/walks/{walk_id}/events")  # returns once the walk has run


class GpsCapture(FakePipeline):
    def run(self, walk, images, names=None):
        self.gps = [image.gps for image in images]
        super().run(walk, images, names)


def test_walk_flow(tmp_path):
    client = _client(tmp_path)
    walk_id = _upload(client).json()["walk_id"]
    body = client.get(f"/walks/{walk_id}/events").text
    assert '"type": "step"' in body and "report_ready" in body
    assert "Read the label before use" in client.get(f"/walks/{walk_id}/report").text
    assert client.get(f"/walks/{walk_id}/stops.csv").headers["content-type"].startswith("text/csv")
    assert client.get(f"/walks/{walk_id}/report.json").json()["decision"] == "no_action"


def test_rejects_bad_crop_and_too_many_photos(tmp_path):
    client = _client(tmp_path)
    assert _upload(client, crop="rice").status_code == 400
    assert _upload(client, n=11).status_code == 400


def test_rate_limit_per_ip(tmp_path):
    client = _client(tmp_path, walks_per_ip_per_hour=1)
    assert _upload(client, shade=1).status_code == 200
    assert _upload(client, shade=2).status_code == 429


def test_rate_limit_counts_the_visitor_behind_the_proxy(tmp_path):
    client = _client(tmp_path, walks_per_ip_per_hour=1)
    assert _upload(client, shade=1, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 200
    assert _upload(client, shade=2, headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}).status_code == 200
    assert _upload(client, shade=3, headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}).status_code == 429


def test_site_wide_hourly_cap(tmp_path):
    client = _client(tmp_path, walks_per_hour=1)
    assert _upload(client, shade=1, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert _upload(client, shade=2, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_oversize_photo_is_rejected_before_any_photo_is_decoded(tmp_path, monkeypatch):
    decoded = []
    monkeypatch.setattr("scout.app.main.load_image", lambda *args: decoded.append(args))
    client = _client(tmp_path, max_upload_bytes=500)
    files = [("files", ("ok.png", png_bytes(40, 40), "image/png")), ("files", ("big.png", b"x" * 600, "image/png"))]
    response = client.post("/walks", data={"crop": "soybean"}, files=files)
    assert response.status_code == 400 and response.json()["detail"].startswith("big.png is over")
    assert decoded == []


def test_body_over_the_limit_is_413(tmp_path):
    client = _client(tmp_path, max_upload_bytes=100, max_photos=1)  # limit: 100 bytes of photo + 1 MB of form
    files = [("files", ("p.png", b"x" * 1_100_000, "image/png"))]
    response = client.post("/walks", data={"crop": "soybean"}, files=files)
    assert response.status_code == 413 and response.json() == {"detail": "That upload is too large."}


def test_events_resume_from_last_event_id(tmp_path):
    client = _client(tmp_path)
    walk_id = _upload(client).json()["walk_id"]
    full = client.get(f"/walks/{walk_id}/events")
    assert "id: 0\n" in full.text and "id: 1\n" in full.text
    assert full.headers["cache-control"] == "no-cache" and full.headers["x-accel-buffering"] == "no"
    resumed = client.get(f"/walks/{walk_id}/events", headers={"Last-Event-ID": "0"}).text
    assert '"type": "step"' not in resumed and "report_ready" in resumed


def test_idle_stream_sends_keep_alive_comments(tmp_path, monkeypatch):
    release = threading.Event()

    class Slow(FakePipeline):
        def run(self, walk, images, names=None):
            release.wait(5)
            super().run(walk, images, names)

    monkeypatch.setattr("scout.app.main.KEEP_ALIVE_S", 0.3)
    client = _client(tmp_path, pipeline=Slow(tmp_path))
    walk_id = _upload(client).json()["walk_id"]
    threading.Timer(0.8, release.set).start()
    body = client.get(f"/walks/{walk_id}/events").text
    assert ": keep-alive\n\n" in body and "report_ready" in body


def test_gps_is_shifted_once_per_walk(tmp_path):
    pipeline = GpsCapture(tmp_path)
    _upload_gps_photos(_client(tmp_path, pipeline=pipeline))
    (lat_a, lon_a), (lat_b, lon_b) = pipeline.gps
    assert (lat_a, lon_a) != (41.5, -93.6)
    assert abs(lat_a - 41.5) <= 0.500001 and abs(lon_a + 93.6) <= 0.500001
    assert round(lat_b - lat_a, 4) == round(1 / 60, 4) and round(lon_b - lon_a, 6) == 0  # pin layout kept


def test_gps_is_kept_when_the_offset_is_off(tmp_path):
    pipeline = GpsCapture(tmp_path)
    _upload_gps_photos(_client(tmp_path, pipeline=pipeline, offset_gps=False))
    assert pipeline.gps == [(41.5, -93.6), (41.516667, -93.6)]


def test_unknown_walk_is_404_and_samples_listed(tmp_path):
    client = _client(tmp_path)
    assert client.get("/walks/ffffffffffffffff/report").status_code == 404
    assert client.get("/walks/ffffffffffffffff/overlay/0.png").status_code == 404
    assert client.get("/samples").json()[0]["name"] == "Sample"


def test_page_has_crop_cards_drop_zone_and_live_view(tmp_path):
    html = _client(tmp_path).get("/").text
    for crop in ("corn", "soybean", "alfalfa"):
        assert f'<input type="radio" name="crop" value="{crop}" required>' in html
    assert "checked" not in html
    for part in ('id="drop-zone"', 'id="live-view"', 'id="strip"', 'id="stage-img"', 'aria-live="polite"',
                 'id="checklist"', "Scout this field"):
        assert part in html


def _saved_walk(root, walk_id, events):
    folder = root / walk_id
    (folder / "steps").mkdir(parents=True)
    report = FieldReport(walk_id=walk_id, crop="corn", stops=[], species_summary=[], decision="no_action",
                         rule_trace=[], explanation="x", options=[])
    (folder / "report.json").write_text(report.model_dump_json())
    if events is not None:
        (folder / "events.json").write_text(json.dumps(events))
    return folder


def _event_data(text):
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_upload_passes_photo_names_to_the_pipeline(tmp_path):
    pipeline = FakePipeline(tmp_path)
    client = _client(tmp_path, pipeline)
    walk_id = _upload(client, n=2).json()["walk_id"]
    client.get(f"/walks/{walk_id}/events")
    assert pipeline.names == ["p0.png", "p1.png"]


def test_step_pictures_are_served_by_valid_name_only(tmp_path):
    folder = _saved_walk(tmp_path, "0123456789abcdef", None)
    (folder / "steps" / "s0-01.jpg").write_bytes(b"jpeg")
    client = _client(tmp_path)
    ok = client.get("/walks/0123456789abcdef/steps/s0-01.jpg")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg" and ok.content == b"jpeg"
    for bad in ("s0-02.jpg", "report.json", "..%2Freport.json"):
        assert client.get(f"/walks/0123456789abcdef/steps/{bad}").status_code == 404
    assert client.get("/walks/ffffffffffffffff/steps/s0-01.jpg").status_code == 404


def test_a_saved_walk_replays_its_events(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "REPLAY_SPEED", 1e9)
    events = [{"type": "walk_start", "crop": "corn", "stops": [{"stop": 0, "name": "a.jpg"}], "t": 0.0},
              {"type": "stop_start", "stop": 0, "picture": None, "t": 0.5},
              {"type": "report_ready", "t": 40.0}]
    _saved_walk(tmp_path, "0123456789abcdef", events)
    replayed = _event_data(_client(tmp_path).get("/walks/0123456789abcdef/events").text)
    assert [e["type"] for e in replayed] == ["walk_start", "stop_start", "report_ready"]
    assert all(e["replay"] is True for e in replayed)


def test_a_saved_walk_without_events_just_reports_ready(tmp_path):
    _saved_walk(tmp_path, "0123456789abcdef", None)
    replayed = _event_data(_client(tmp_path).get("/walks/0123456789abcdef/events").text)
    assert replayed == [{"type": "report_ready"}]


def test_replay_waits_the_gap_over_the_speed_capped_and_never_negative(tmp_path, monkeypatch):
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)  # a replayed walk is already done, so these are all replay pacing, no POLL_S polls

    monkeypatch.setattr(main_module, "REPLAY_SPEED", 4.0)
    monkeypatch.setattr(main_module, "REPLAY_MAX_GAP_S", 3.0)
    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)
    _saved_walk(tmp_path, "0123456789abcdef", [{"type": "thought", "t": t} for t in (0.0, 8.0, 7.9, 30.0)])
    replayed = _event_data(_client(tmp_path).get("/walks/0123456789abcdef/events").text)
    assert len(replayed) == 4
    assert waits == [2.0, 0, 3.0]


def test_uploading_a_saved_walk_again_keeps_its_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "REPLAY_SPEED", 1e9)
    client = _client(tmp_path)
    walk_id = _upload(client).json()["walk_id"]
    client.get(f"/walks/{walk_id}/events")  # waits for the first run, which saves the report
    events = [{"type": "walk_start", "crop": "soybean", "stops": [], "t": 0.0}, {"type": "report_ready", "t": 9.0}]
    (tmp_path / walk_id / "events.json").write_text(json.dumps(events))
    later = _client(tmp_path)  # a later server: the walk is saved, not in memory
    assert _upload(later).json()["walk_id"] == walk_id
    replayed = _event_data(later.get(f"/walks/{walk_id}/events").text)
    assert [e["type"] for e in replayed] == ["walk_start", "report_ready"] and all(e["replay"] for e in replayed)
