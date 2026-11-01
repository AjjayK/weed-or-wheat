import io
import threading

from fastapi.testclient import TestClient
from PIL import Image

from scout.app.main import create_app
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

    def run(self, walk, images):
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
    def run(self, walk, images):
        self.gps = [image.gps for image in images]
        super().run(walk, images)


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
        def run(self, walk, images):
            release.wait(5)
            super().run(walk, images)

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
