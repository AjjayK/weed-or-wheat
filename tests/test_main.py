from fastapi.testclient import TestClient

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


def _client(tmp_path, **settings):
    return TestClient(create_app(FakePipeline(tmp_path), Settings(api_key="t", **settings)))


def _upload(client, n=1, crop="soybean", shade=0):
    files = [("files", (f"p{i}.png", png_bytes(40, 40, (i, 100, shade)), "image/png")) for i in range(n)]
    return client.post("/walks", data={"crop": crop}, files=files)


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


def test_unknown_walk_is_404_and_samples_listed(tmp_path):
    client = _client(tmp_path)
    assert client.get("/walks/ffffffffffffffff/report").status_code == 404
    assert client.get("/walks/ffffffffffffffff/overlay/0.png").status_code == 404
    assert client.get("/samples").json()[0]["name"] == "Sample"
