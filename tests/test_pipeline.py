import hashlib
import json
import logging
import threading
from itertools import count
from types import SimpleNamespace

from PIL import Image

from scout.agent.state import StopState
from scout.app import pipeline as pipeline_module
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings
from scout.models import StopResult
from tests.helpers import FakeLlm, make_image, small_weedbook


def fake_runner(image, crop, llm, sam, weeds, settings, on_step=None, picture_sink=None):
    state = StopState(image, crop, on_step=on_step, picture_sink=picture_sink)
    state.log("think", "Looking over the rows", detail=["Counting weeds"])
    state.log("mark", "Mark weed: 1 of 1 boxes kept", picture=image.work)
    return StopResult(stop_index=image.index, image_hash=image.hash, species={"waterhemp": 1},
                      species_cover_pct={"waterhemp": 1.0}, weed_cover_pct=1.0, weed_count=1), state


def _pipeline(tmp_path, runner):
    return Pipeline(FakeLlm([]), None, small_weedbook(), [], Settings(api_key="t"), tmp_path / "store",
                    demo_dir=tmp_path / "demo", stop_runner=runner)


def test_run_writes_report_and_overlays(tmp_path):
    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    assert walk.status == "done"
    kinds = [e["type"] for e in walk.events]
    assert kinds.count("stop_done") == 2 and kinds.count("step") == 2 and kinds[-1] == "report_ready"
    folder = pipeline.find_dir(walk.id)
    assert (folder / "report.json").exists() and (folder / "stop_1.png").exists()
    assert not (folder / "report.json.tmp").exists()
    assert pipeline.cached(walk.id).walk_id == walk.id


def test_walk_where_every_stop_fails_is_failed_and_not_cached(tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, boom)
    walk = Walk(id=walk_id_for("corn", images), crop="corn")
    pipeline.run(walk, images)
    assert walk.status == "failed" and walk.events[-1]["type"] == "error"
    assert pipeline.cached(walk.id) is None  # a retry must run the stops again, not replay the outage
    assert not (tmp_path / "store" / walk.id / "events.json").exists()


def test_failed_stop_does_not_sink_walk(tmp_path):
    def flaky(image, *args, **kwargs):
        if image.index == 1:
            raise RuntimeError("api down")
        return fake_runner(image, *args, **kwargs)

    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, flaky)
    walk = Walk(id=walk_id_for("corn", images), crop="corn")
    pipeline.run(walk, images)
    assert walk.status == "done" and [s.status for s in walk.report.stops] == ["ok", "failed"]
    assert pipeline.cached(walk.id) is not None
    done = {e["stop"]: e for e in walk.events if e["type"] == "stop_done"}
    assert done[1]["status"] == "failed" and done[1]["picture"] is None
    assert done[0]["status"] == "ok" and done[0]["picture"] is not None


def test_cache_hit_does_not_run_the_stops(tmp_path):
    images = [make_image(index=0)]
    walk_id = walk_id_for("corn", images)
    _pipeline(tmp_path, fake_runner).run(Walk(id=walk_id, crop="corn"), images)

    def never(*args, **kwargs):
        raise AssertionError("stop runner called on a cache hit")

    again = Walk(id=walk_id, crop="corn")
    _pipeline(tmp_path, never).run(again, images)
    assert again.status == "done" and again.report.walk_id == walk_id


def test_walk_id_is_stable_and_hex():
    images = [make_image()]
    assert walk_id_for("corn", images) == walk_id_for("corn", images)
    assert walk_id_for("corn", images) != walk_id_for("soybean", images)
    assert len(walk_id_for("corn", images)) == 16


def test_walk_id_changes_with_the_agent_version(monkeypatch):
    images = [make_image()]
    old_key = "soybean|" + ",".join(image.hash for image in images)
    unversioned = hashlib.sha256(old_key.encode()).hexdigest()[:16]
    current = walk_id_for("soybean", images)
    assert current != unversioned  # cached reports from before the version was part of the key are not served
    monkeypatch.setattr(pipeline_module, "WALK_VERSION", "next")
    assert walk_id_for("soybean", images) != current


def test_find_dir_rejects_bad_ids(tmp_path):
    assert _pipeline(tmp_path, fake_runner).find_dir("../etc") is None


def test_from_settings_warns_when_label_table_is_empty(monkeypatch, caplog):
    monkeypatch.setattr("scout.app.pipeline.load_records", lambda: [])
    with caplog.at_level(logging.WARNING, logger="scout.app.pipeline"):
        Pipeline.from_settings(Settings(api_key="t"))
    assert "label table is empty" in caplog.text


def test_events_tell_the_whole_walk_with_pictures_and_time(tmp_path):
    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images, names=["a.jpg", "b.jpg"])
    events = walk.events
    assert all("t" in e for e in events)
    first = events[0]
    assert first["type"] == "walk_start" and first["crop"] == "soybean"
    assert first["stops"] == [{"stop": 0, "name": "a.jpg"}, {"stop": 1, "name": "b.jpg"}]
    settings = Settings(api_key="t")
    assert first["budget"] == {"calls": settings.max_tool_calls, "seconds": settings.stop_timeout_s}
    starts = [e for e in events if e["type"] == "stop_start"]
    assert len(starts) == 2 and all(e["picture"].startswith(f"/walks/{walk.id}/steps/s") for e in starts)
    thoughts = [e for e in events if e["type"] == "thought"]
    assert thoughts[0]["text"] == "Looking over the rows" and thoughts[0]["detail"] == ["Counting weeds"]
    marks = [e for e in events if e["type"] == "step"]
    assert len(marks) == 2 and all(e["picture"].endswith(".jpg") and e["tool"] == "mark" for e in marks)
    done = [e for e in events if e["type"] == "stop_done"]
    assert {e["status"] for e in done} == {"ok"} and all(e["cover"] == 1.0 and e["weeds"] == 1 for e in done)
    assert all(e["picture"].endswith(".jpg") for e in done)
    assert [e["type"] for e in events][-2:] == ["report_building", "report_ready"]
    folder = pipeline.find_dir(walk.id)
    assert sorted(p.name for p in (folder / "steps").iterdir()) == [
        "s0-00.jpg", "s0-01.jpg", "s0-02.jpg", "s1-00.jpg", "s1-01.jpg", "s1-02.jpg"]
    saved = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    assert saved == events and pipeline.saved_events(walk.id) == events


def test_step_pictures_are_found_by_valid_name_only(tmp_path):
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    assert pipeline.step_picture(walk.id, "s0-01.jpg").is_file()
    for bad in ("s0-99.jpg", "../report.json", "s0-01.png", "x.jpg"):
        assert pipeline.step_picture(walk.id, bad) is None
    assert pipeline.step_picture("not-a-walk", "s0-01.jpg") is None


def test_samples_carry_a_thumbnail_the_stop_count_and_the_decision(tmp_path):
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    demo = tmp_path / "demo"
    demo.mkdir()
    (tmp_path / "store" / walk.id).rename(demo / walk.id)
    (demo / "samples.json").write_text(json.dumps([{"walk_id": walk.id, "name": "Soybean, waterhemp"}]))
    [sample] = pipeline.samples()
    assert sample["name"] == "Soybean, waterhemp" and sample["stops"] == 1
    assert sample["thumbnail"] == f"/walks/{walk.id}/steps/s0-00.jpg"
    assert sample["decision"] == "No action"  # small_weedbook's waterhemp: 1% cover against the 5% tool default


def test_walk_still_finishes_when_its_replay_events_cannot_be_saved(tmp_path, monkeypatch, caplog):
    real_write = pipeline_module._write_atomic

    def write(path, text):
        if path.name == "events.json":
            raise OSError("disk full")
        real_write(path, text)

    monkeypatch.setattr(pipeline_module, "_write_atomic", write)
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    with caplog.at_level(logging.WARNING, logger="scout.app.pipeline"):
        pipeline.run(walk, images)
    assert walk.status == "done" and walk.events[-1]["type"] == "report_ready"
    folder = tmp_path / "store" / walk.id
    assert (folder / "report.json").exists() and not (folder / "events.json").exists()
    assert "replay events could not be saved" in caplog.text


def test_step_pictures_are_found_before_the_report_exists_and_in_the_demo_folder(tmp_path):
    pipeline = _pipeline(tmp_path, fake_runner)
    running, shipped = "a" * 16, "b" * 16
    for base, walk_id in ((tmp_path / "store", running), (tmp_path / "demo", shipped)):
        (base / walk_id / "steps").mkdir(parents=True)
        (base / walk_id / "steps" / "s0-00.jpg").write_bytes(b"jpeg")
    assert pipeline.find_dir(running) is None  # a running walk has no report yet
    assert pipeline.step_picture(running, "s0-00.jpg") == tmp_path / "store" / running / "steps" / "s0-00.jpg"
    assert pipeline.step_picture(shipped, "s0-00.jpg") == tmp_path / "demo" / shipped / "steps" / "s0-00.jpg"
    assert pipeline.step_picture(running, "s0-01.jpg") is None


def test_a_cached_run_leaves_the_saved_events_alone(tmp_path):
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    first = Walk(id=walk_id_for("corn", images), crop="corn")
    pipeline.run(first, images)
    saved = tmp_path / "store" / first.id / "events.json"
    before = saved.read_bytes()
    again = Walk(id=first.id, crop="corn")
    pipeline.run(again, images)
    assert saved.read_bytes() == before
    assert again.status == "done" and [e["type"] for e in again.events] == ["report_ready"]


def test_pictures_that_cannot_be_saved_leave_the_events_without_pictures(tmp_path, monkeypatch):
    real_save = Image.Image.save

    def save(self, fp, format=None, **params):
        if format == "JPEG":
            raise OSError("disk full")
        return real_save(self, fp, format, **params)

    monkeypatch.setattr(Image.Image, "save", save)
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    assert walk.status == "done"
    for kind in ("stop_start", "step", "stop_done"):
        [event] = [e for e in walk.events if e["type"] == kind]
        assert event["picture"] is None, kind
    assert list((tmp_path / "store" / walk.id / "steps").iterdir()) == []
    assert (tmp_path / "store" / walk.id / "stop_0.png").exists()  # the PNG overlay is not a step picture


def test_every_picture_in_the_events_can_be_fetched(tmp_path):
    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    urls = [e["picture"] for e in walk.events if e.get("picture")]
    assert len(urls) == 6
    for url in urls:
        prefix, name = url.rsplit("/", 1)
        assert prefix == f"/walks/{walk.id}/steps"
        assert pipeline.step_picture(walk.id, name).is_file(), url


def test_step_picture_names_are_ascii_digits_only(tmp_path):
    pipeline = _pipeline(tmp_path, fake_runner)
    walk_id = "c" * 16
    steps = tmp_path / "store" / walk_id / "steps"
    steps.mkdir(parents=True)
    arabic_zero = "\u0660"  # \d matches this; a step picture name never has it
    (steps / f"s{arabic_zero}-00.jpg").write_bytes(b"jpeg")
    assert pipeline.step_picture(walk_id, f"s{arabic_zero}-00.jpg") is None


def test_events_are_listed_in_the_order_of_their_times(monkeypatch):
    walk = Walk(id="w", crop="corn")
    calls, reading, resume = count(), threading.Event(), threading.Event()

    def clock():
        if next(calls) == 0:  # the first emit reads the clock early, then is held up before it can append
            reading.set()
            assert resume.wait(5)
            return walk.started + 1.0
        return walk.started + 9.0

    monkeypatch.setattr(pipeline_module, "time", SimpleNamespace(monotonic=clock))
    first = threading.Thread(target=walk.emit, args=({"type": "first"},))
    second = threading.Thread(target=walk.emit, args=({"type": "second"},))
    first.start()
    assert reading.wait(5)
    second.start()
    second.join(0.3)  # without a lock the second emit finishes here, ahead of the first
    resume.set()
    first.join(5)
    second.join(5)
    assert [(e["type"], e["t"]) for e in walk.events] == [("first", 1.0), ("second", 9.0)]


def _ship(pipeline, tmp_path, images, crop="soybean"):
    """Run a walk and move its folder from the store to the demo folder, as a shipped sample."""
    walk = Walk(id=walk_id_for(crop, images), crop=crop)
    pipeline.run(walk, images)
    (tmp_path / "demo").mkdir(exist_ok=True)
    (tmp_path / "store" / walk.id).rename(tmp_path / "demo" / walk.id)
    return walk


def test_the_sample_thumbnail_is_the_first_stops_photo(tmp_path):
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = _ship(pipeline, tmp_path, [make_image(index=0), make_image(index=1, color=(10, 10, 10))])
    saved = tmp_path / "demo" / walk.id / "events.json"
    events = json.loads(saved.read_text(encoding="utf-8"))
    saved.write_text(json.dumps(list(reversed(events))), encoding="utf-8")  # stop 1 now starts first in the list
    (tmp_path / "demo" / "samples.json").write_text(json.dumps([{"walk_id": walk.id, "name": "Two stops"}]))
    [sample] = pipeline.samples()
    assert sample["stops"] == 2 and sample["thumbnail"] == f"/walks/{walk.id}/steps/s0-00.jpg"


def test_a_sample_that_cannot_be_read_comes_back_bare_and_one_without_an_id_is_skipped(tmp_path, caplog):
    pipeline = _pipeline(tmp_path, fake_runner)
    good = _ship(pipeline, tmp_path, [make_image(index=0)])
    broken = _ship(pipeline, tmp_path, [make_image(index=0, color=(10, 10, 10))])
    (tmp_path / "demo" / broken.id / "report.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "demo" / "samples.json").write_text(json.dumps([
        {"walk_id": good.id, "name": "Good"},
        {"name": "No id"},
        {"walk_id": broken.id, "name": "Broken"}]))
    with caplog.at_level(logging.WARNING, logger="scout.app.pipeline"):
        samples = pipeline.samples()
    assert [s["name"] for s in samples] == ["Good", "Broken"]
    assert samples[0]["stops"] == 1 and samples[0]["decision"] == "No action" and samples[0]["thumbnail"]
    assert samples[1] == {"walk_id": broken.id, "name": "Broken", "thumbnail": None, "decision": None, "stops": None}
    assert broken.id in caplog.text
