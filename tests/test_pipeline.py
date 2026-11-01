from scout.agent.state import StopState
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings
from scout.models import StopResult
from tests.helpers import FakeLlm, make_image, small_weedbook


def fake_runner(image, crop, llm, sam, weeds, settings, on_step=None):
    state = StopState(image, crop, on_step=on_step)
    state.log("segment", "fake step")
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
    assert pipeline.cached(walk.id).walk_id == walk.id


def test_failed_stop_does_not_sink_walk(tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, boom)
    walk = Walk(id=walk_id_for("corn", images), crop="corn")
    pipeline.run(walk, images)
    assert walk.status == "done" and walk.report.stops[0].status == "failed"


def test_walk_id_is_stable_and_hex():
    images = [make_image()]
    assert walk_id_for("corn", images) == walk_id_for("corn", images)
    assert walk_id_for("corn", images) != walk_id_for("soybean", images)
    assert len(walk_id_for("corn", images)) == 16


def test_find_dir_rejects_bad_ids(tmp_path):
    assert _pipeline(tmp_path, fake_runner).find_dir("../etc") is None
