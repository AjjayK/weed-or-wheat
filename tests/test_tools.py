import base64
import io
import json

import pytest
from PIL import Image

from scout.agent.state import StopFailed, StopState
from scout.agent.tools import TOOL_SCHEMAS, ToolRunner
from tests.helpers import FakeSam, field_scene, make_image, mask_record, rect, small_weedbook

# Boxes as the agent gives them, 0-1000; the test photo is 100x100, so pixels are a tenth of these.
ROW, PLANT = [100, 0, 300, 1000], [400, 300, 520, 420]
WEED_A, WEED_B, EDGE = [600, 600, 700, 700], [800, 100, 900, 200], [260, 400, 360, 500]
ON_PLANT, IN_ROW = [420, 320, 500, 400], [120, 500, 220, 600]
PATCH, BARE = [400, 750, 1000, 1000], [400, 500, 500, 600]


def _runner(sam=None, max_calls=12):
    state = StopState(make_image(100, 100), "soybean")
    return ToolRunner(state, sam or FakeSam(), small_weedbook(), max_calls=max_calls), state


def _mark(runner, kind, *boxes):
    items = [{"box": box, "label": f"{kind} {n}"} for n, box in enumerate(boxes, start=1)]
    return json.loads(runner.run("mark", {"kind": kind, "boxes": items}).output)


def test_schemas_name_six_tools():
    assert [t["name"] for t in TOOL_SCHEMAS] == ["segment", "mark", "zoom", "classify", "measure", "finish"]


def test_mark_sends_pixels_in_one_request_and_reports_thousandths():
    sam = FakeSam()
    runner, state = _runner(sam)
    crop = _mark(runner, "crop", ROW, PLANT)
    assert [f["box"] for f in crop["found"]] == [ROW, PLANT] and crop["notes"] == []
    out = _mark(runner, "weed", WEED_A, WEED_B, EDGE, ON_PLANT, IN_ROW, PATCH, BARE)
    assert sam.calls[-1] == ("boxes", ((60, 60, 70, 70), (80, 10, 90, 20), (26, 40, 36, 50), (42, 32, 50, 40),
                                       (12, 50, 22, 60), (40, 75, 100, 100), (40, 50, 50, 60)))
    assert [(f["box_no"], f["id"]) for f in out["found"]] == [(1, "s0-i3"), (2, "s0-i4"), (3, "s0-i5"), (5, "s0-i6"),
                                                              (6, "s0-i7")]
    assert out["found"][2]["box"] == EDGE and out["found"][2]["check_this"] is True
    assert "check_this" not in out["found"][0] and out["found"][3]["check_this"] is True  # in the row: zoom on it
    assert out["notes"] == ["box 4: mostly crop", "box 6: large patch; counted as one, approximate",
                            "box 7: nothing found"]
    assert state.instances["s0-i3"].prompt == "weed 1" and state.instances["s0-i3"].role == "weed"
    assert state.steps[-1].summary == "Mark weed: 5 of 7 boxes kept"


def test_segment_phrase_adds_crop_or_checked_weeds():
    scene = field_scene()
    sam = FakeSam(concepts={"crop row": [mask_record("0", scene["row"], 0.7)],
                            "flower": [mask_record("0", scene["weed_b"]),
                                       mask_record("1", scene["plant"] & rect(40, 30, 50, 40))]})
    runner, state = _runner(sam)
    _mark(runner, "crop", PLANT)
    crop = json.loads(runner.run("segment", {"phrase": " crop row ", "kind": "crop"}).output)
    assert crop["total_found"] == 1 and state.instances["s0-i2"].role == "crop"
    weeds = json.loads(runner.run("segment", {"phrase": "flower", "kind": "weed"}).output)
    assert weeds["total_found"] == 1 and weeds["notes"] == ["match 2: mostly crop"]
    assert state.instances["s0-i3"].prompt == "flower"
    assert state.steps[-1].summary == 'Segment "flower": 1 found, 1 dropped as crop'
    assert runner.run("segment", {"phrase": "plant", "kind": "all_plants"}).output.startswith("error:")


def test_mark_rejects_bad_boxes_before_calling_sam():
    sam = FakeSam()
    runner, _ = _runner(sam)
    one = [{"box": WEED_A, "label": "w"}]
    assert runner.run("mark", {"kind": "weed", "boxes": []}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": one * 17}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [{"box": [1, 2, 3], "label": "w"}]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [[600, 600, 700, 700]]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "plant", "boxes": one}).output.startswith("error:")
    assert sam.calls == []


def test_sixteen_weed_boxes_make_the_count_approximate():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, *[BARE] * 15)
    assert state.box_cap_hit and len(state.unclassified()) == 1
    metrics = json.loads(runner.run("measure", {"view_type": "close_up"}).output)
    assert metrics["weed_count"] == 1 and metrics["count_approximate"] is True


def test_classify_resolves_synonyms_and_rejects_unknown():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, WEED_B)
    ok = runner.run("classify", {"items": [{"instance_id": "s0-i2", "role": "weed", "species_id": "Marestail",
                                            "height_in": 2.5, "leaf_count": 6, "id_confidence": "low"}]})
    assert json.loads(ok.output) == {"ok": ["s0-i2"]}
    inst = state.instances["s0-i2"]
    assert inst.species_id == "horseweed" and inst.check_this is True
    bad = runner.run("classify", {"items": [{"instance_id": "s0-i1", "role": "weed", "species_id": "dragonweed",
                                             "id_confidence": "high"}]})
    assert bad.output.startswith("error:") and state.instances["s0-i1"].id_confidence is None


def test_zoom_on_an_instance_or_a_thousandths_box():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    result = runner.run("zoom", {"instance_id": "s0-i1"})
    assert result.images and result.images[0][1].startswith("data:image/jpeg;base64,")
    boxed = runner.run("zoom", {"box": [500, 500, 600, 600]})
    close_up = Image.open(io.BytesIO(base64.b64decode(boxed.images[0][1].split(",", 1)[1])))
    assert close_up.size == (12, 12)  # pixels (50, 50, 60, 60), padded by 15%
    assert state.steps[-1].summary == "Zoom on box [500, 500, 600, 600]"


def test_unknown_tool_and_bad_arguments():
    runner, _ = _runner()
    assert runner.run("explode", {}).output.startswith("error:")
    assert runner.run("segment_at", {"kind": "weed", "box": WEED_A}).output.startswith("error:")  # not a tool
    assert runner.run("zoom", {"__invalid__": "{"}).output.startswith("error:")
    assert runner.run("measure", {"view": "row"}).output.startswith("error:")


def test_finish_needs_classified_weeds_and_measure():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    _mark(runner, "weed", WEED_A, WEED_B, EDGE)
    refused = runner.run("finish", {"crop_stage": None, "notes": ""})
    assert refused.output.startswith("error:") and "s0-i2" in refused.output and not refused.finished
    runner.run("classify", {"items": [{"instance_id": i, "role": "weed", "species_id": None, "height_in": None,
                                       "leaf_count": None, "id_confidence": "medium"}
                                      for i in ("s0-i2", "s0-i3", "s0-i4")]})
    assert runner.run("finish", {"crop_stage": None, "notes": ""}).output == "error: call measure first"
    metrics = json.loads(runner.run("measure", {"view_type": "row"}).output)
    assert metrics["weed_count"] == 3 and metrics["in_row_weed_share"] > 0
    assert runner.run("finish", {"crop_stage": "V3", "notes": "ok"}).finished
    assert state.crop_stage == "V3" and state.finished


def test_two_bad_requests_fail_the_stop():
    runner, _ = _runner(FakeSam(fail={"boxes"}))
    args = {"kind": "weed", "boxes": [{"box": WEED_A, "label": "w"}]}
    assert runner.run("mark", args).output.startswith("error:")
    with pytest.raises(StopFailed):
        runner.run("mark", args)
