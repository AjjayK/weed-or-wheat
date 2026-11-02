import base64
import io
import json
import logging

import pytest
from PIL import Image

from scout.agent.state import StopFailed, StopState
from scout.agent.tools import OVERLAY_CAPTION, TOOL_SCHEMAS, ToolRunner
from scout.retry import BadRequest
from scout.vision.sam_parse import SamRecord
from tests.helpers import FakeSam, field_photo, field_scene, make_image, mask_record, rect, small_weedbook

# Boxes as the agent gives them, 0-1000; the test photo is 100x100, so pixels are a tenth of these.
ROW, PLANT = [100, 0, 300, 1000], [400, 300, 520, 420]
WEED_A, WEED_B, EDGE = [600, 600, 700, 700], [800, 100, 900, 200], [260, 400, 360, 500]
ON_PLANT, IN_ROW = [420, 320, 500, 400], [120, 500, 220, 600]
PATCH, BARE = [400, 750, 1000, 1000], [400, 500, 500, 600]


def _runner(sam=None, max_calls=12, image=None):
    state = StopState(image or make_image(100, 100), "soybean")
    return ToolRunner(state, sam or FakeSam(), small_weedbook(), max_calls=max_calls), state


def _mark(runner, kind, *boxes):
    items = [{"box": box, "label": f"{kind} {n}"} for n, box in enumerate(boxes, start=1)]
    return json.loads(runner.run("mark", {"kind": kind, "boxes": items}).output)


def test_schemas_name_seven_tools():
    assert [t["name"] for t in TOOL_SCHEMAS] == ["survey", "segment", "mark", "zoom", "classify", "measure", "finish"]
    schemas = {t["name"]: t["parameters"] for t in TOOL_SCHEMAS}
    assert schemas["mark"]["properties"]["boxes"]["maxItems"] == 120
    assert schemas["mark"]["properties"]["kind"]["enum"] == ["crop", "weed", "patch"]
    assert "all_weeds_marked" in schemas["finish"]["required"]
    assert schemas["survey"]["required"] == ["crop_seen", "weeds_visible"]
    assert schemas["survey"]["properties"]["weeds_visible"]["maximum"] == 1000


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
                            "yellow flower": [mask_record("0", scene["weed_b"]),
                                       mask_record("1", scene["plant"] & rect(40, 30, 50, 40))]})
    runner, state = _runner(sam)
    _mark(runner, "crop", PLANT)
    crop = json.loads(runner.run("segment", {"phrase": " crop row ", "kind": "crop"}).output)
    assert crop["total_found"] == 1 and state.instances["s0-i2"].role == "crop"
    weeds = json.loads(runner.run("segment", {"phrase": "yellow flower", "kind": "weed"}).output)
    assert weeds["total_found"] == 1 and weeds["notes"] == ["match 2: mostly crop"]
    assert state.instances["s0-i3"].prompt == "yellow flower"
    assert state.steps[-1].summary == 'Segment "yellow flower": 1 found, 1 dropped as crop'
    assert runner.run("segment", {"phrase": "plant", "kind": "all_plants"}).output.startswith("error:")


def test_mark_rejects_bad_boxes_before_calling_sam():
    sam = FakeSam()
    runner, _ = _runner(sam)
    one = [{"box": WEED_A, "label": "w"}]
    assert runner.run("mark", {"kind": "weed", "boxes": []}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": one * 121}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [{"box": [1, 2, 3], "label": "w"}]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [[600, 600, 700, 700]]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "plant", "boxes": one}).output.startswith("error:")
    assert sam.calls == []


def test_a_full_mark_call_makes_the_count_a_floor():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, *[BARE] * 119)
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


def _verdict(instance_id, role="weed", species=None, confidence="medium", **extra):
    return {"instance_id": instance_id, "role": role, "species_id": species, "height_in": None, "leaf_count": None,
            "id_confidence": confidence, **extra}


def test_classify_coerces_numbers_and_ignores_crop_names():
    runner, state = _runner()
    _mark(runner, "crop", PLANT)
    _mark(runner, "weed", WEED_A, WEED_B)  # s0-i1 is the crop plant; s0-i2 and s0-i3 are weeds
    ok = runner.run("classify", {"items": [
        _verdict("s0-i2", species="", confidence="high", height_in="3", leaf_count="4.0"),
        _verdict("s0-i1", role="crop", species="soybean", confidence="high"),  # a crop's name is not a weed species
        _verdict("s0-i3", species="Marestail", height_in=2.5, leaf_count=6)]})
    assert json.loads(ok.output) == {"ok": ["s0-i2", "s0-i1", "s0-i3"]}
    weed, crop, other = (state.instances[i] for i in ("s0-i2", "s0-i1", "s0-i3"))
    assert (weed.height_in, weed.leaf_count, weed.species_id) == (3.0, 4, None)
    assert (crop.role, crop.species_id) == ("crop", None)
    assert (other.species_id, other.height_in, other.leaf_count) == ("horseweed", 2.5, 6)


def test_classify_rejects_bad_values_before_changing_anything():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, WEED_B)
    good = _verdict("s0-i1", species="waterhemp", confidence="high", height_in=3)
    for bad in ({"height_in": "tall"}, {"height_in": -2}, {"height_in": "nan"}, {"height_in": True},
                {"leaf_count": "many"}, {"leaf_count": [3]}, {"id_confidence": "sure"}, {"role": "tree"}):
        out = runner.run("classify", {"items": [good, _verdict("s0-i2", **bad)]})
        assert out.output.startswith("error:"), bad
        assert all(i.id_confidence is None and i.height_in is None for i in state.instances.values()), bad


def test_a_confident_verdict_clears_the_check_flag_but_patches_stay_flagged():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    _mark(runner, "weed", EDGE, WEED_A, PATCH)  # s0-i2 sits partly in the row canopy, s0-i4 is a patch
    assert [state.instances[i].check_this for i in ("s0-i2", "s0-i3", "s0-i4")] == [True, False, True]
    runner.run("classify", {"items": [_verdict("s0-i2", confidence="high"), _verdict("s0-i3", confidence="low"),
                                      _verdict("s0-i4", confidence="high")]})
    assert [state.instances[i].check_this for i in ("s0-i2", "s0-i3", "s0-i4")] == [False, True, True]


def test_finish_tidies_stage_and_notes():
    runner, state = _runner()
    runner.run("measure", {"view_type": "close_up"})
    assert runner.run("finish", {"crop_stage": " V3 ", "notes": None}).finished
    assert (state.crop_stage, state.notes) == ("V3", "")
    assert runner.run("finish", {"crop_stage": "  ", "notes": 7}).finished
    assert (state.crop_stage, state.notes) == (None, "7")


class _ZeroBasedSam(FakeSam):
    """Numbers its objects from 0, though the first box's object should be 1."""

    def segment_boxes(self, image, boxes):
        return [SamRecord(str(int(r.object_id) - 1), r.box, r.confidence, r.mask)
                for r in super().segment_boxes(image, boxes)]


def test_mark_logs_object_ids_sam_was_not_asked_for():
    runner, state = _runner(_ZeroBasedSam())
    _mark(runner, "weed", WEED_A, WEED_B)
    odd = [s for s in state.steps if s.tool == "mark" and not s.ok]
    assert [s.summary for s in odd] == ["SAM returned unexpected object ids ['0']"]
    runner, state = _runner()  # ids 1..N, as asked: nothing to log
    _mark(runner, "weed", WEED_A, WEED_B)
    assert all(s.ok for s in state.steps)


def test_segment_lists_up_to_a_hundred_matches():
    cells = [(n % 10 * 10, n // 10 * 10) for n in range(100)]
    matches = [mask_record(str(n), rect(x, y, x + 4, y + 4)) for n, (x, y) in enumerate(cells)]
    runner, _ = _runner(FakeSam(concepts={"dandelion": matches}))
    out = json.loads(runner.run("segment", {"phrase": "dandelion", "kind": "weed"}).output)
    assert out["total_found"] == 100 and len(out["found"]) == 100


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


def test_survey_stores_the_crop_seen_and_the_estimate():
    runner, state = _runner()
    assert json.loads(runner.run("survey", {"crop_seen": "corn", "weeds_visible": 52}).output) == {"ok": True}
    assert (state.crop_seen, state.weeds_estimate) == ("corn", 52)
    assert state.steps[-1].summary == "Survey: looks like corn, about 52 weeds"
    runner.run("survey", {"crop_seen": "unsure", "weeds_visible": "3"})
    assert (state.crop_seen, state.weeds_estimate) == ("unsure", 3)
    assert state.steps[-1].summary == "Survey: crop unclear, about 3 weeds"
    for bad in ({"crop_seen": "wheat", "weeds_visible": 1}, {"crop_seen": "corn", "weeds_visible": -1},
                {"crop_seen": "corn", "weeds_visible": None}):
        assert runner.run("survey", bad).output.startswith("error:")
    assert (state.crop_seen, state.weeds_estimate) == ("unsure", 3)


def test_mark_splits_many_boxes_into_requests_of_sixteen():
    sam = FakeSam()
    runner, state = _runner(sam)
    out = _mark(runner, "weed", *[BARE] * 20, WEED_B, *[BARE] * 15)
    assert sorted(len(boxes) for kind, boxes in sam.calls if kind == "boxes") == [4, 16, 16]
    assert [(f["box_no"], f["box"]) for f in out["found"]] == [(21, WEED_B)]
    assert out["notes"][0] == "box 1: nothing found" and out["notes"][-1] == "box 36: nothing found"
    assert not state.box_cap_hit and state.steps[-1].summary == "Mark weed: 1 of 36 boxes kept"


def test_mark_answers_with_counts_a_hint_and_a_picture():
    runner, state = _runner()
    runner.run("survey", {"crop_seen": "soybean", "weeds_visible": 2})
    result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "waterhemp"}]})
    out = json.loads(result.output)
    assert out["weeds_marked"] == {"plants": 1, "patches": 0} and out["weeds_estimate"] == 2
    assert out["hint"] == ("You estimated 2 weeds; your marks count as about 1: 1 boxed plant. Mark the rest: more "
                           "boxes, or kind patch where they are too dense.")
    assert [caption for caption, _ in result.images] == [OVERLAY_CAPTION]
    assert result.images[0][1].startswith("data:image/jpeg;base64,")
    out = _mark(runner, "weed", WEED_B)
    assert out["weeds_marked"] == {"plants": 2, "patches": 0} and "hint" not in out


def test_patch_kind_measures_green_pixels_and_follows_the_crop():
    sam = FakeSam(scene={"row": rect(10, 0, 30, 100), "leaf": rect(70, 60, 75, 80)})
    runner, state = _runner(sam, image=field_photo())
    _mark(runner, "crop", ROW)
    out = _mark(runner, "patch", [500, 500, 900, 900], [50, 100, 350, 900], [350, 100, 550, 400])
    assert [(f["box_no"], f["plants"], f["area_pct"]) for f in out["found"]] == [(1, 2, 2.0)]
    assert out["notes"] == ["box 2: mostly crop: box the weeds between the rows, not the row",
                            "box 3: no green weed pixels in this area"]
    assert out["weeds_marked"] == {"plants": 0, "patches": 1}
    assert state.steps[-1].summary == "Mark patch: 1 of 3 areas kept"
    assert ("boxes", ((50, 50, 90, 90),)) not in sam.calls  # weed areas never go to SAM
    _mark(runner, "crop", [700, 600, 750, 800])  # a crop leaf over the second shoot
    area = state.instances[out["found"][0]["id"]]
    assert area.area_px == 100 and area.plant_count == 1


def test_finish_records_whether_every_weed_is_marked():
    runner, state = _runner()
    runner.run("measure", {"view_type": "close_up"})
    assert runner.run("finish", {"crop_stage": None, "notes": "", "all_weeds_marked": False}).finished
    assert state.all_weeds_marked is False
    runner.run("finish", {"crop_stage": None, "notes": "", "all_weeds_marked": "yes"})
    assert state.all_weeds_marked is None  # only a real true or false counts


def test_the_hint_counts_weed_area_pieces_and_says_so():
    runner, state = _runner(FakeSam(scene={"row": rect(10, 0, 30, 100)}), image=field_photo())
    _mark(runner, "crop", ROW)
    runner.run("survey", {"crop_seen": "soybean", "weeds_visible": 10})
    out = _mark(runner, "patch", [500, 500, 900, 900])
    assert out["weeds_marked"] == {"plants": 0, "patches": 1}
    assert out["hint"] == ("You estimated 10 weeds; your marks count as about 2: 0 boxed plants and 1 weed area "
                           "holding about 2. Mark the rest: more boxes, or kind patch where they are too dense.")


def test_a_crop_mark_never_gets_the_weed_hint():
    runner, state = _runner(FakeSam(scene={"row": rect(10, 0, 30, 100)}), image=field_photo())
    runner.run("survey", {"crop_seen": "soybean", "weeds_visible": 10})
    out = _mark(runner, "crop", ROW)
    assert "hint" not in out and out["weeds_estimate"] == 10
    assert out["weeds_marked"] == {"plants": 0, "patches": 0}


def test_a_failed_overlay_leaves_a_warning_and_a_note_but_the_counts_stand(monkeypatch, caplog):
    runner, state = _runner()

    def broken():
        raise RuntimeError("no picture today")

    monkeypatch.setattr(state, "overlay", broken)
    with caplog.at_level(logging.WARNING, logger="scout.agent.tools"):
        result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "waterhemp"}]})
    out = json.loads(result.output)
    assert out["picture"] == "unavailable" and out["weeds_marked"] == {"plants": 1, "patches": 0}
    assert result.images == []
    assert [r.getMessage() for r in caplog.records] == ["overlay picture failed"]
    assert caplog.records[0].exc_info is not None


def test_a_picture_that_works_has_no_unavailable_note():
    runner, _ = _runner()
    result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "waterhemp"}]})
    assert "picture" not in json.loads(result.output) and len(result.images) == 1


class _SecondChunkFails(FakeSam):
    """Rejects the 4-box request, which is the second chunk of a 20-box call."""

    def segment_boxes(self, image, boxes):
        if len(boxes) == 4:
            self.calls.append(("boxes", tuple(boxes)))
            raise BadRequest("too many masks")
        return super().segment_boxes(image, boxes)


def test_a_rejected_chunk_names_its_boxes_and_registers_nothing():
    runner, state = _runner(_SecondChunkFails())
    items = [{"box": box, "label": "w"} for box in [WEED_A, WEED_B, *[BARE] * 18]]
    result = runner.run("mark", {"kind": "weed", "boxes": items})
    assert result.output == ("error: SAM rejected the request: boxes 17-20: too many masks. Nothing from this call "
                             "was marked: send all the boxes again, changing those.")
    assert state.instances == {} and state.bad_requests["mark"] == 1
    assert result.images == []


def test_mark_names_the_box_that_is_malformed():
    sam = FakeSam()
    runner, _ = _runner(sam)
    good = {"box": WEED_A, "label": "w"}
    for n, bad in ((2, {"label": "w"}), (2, [600, 600, 700, 700]), (2, {"box": [1, 2, 3], "label": "w"}),
                   (2, {"box": [1, 2, 3, "x"], "label": "w"}), (2, {"box": [1, 2, 3, None], "label": "w"}),
                   (2, {"box": [1, 2, 3, True], "label": "w"}), (2, {"box": "600,600,700,700", "label": "w"}),
                   (2, {"box": [1, 2, 3, float("nan")], "label": "w"})):
        out = runner.run("mark", {"kind": "weed", "boxes": [good, bad]}).output
        assert out == f"error: box {n}: needs a box of four numbers", bad
    assert sam.calls == []


def test_survey_rounds_the_estimate_and_caps_it():
    runner, state = _runner()
    runner.run("survey", {"crop_seen": "corn", "weeds_visible": 3.6})
    assert state.weeds_estimate == 4
    runner.run("survey", {"crop_seen": "corn", "weeds_visible": "5000"})
    assert state.weeds_estimate == 1000 and state.steps[-1].summary == "Survey: looks like corn, about 1000 weeds"
    assert runner.run("survey", {"crop_seen": "corn", "weeds_visible": None}).output == (
        "error: weeds_visible must be a number")


def test_the_overlay_picture_is_a_smaller_jpeg_and_explains_the_gray_zone(monkeypatch):
    runner, _ = _runner()
    qualities = []
    monkeypatch.setattr("scout.agent.tools.to_data_url",
                        lambda image, quality=90: qualities.append(quality) or "data:image/jpeg;base64,")
    result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "waterhemp"}]})
    assert qualities == [80] and result.images == [(OVERLAY_CAPTION, "data:image/jpeg;base64,")]
    assert "magenta is weed, gray is the in-row zone, blue boxes" in OVERLAY_CAPTION
