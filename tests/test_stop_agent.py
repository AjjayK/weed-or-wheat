import itertools
import json
from dataclasses import replace

from scout.agent.llm import LlmError, LlmTurn, ToolCall
from scout.agent.prompts import stop_instructions
from scout.agent.state import MAX_MARK_BOXES
from scout.agent.stop_agent import MAX_CHECK_FLAGS, UNMARKED_FLAG, crop_flag, latest_overlay_only, run_stop
from scout.agent.tools import OVERLAY_CAPTION
from scout.config import Settings
from scout.retry import BadRequest
from tests.helpers import FakeLlm, FakeSam, make_image, rect, small_weedbook

SETTINGS = Settings(api_key="test")
MARK_CROP = ToolCall("c1", "mark", {"kind": "crop", "boxes": [{"box": [100, 0, 300, 1000], "label": "soybean row"},
                                                              {"box": [400, 300, 520, 420], "label": "soybean"}]})
MARK_WEEDS = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "waterhemp"},
                                                               {"box": [800, 100, 900, 200], "label": "marestail"},
                                                               {"box": [420, 320, 500, 400], "label": "pigweed"}]})


def _classify(*ids, species=None, confidence="medium"):
    return ToolCall("c-cls", "classify", {"items": [
        {"instance_id": i, "role": "weed", "species_id": species, "height_in": None, "leaf_count": None,
         "id_confidence": confidence} for i in ids]})


def _close(view_type="row", all_marked=True):
    return [ToolCall("c-m", "measure", {"view_type": view_type}),
            ToolCall("c-f", "finish", {"crop_stage": None, "notes": "done", "all_weeds_marked": all_marked})]


class TimeoutSam(FakeSam):
    """A SAM whose calls time out: the sky prompt, or every box request after the first."""

    def __init__(self, sky=False, later_boxes=False):
        super().__init__()
        self.sky, self.later_boxes = sky, later_boxes

    def segment_concept(self, image, phrase):
        if self.sky and phrase == "sky":
            self.calls.append(("concept", phrase))
            raise TimeoutError("sky timed out")
        return super().segment_concept(image, phrase)

    def segment_boxes(self, image, boxes):
        if self.later_boxes and any(kind == "boxes" for kind, _ in self.calls):
            raise TimeoutError("boxes timed out")
        return super().segment_boxes(image, boxes)


def _run(llm, crop="soybean", settings=SETTINGS, sam=None, **kwargs):
    return run_stop(make_image(100, 100), crop, llm, sam or FakeSam(), small_weedbook(), settings, **kwargs)


def test_full_stop():
    llm = FakeLlm([
        LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
        LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i4"})], ""),
        LlmTurn("r3", [ToolCall("c4", "classify", {"items": [
            {"instance_id": "s0-i3", "role": "weed", "species_id": "waterhemp", "height_in": 3, "leaf_count": 5,
             "id_confidence": "high"},
            {"instance_id": "s0-i4", "role": "weed", "species_id": "marestail", "height_in": None, "leaf_count": None,
             "id_confidence": "low"}]})], ""),
        LlmTurn("r4", [ToolCall("c5", "measure", {"view_type": "top_down"}),
                       ToolCall("c6", "finish", {"crop_stage": "V3", "notes": "Two weeds between rows.",
                                                 "all_weeds_marked": True})], ""),
    ])
    steps = []
    result, _ = _run(llm, on_step=steps.append)
    assert result.status == "ok" and result.weed_count == 2  # the pigweed box sat on a soybean plant
    assert result.species == {"waterhemp": 1, "horseweed": 1}
    assert result.crop_stage == "V3" and result.view_type == "top_down" and result.density_per_m2 is not None
    assert result.flags == ["Check s0-i4: low confidence"]
    assert len(steps) == len(result.steps) == 6
    first, third = llm.requests[0], llm.requests[2]
    assert [t["name"] for t in first["tools"]][:3] == ["survey", "segment", "mark"] and "0 to 1000" in first["instructions"]
    assert third["previous_response_id"] == "r2"
    assert [part["type"] for part in third["input"]] == ["function_call_output", "message"]  # the zoom image rides
    # in a user message after the tool output


def test_crop_marks_run_first_within_a_turn():
    llm = FakeLlm([LlmTurn("r1", [MARK_WEEDS, MARK_CROP], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, state = _run(llm)
    assert [s.summary for s in result.steps[:2]] == ["Mark crop: 2 of 2 boxes kept", "Mark weed: 2 of 3 boxes kept"]
    outputs = [p["call_id"] for p in llm.requests[1]["input"] if p.get("type") == "function_call_output"]
    assert outputs == ["c2", "c1"]  # outputs keep the model's order
    assert result.status == "ok" and result.weed_count == 2 and state.instances["s0-i1"].role == "crop"


def test_call_limit_closes_stop_and_flags_unclassified_weeds():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i3"})], "")])
    result, _ = _run(llm, settings=replace(SETTINGS, max_tool_calls=3))
    assert result.status == "ok" and result.weed_count == 2
    assert any(step.tool == "limit" for step in result.steps)
    assert result.flags == [UNMARKED_FLAG, "Check s0-i3: marked but never classified",
                            "Check s0-i4: marked but never classified"]  # stop-level flags come first
    assert result.count_approximate is True


PATCH = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [400, 750, 1000, 1000], "label": "foxtail"}]})


def test_patch_makes_the_count_approximate():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, PATCH], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("top_down")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is True
    assert ("Weed count is approximate: some weeds merged into patches or were counted in weed areas"
            in result.flags)


def test_a_big_plant_in_a_close_up_is_not_a_patch():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, PATCH], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("close_up")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is False
    assert result.flags == ["Check s0-i3: may be crop or several plants"]


def test_repeated_bad_request_fails_stop_and_keeps_steps():
    call = [ToolCall("c1", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "w"}]})]
    llm = FakeLlm([LlmTurn("r1", call, ""), LlmTurn("r2", call, "")])
    result, _ = _run(llm, sam=FakeSam(fail={"boxes"}))
    assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "mark", "error"]


def test_model_errors_fail_the_stop_and_keep_steps():
    for error in (LlmError("response r2 was incomplete"), BadRequest("image too large"),
                  TimeoutError("request timed out"), ConnectionError("connection dropped")):
        result, _ = _run(FakeLlm([LlmTurn("r1", [MARK_CROP], ""), error]))
        assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "error"]


LOST = BadRequest("Error code: 400 - {'error': {'message': 'referenced response not found or expired', "
                  "'type': 'invalid_request_error'}}")


def _turn(response_id, calls):
    """A turn whose output items replay its calls, as a real Llm.turn fills them."""
    items = [{"type": "function_call", "call_id": c.call_id, "name": c.name, "arguments": json.dumps(c.arguments)}
             for c in calls]
    return LlmTurn(response_id, calls, "", items)


def _kinds(items):
    return [item.get("type") or item["role"] for item in items]


def test_a_lost_conversation_is_resent_in_full_and_the_stop_goes_on_stateless():
    zoom = ToolCall("c3", "zoom", {"instance_id": "s0-i4"})
    llm = FakeLlm([_turn("r1", [MARK_CROP, MARK_WEEDS]), LOST, _turn("r2", [zoom]),
                   _turn("r3", [_classify("s0-i3", "s0-i4"), *_close()])])
    result, state = _run(llm)
    assert result.status == "ok" and state.finished and result.weed_count == 2
    first, lost, retry, last = llm.requests
    assert first["previous_response_id"] is None and lost["previous_response_id"] == "r1"
    assert retry["previous_response_id"] is None  # the same turn, resent without the chain
    assert retry["input"][0] == first["input"][0]  # starts with the photo message
    assert _kinds(retry["input"]) == ["message", "function_call", "function_call",
                                      "function_call_output", "function_call_output", "message"]
    assert [i["call_id"] for i in retry["input"][1:3]] == [i["call_id"] for i in retry["input"][3:5]] == ["c1", "c2"]
    assert retry["input"][1]["name"] == "mark" and json.loads(retry["input"][1]["arguments"])["kind"] == "crop"
    assert retry["input"][3:] == lost["input"]  # exactly the items the lost request carried: outputs and the picture
    assert last["previous_response_id"] is None  # the rest of the stop stays stateless
    assert last["input"][:6] == retry["input"]  # the history only grows
    assert _kinds(last["input"][6:]) == ["function_call", "function_call_output", "message"]
    assert [i["call_id"] for i in last["input"][6:8]] == ["c3", "c3"]  # the zoom call, its output, then the close-up
    retries = [s for s in result.steps if s.tool == "retry"]
    assert [(s.ok, s.summary) for s in retries] == [(False, "Muse Spark lost the conversation; resent it in full")]


def test_a_second_failure_after_the_resend_fails_the_stop_and_keeps_steps():
    llm = FakeLlm([_turn("r1", [MARK_CROP]), LOST, LOST])
    result, _ = _run(llm)
    assert result.status == "failed" and len(llm.requests) == 3  # one resend only
    assert [s.tool for s in result.steps] == ["mark", "retry", "error"]


def test_a_resent_history_keeps_assistant_text_and_the_nudge():
    prose = LlmTurn("r1", [], "I see weeds.", [{"role": "assistant", "content": "I see weeds."}])
    llm = FakeLlm([prose, LOST, _turn("r2", [MARK_CROP]), _turn("r3", _close("top_down"))])
    result, _ = _run(llm)
    assert result.status == "ok"
    retry = llm.requests[2]
    assert _kinds(retry["input"]) == ["message", "assistant", "message"]  # photo, its own text, the nudge
    assert retry["input"][1] == {"role": "assistant", "content": "I see weeds."}
    assert retry["input"][2]["content"][0]["text"].startswith("Call a tool")


def test_a_stateless_stop_does_not_resend_again():
    llm = FakeLlm([_turn("r1", [MARK_CROP]), LOST, _turn("r2", [MARK_WEEDS]), LOST])
    result, _ = _run(llm)
    assert result.status == "failed" and len(llm.requests) == 4
    assert [s.tool for s in result.steps].count("retry") == 1


def test_prose_reply_is_nudged_then_fails_the_stop():
    llm = FakeLlm([LlmTurn("r1", [], "I think there are weeds."), LlmTurn("r2", [], "Still prose.")])
    result, _ = _run(llm)
    assert result.status == "failed" and len(llm.requests) == 2  # the agent produced nothing: not "no crop found"
    assert result.flags == ["This stop could not be analyzed"]


def test_no_crop_found_needs_a_stop_the_agent_worked_on():
    llm = FakeLlm([LlmTurn("r1", _close("top_down"), "")])
    result, _ = _run(llm)
    assert result.status == "no_crop_found" and result.weed_count == 0


def test_sky_timeout_does_not_fail_the_stop():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close("row")], "")])
    sam = TimeoutSam(sky=True)
    result, state = _run(llm, sam=sam)
    assert result.status == "ok" and result.weed_count == 2 and state.sky is None
    assert [s.tool for s in result.steps].count("sky") == 1  # logged once
    assert sam.calls.count(("concept", "sky")) == 1  # asked once, by measure; finalize does not ask again


def test_a_sam_timeout_fails_only_this_stop_and_keeps_steps():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP], ""), LlmTurn("r2", [MARK_WEEDS], "")])
    result, _ = _run(llm, sam=TimeoutSam(later_boxes=True))
    assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "error"]
    assert result.steps[-1].summary.startswith("TimeoutError")


def test_measure_and_finish_run_last_whatever_order_the_model_lists_them():
    finish, measure = _close()[1], _close()[0]
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""), LlmTurn("r2", [_classify("s0-i3", "s0-i4")], ""),
                   LlmTurn("r3", [finish, measure], "")])
    result, state = _run(llm)
    assert len(llm.requests) == 3 and state.finished and result.notes == "done"
    assert [s.tool for s in result.steps][-2:] == ["measure", "finish"]
    # classify, measure and finish together in one turn, listed backwards
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [finish, measure, _classify("s0-i3", "s0-i4")], "")])
    result, state = _run(llm)
    assert len(llm.requests) == 2 and state.finished and result.weed_count == 2 and result.flags == []
    assert [s.tool for s in result.steps] == ["mark", "mark", "classify", "measure", "finish"]


def test_time_limit_closes_the_stop_after_the_turn_in_progress():
    ticks = itertools.count(0, 200)  # every clock reading is 200 s later: the 360 s budget is gone after one turn
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""), LlmTurn("r2", [_classify("s0-i3", "s0-i4")], "")])
    result, _ = _run(llm, clock=lambda: float(next(ticks)))
    assert len(llm.requests) == 1 and result.status == "ok"
    assert result.steps[-1].tool == "limit" and result.steps[-1].ok is False


def test_weed_free_top_down_stop_has_zero_density():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP], ""), LlmTurn("r2", _close("top_down"), "")])
    result, _ = _run(llm)
    assert result.status == "ok" and result.weed_count == 0 and result.flags == []
    assert result.density_per_m2 == 0.0 and result.species_density == {}


def test_alfalfa_needs_no_crop_boxes():
    weeds = ToolCall("c1", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "dandelion"},
                                                              {"box": [800, 100, 900, 200], "label": "dandelion"}]})
    llm = FakeLlm([LlmTurn("r1", [weeds], ""), LlmTurn("r2", [_classify("s0-i1", "s0-i2"), *_close("close_up")], "")])
    result, _ = _run(llm, crop="alfalfa")
    assert result.status == "ok" and result.weed_count == 2 and result.in_row_weed_share is None


def test_instructions_use_thousandths_and_crop_hints():
    soybean = stop_instructions("soybean", ["waterhemp", "horseweed"], 12)
    alfalfa = stop_instructions("alfalfa", ["dandelion"], 12)
    assert "0 to 1000" in soybean and "waterhemp, horseweed" in soybean and "12 tool calls" in soybean
    assert '"crop row"' in soybean and "crop boxes are optional" in alfalfa
    assert "crop boxes are optional" not in soybean
    assert "no weeds" in soybean and '"yellow flower"' in soybean and '"flower"' not in soybean  # alfalfa blooms
    assert "estimate" in soybean and "null" in soybean
    assert "close to the camera" in soybean and "close to the camera" in alfalfa
    assert "hug each row's plants" in soybean


SURVEY = ToolCall("c0", "survey", {"crop_seen": "corn", "weeds_visible": 2})


def test_survey_runs_first_and_a_mismatched_crop_is_flagged():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS, SURVEY], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm)
    assert result.steps[0].tool == "survey" and result.crop_seen == "corn"
    assert result.flags == ["This photo looks like corn, not soybean"]
    weeds = next(p for p in llm.requests[1]["input"] if p.get("call_id") == "c2")
    assert json.loads(weeds["output"])["weeds_estimate"] == 2  # the survey ran before the weed mark


def test_only_the_latest_marking_picture_rides_along():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    _run(llm)
    messages = [p for p in llm.requests[1]["input"] if p.get("type") == "message"]
    assert len(messages) == 1
    images = [c for c in messages[0]["content"] if c["type"] == "input_image"]
    assert len(images) == 1 and OVERLAY_CAPTION in messages[0]["content"][0]["text"]


def test_marking_well_under_the_estimate_is_flagged():
    survey = ToolCall("c0", "survey", {"crop_seen": "soybean", "weeds_visible": 10})
    llm = FakeLlm([LlmTurn("r1", [survey, MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm)
    assert result.flags == [UNMARKED_FLAG] and result.count_approximate is True


def test_the_agent_saying_weeds_remain_is_flagged():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close(all_marked=False)], "")])
    result, _ = _run(llm)
    assert result.flags == [UNMARKED_FLAG]


def test_crop_flag_wording():
    assert crop_flag("corn", "soybean") == "This photo looks like corn, not soybean"
    assert crop_flag("other", "corn") == "This photo doesn't look like corn"
    assert crop_flag("unsure", "corn") is None and crop_flag(None, "corn") is None
    assert crop_flag("corn", "corn") is None


def test_the_floor_flag_names_the_mark_limit():
    many = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "w"}] +
                                   [{"box": [400, 500, 500, 600], "label": "w"}] * (MAX_MARK_BOXES - 1)})
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, many], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("close_up")], "")])
    result, _ = _run(llm)
    assert f"Weed count is a floor: a mark call used all {MAX_MARK_BOXES} boxes" in result.flags


def test_a_failed_stop_keeps_the_crop_it_saw():
    result, _ = _run(FakeLlm([LlmTurn("r1", [SURVEY, MARK_CROP], ""), LlmError("response r2 was incomplete")]))
    assert result.status == "failed" and result.crop_seen == "corn"
    result, _ = _run(FakeLlm([LlmTurn("r1", [SURVEY], ""), LlmTurn("r2", [], "Prose."), LlmTurn("r3", [], "Prose.")]))
    assert result.status == "failed" and result.crop_seen == "corn"  # surveyed, then marked nothing


def test_latest_overlay_only_keeps_the_last_overlay_and_every_other_picture():
    pictures = [(OVERLAY_CAPTION, "a"), ("Close-up of x", "b"), (OVERLAY_CAPTION, "c")]
    assert latest_overlay_only(pictures) == [("Close-up of x", "b"), (OVERLAY_CAPTION, "c")]
    closeups = [("Close-up of x", "a"), ("Close-up of y", "b")]
    assert latest_overlay_only(closeups) == closeups and latest_overlay_only([]) == []


def test_stop_level_flags_come_first_and_per_plant_checks_are_capped():
    spots = [(x, y) for x in (40, 55, 70, 85) for y in (5, 35, 65)]  # 12 separate 6 px weeds, clear of the row
    scene = {"row": rect(10, 0, 30, 100), **{f"w{n}": rect(x, y, x + 6, y + 6) for n, (x, y) in enumerate(spots)}}
    crop = ToolCall("c1", "mark", {"kind": "crop", "boxes": [{"box": [100, 0, 300, 1000], "label": "soybean row"}]})
    weeds = ToolCall("c2", "mark", {"kind": "weed", "boxes": [
        {"box": [x * 10, y * 10, (x + 6) * 10, (y + 6) * 10], "label": "weed"} for x, y in spots]})
    ids = [f"s0-i{n}" for n in range(2, 14)]
    llm = FakeLlm([LlmTurn("r1", [SURVEY, crop, weeds], ""),
                   LlmTurn("r2", [_classify(*ids, confidence="low"), *_close()], "")])
    result, _ = _run(llm, sam=FakeSam(scene))
    assert result.weed_count == 12 and MAX_CHECK_FLAGS == 10
    assert result.flags == ["This photo looks like corn, not soybean",
                            *[f"Check s0-i{n}: low confidence" for n in range(2, 12)],
                            "…and 2 more plants to check"]


def test_instructions_ask_for_every_weed_not_a_sample():
    text = stop_instructions("corn", ["field_horsetail"], 30)
    assert "survey" in text and 'kind "patch"' in text and "all_weeds_marked" in text
    assert "not a sample" in text and "up to 6" in text and "30 tool calls" in text and "picture" in text
    assert "at most 16" not in text and "6 turns" not in text and "largest first" not in text
    assert "at most 2" not in text
    assert "unsure" in text  # no crop in view (a close-up of a weed): don't call it another crop
