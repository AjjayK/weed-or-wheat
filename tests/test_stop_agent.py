import itertools
from dataclasses import replace

from scout.agent.llm import LlmError, LlmTurn, ToolCall
from scout.agent.prompts import stop_instructions
from scout.agent.stop_agent import run_stop
from scout.config import Settings
from scout.retry import BadRequest
from tests.helpers import FakeLlm, FakeSam, make_image, small_weedbook

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


def _close(view_type="row"):
    return [ToolCall("c-m", "measure", {"view_type": view_type}),
            ToolCall("c-f", "finish", {"crop_stage": None, "notes": "done"})]


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
                       ToolCall("c6", "finish", {"crop_stage": "V3", "notes": "Two weeds between rows."})], ""),
    ])
    steps = []
    result, _ = _run(llm, on_step=steps.append)
    assert result.status == "ok" and result.weed_count == 2  # the pigweed box sat on a soybean plant
    assert result.species == {"waterhemp": 1, "horseweed": 1}
    assert result.crop_stage == "V3" and result.view_type == "top_down" and result.density_per_m2 is not None
    assert result.flags == ["Check s0-i4: low confidence"]
    assert len(steps) == len(result.steps) == 6
    first, third = llm.requests[0], llm.requests[2]
    assert [t["name"] for t in first["tools"]][:2] == ["segment", "mark"] and "0 to 1000" in first["instructions"]
    assert third["previous_response_id"] == "r2"
    assert [part["type"] for part in third["input"]] == ["function_call_output", "message"]  # the zoom image rides
    # in a user message after the tool output


def test_crop_marks_run_first_within_a_turn():
    llm = FakeLlm([LlmTurn("r1", [MARK_WEEDS, MARK_CROP], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, state = _run(llm)
    assert [s.summary for s in result.steps[:2]] == ["Mark crop: 2 of 2 boxes kept", "Mark weed: 2 of 3 boxes kept"]
    assert [p["call_id"] for p in llm.requests[1]["input"]] == ["c2", "c1"]  # outputs keep the model's order
    assert result.status == "ok" and result.weed_count == 2 and state.instances["s0-i1"].role == "crop"


def test_call_limit_closes_stop_and_flags_unclassified_weeds():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i3"})], "")])
    result, _ = _run(llm, settings=replace(SETTINGS, max_tool_calls=3))
    assert result.status == "ok" and result.weed_count == 2
    assert any(step.tool == "limit" for step in result.steps)
    assert result.flags == ["Check s0-i3: marked but never classified", "Check s0-i4: marked but never classified"]


PATCH = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [400, 750, 1000, 1000], "label": "foxtail"}]})


def test_patch_makes_the_count_approximate():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, PATCH], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("top_down")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is True
    assert "Weed count is approximate: some weeds merged into patches" in result.flags


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
    ticks = itertools.count(0, 100)  # every clock reading is 100 s later: the 240 s budget is gone after one turn
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
