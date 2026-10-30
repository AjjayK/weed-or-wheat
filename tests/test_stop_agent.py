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
    assert any(part.get("type") == "message" for part in third["input"])  # zoom image rides in a user message


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


def test_patch_makes_the_count_approximate():
    patch = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [400, 750, 1000, 1000], "label": "foxtail"}]})
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, patch], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("close_up")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is True
    assert "Weed count is approximate: some weeds merged into patches" in result.flags


def test_repeated_bad_request_fails_stop_and_keeps_steps():
    call = [ToolCall("c1", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "w"}]})]
    llm = FakeLlm([LlmTurn("r1", call, ""), LlmTurn("r2", call, "")])
    result, _ = _run(llm, sam=FakeSam(fail={"boxes"}))
    assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "mark", "error"]


def test_model_errors_fail_the_stop_and_keep_steps():
    for error in (LlmError("response r2 was incomplete"), BadRequest("image too large")):
        result, _ = _run(FakeLlm([LlmTurn("r1", [MARK_CROP], ""), error]))
        assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "error"]


def test_prose_reply_is_nudged_then_closed():
    llm = FakeLlm([LlmTurn("r1", [], "I think there are weeds."), LlmTurn("r2", [], "Still prose.")])
    result, _ = _run(llm)
    assert result.status == "no_crop_found" and len(llm.requests) == 2


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
