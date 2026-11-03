from types import SimpleNamespace as NS

import pytest

from scout.agent.llm import Llm, LlmError, LlmTurn, is_lost_response, user_message
from scout.retry import BadRequest


def _response(output, text="", rid="resp_1"):
    return NS(id=rid, output=output, output_text=text)


def test_turn_parses_function_calls():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _response([NS(type="reasoning"),
                          NS(type="function_call", call_id="c1", name="segment",
                             arguments='{"phrase": "plant", "kind": "all_plants"}')])

    turn = Llm(create, "muse-spark-1.3").turn([user_message("hi")], instructions="be brief",
                                             tools=[{"type": "function", "name": "segment"}],
                                             previous_response_id="resp_0")
    assert turn.response_id == "resp_1"
    assert turn.calls[0].name == "segment"
    assert turn.calls[0].arguments == {"phrase": "plant", "kind": "all_plants"}
    assert seen["previous_response_id"] == "resp_0" and seen["instructions"] == "be brief"
    assert seen["model"] == "muse-spark-1.3"


def test_invalid_arguments_are_flagged():
    create = lambda **k: _response([NS(type="function_call", call_id="c1", name="zoom", arguments="{not json")])
    assert "__invalid__" in Llm(create, "m").turn("x").calls[0].arguments


def test_text_format_is_passed_as_text_param():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _response([], text='{"records": []}')

    turn = Llm(create, "m").turn("x", text_format={"type": "json_schema", "name": "r", "schema": {}})
    assert seen["text"] == {"format": {"type": "json_schema", "name": "r", "schema": {}}}
    assert turn.text == '{"records": []}' and turn.calls == []


def test_user_message_with_images_and_file():
    msg = user_message("look", ["data:image/jpeg;base64,AAA"], file=("label.pdf", "data:application/pdf;base64,BBB"))
    assert [part["type"] for part in msg["content"]] == ["input_text", "input_image", "input_file"]
    assert msg["role"] == "user" and msg["type"] == "message"


@pytest.mark.parametrize("raw", ["{not json", "[]", '"x"', "null", "5"])
def test_non_object_arguments_are_flagged(raw):
    create = lambda **k: _response([NS(type="function_call", call_id="c1", name="zoom", arguments=raw)])
    assert Llm(create, "m").turn("x").calls[0].arguments == {"__invalid__": raw}


def test_unfinished_response_raises():
    create = lambda **k: NS(id="r", output=[], output_text="", status="incomplete",
                            incomplete_details=NS(reason="max_output_tokens"), error=None)
    with pytest.raises(LlmError, match="incomplete"):
        Llm(create, "m").turn("x")


def test_real_sdk_response_is_parsed():
    from openai._models import construct_type
    from openai.types.responses import Response
    response = construct_type(type_=Response, value={
        "id": "resp_1", "object": "response", "created_at": 0, "model": "m", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                   {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "zoom",
                    "arguments": '{"instance_id": "s0-i1"}', "status": "completed"},
                   {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}]}]})
    turn = Llm(lambda **k: response, "m").turn("x")
    assert turn.response_id == "resp_1" and turn.text == "done"
    assert [(c.call_id, c.name, c.arguments) for c in turn.calls] == [("c1", "zoom", {"instance_id": "s0-i1"})]


def test_output_items_can_be_replayed_as_input():
    from openai._models import construct_type
    from openai.types.responses import Response
    response = construct_type(type_=Response, value={
        "id": "resp_1", "object": "response", "created_at": 0, "model": "m", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                   {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "zoom",
                    "arguments": '{"instance_id":  "s0-i1"}', "status": "completed"},
                   {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "looking", "annotations": []}]}]})
    turn = Llm(lambda **k: response, "m").turn("x")
    assert turn.output_items == [
        {"type": "function_call", "call_id": "c1", "name": "zoom", "arguments": '{"instance_id":  "s0-i1"}'},
        {"role": "assistant", "content": "looking"}]  # the raw argument string; reasoning is left out


def test_output_items_default_to_empty():
    assert LlmTurn("r1", [], "").output_items == []
    turn = Llm(lambda **k: _response([NS(type="reasoning")]), "m").turn("x")
    assert turn.output_items == []


def test_is_lost_response():
    lost = BadRequest("Error code: 400 - {'error': {'message': 'referenced response not found or expired', "
                      "'type': 'invalid_request_error'}}")
    assert is_lost_response(lost)
    assert is_lost_response(BadRequest("Referenced Response Not Found"))
    assert not is_lost_response(BadRequest("image too large"))
    assert not is_lost_response(ValueError("referenced response not found or expired"))  # only a rejected request


def test_reasoning_summaries_are_asked_for_and_read():
    seen = {}

    def create(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return _response([NS(type="reasoning", summary=[NS(type="summary_text", text="Counting weeds."),
                                                         NS(type="summary_text", text="Boxing the rows.")]),
                          NS(type="message", role="assistant",
                             content=[NS(type="output_text", text="Scouting the rows.")]),
                          NS(type="function_call", call_id="c1", name="survey", arguments="{}")],
                         text="Scouting the rows.")

    turn = Llm(create, "m").turn("x", reasoning_summary=True)
    assert seen["reasoning"] == {"summary": "auto"}
    assert turn.reasoning == ["Counting weeds.", "Boxing the rows."] and turn.text == "Scouting the rows."
    Llm(create, "m").turn("x")
    assert "reasoning" not in seen


def test_reasoning_without_summaries_is_empty():
    turn = Llm(lambda **k: _response([NS(type="reasoning")]), "m").turn("x")
    assert turn.reasoning == []
