from types import SimpleNamespace as NS

from scout.agent.llm import Llm, user_message


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
