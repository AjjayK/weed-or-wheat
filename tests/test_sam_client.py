import numpy as np
import pytest

from scout.retry import BadRequest
from scout.vision.sam_client import SamClient
from tests.helpers import make_image, sam_text


def test_segment_concept_builds_request_and_parses():
    image = make_image(64, 48)
    sent = []

    def send(payload):
        sent.append(payload)
        return sam_text([("0", (1, 2, 5, 6), np.ones((4, 4), bool), 0.8)], image.size)

    records = SamClient(send, model="sam-3.1", sleep=lambda s: None).segment_concept(image, "  plant ")
    assert len(records) == 1 and records[0].confidence == 0.8
    content = sent[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "plant"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert sent[0]["model"] == "sam-3.1"
    assert sent[0]["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}


def test_repeated_prompt_uses_cache():
    image = make_image(32, 32)
    calls = []
    sam = SamClient(lambda p: calls.append(p) or "", sleep=lambda s: None)
    sam.segment_concept(image, "sky")
    sam.segment_concept(image, "sky")
    assert len(calls) == 1


def test_bad_request_raises():
    class Rejected(Exception):
        status_code = 400

    def send(payload):
        raise Rejected("no")

    with pytest.raises(BadRequest):
        SamClient(send, sleep=lambda s: None).segment_concept(make_image(32, 32), "plant")


def test_segment_at_sends_object_prompt():
    image = make_image(100, 80)
    sent = []
    sam = SamClient(lambda p: sent.append(p) or "", sleep=lambda s: None)
    sam.segment_at(image, points=[(10, 20, True), (30, 40, False)])
    text = sent[0]["input"][0]["content"][0]["text"]
    assert text == "<0f>1<|point;x=10;y=20;w=100;h=80|>-<|point;x=30;y=40;w=100;h=80|>"
    sam.segment_at(image, box=(5, 6, 50, 60))
    assert sent[1]["input"][0]["content"][0]["text"] == "<0f>1<|box;x1=5;y1=6;x2=49;y2=59;w=100;h=80|>"
