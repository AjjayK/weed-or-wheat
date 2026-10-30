import numpy as np

from scout.vision.sam_parse import parse_sam_output
from tests.helpers import sam_record_text, sam_text


def _corrupt(record):
    """Change the payload's last digit so the strict one_bit decode rejects it."""
    return record[:-3] + ("A" if record[-3] != "A" else "B") + record[-2:]


def test_parses_two_records_with_confidence():
    first = np.zeros((4, 6), bool)
    first[1:3, 2:5] = True
    second = np.ones((3, 3), bool)
    text = sam_text([("0", (10, 20, 16, 24), first, 0.91), ("1", (0, 0, 3, 3), second, 0.5)], (100, 80))
    records = parse_sam_output(text)
    assert [r.object_id for r in records] == ["0", "1"]
    assert records[0].box == (10, 20, 16, 24)
    assert records[0].confidence == 0.91
    assert np.array_equal(records[0].mask, first)
    full = records[0].full_mask((100, 80))
    assert full.shape == (80, 100) and full.sum() == first.sum() and full[21, 12]


def test_resizes_mask_that_does_not_match_box():
    text = sam_text([("0", (0, 0, 4, 4), np.ones((2, 2), bool))], (10, 10))
    record = parse_sam_output(text)[0]
    assert record.mask.shape == (4, 4) and record.mask.all() and record.resized


def test_resize_keeps_orientation():
    pattern = np.array([[1, 0, 0], [0, 0, 1]], bool)
    record = parse_sam_output(sam_text([("0", (0, 0, 6, 4), pattern)], (10, 10)))[0]
    assert record.mask.shape == (4, 6) and record.mask[0, 0] and record.mask[3, 5] and not record.mask[0, 5]


def test_empty_output_gives_no_records():
    assert parse_sam_output("") == []


def test_bad_record_is_skipped_and_logged(caplog):
    good = sam_record_text("1", (0, 0, 3, 3), np.ones((3, 3), bool), (10, 10))
    bad = _corrupt(sam_record_text("0", (5, 5, 9, 9), np.eye(4, dtype=bool), (10, 10)))
    assert [r.object_id for r in parse_sam_output("<0f>" + bad + "," + good)] == ["1"]
    assert "kept 1 records" in caplog.text


def test_truncated_last_record_is_counted(caplog):
    text = sam_text([("0", (0, 0, 3, 3), np.ones((3, 3), bool)), ("1", (4, 4, 7, 7), np.ones((3, 3), bool))], (10, 10))
    assert [r.object_id for r in parse_sam_output(text[:-2])] == ["0"]
    assert "1 record(s) not in" in caplog.text


def test_fields_in_any_order_with_spaces():
    payload = sam_record_text("0", (2, 2, 5, 5), np.ones((3, 3), bool), (10, 10)).split("data=3,3,")[1][:-2]
    text = f"<0f>0<|box; y1 = 2 ;x1=2;x2=4;y2=4;w=10;h=10|><|mask;data=3,3,{payload}; c = 0.7 ;x=0;y=0|>"
    record = parse_sam_output(text)[0]
    assert record.box == (2, 2, 5, 5) and record.confidence == 0.7 and record.mask.all()
