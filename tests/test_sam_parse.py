import numpy as np

from scout.vision.sam_parse import parse_sam_output
from tests.helpers import sam_text


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
    assert record.mask.shape == (4, 4) and record.mask.all()


def test_empty_output_gives_no_records():
    assert parse_sam_output("") == []
