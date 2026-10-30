import numpy as np

from scout.vision import masks as mk


def test_paste_clips_to_canvas():
    canvas = mk.paste(np.ones((4, 4), dtype=bool), (-2, 8, 2, 12), (10, 10))
    assert canvas.sum() == 4
    assert canvas[8:10, 0:2].all()


def test_union_and_bbox():
    a = mk.paste(np.ones((2, 2), bool), (1, 1, 3, 3), (10, 10))
    b = mk.paste(np.ones((2, 2), bool), (6, 5, 8, 7), (10, 10))
    both = mk.union([a, b], (10, 10))
    assert both.sum() == 8
    assert mk.bbox(both) == (1, 1, 8, 7)


def test_bbox_of_empty_mask():
    assert mk.bbox(np.zeros((5, 5), bool)) == (0, 0, 0, 0)


def test_pieces_splits_blobs_and_drops_small():
    canvas = np.zeros((40, 40), bool)
    canvas[2:12, 2:12] = True
    canvas[20:30, 25:35] = True
    canvas[35, 35] = True
    found = mk.pieces(canvas, min_area=20)
    assert sorted(int(p.sum()) for p in found) == [100, 100]


def test_pieces_drops_slivers_along_excluded_edges():
    crop = np.zeros((60, 60), bool)
    crop[10:40, 10:40] = True
    plant = np.zeros((60, 60), bool)
    plant[8:42, 8:42] = True
    assert mk.pieces(plant, min_area=5, exclude=crop) == []


def test_pieces_keeps_thin_blades():
    blade = np.zeros((60, 60), bool)
    for y in range(5, 55):
        blade[y, y - 2:y + 1] = True
    assert [int(p.sum()) for p in mk.pieces(blade, min_area=20)] == [int(blade.sum())]


def test_pad_box_pads_thin_boxes_on_both_axes():
    assert mk.pad_box((50, 10, 53, 110), 0.15, (200, 200)) == (35, 0, 68, 125)


def test_pad_box_clamps():
    assert mk.pad_box((0, 0, 10, 10), 0.5, (12, 12)) == (0, 0, 12, 12)
    assert mk.pad_box((10, 10, 20, 20), 0.1, (100, 100)) == (9, 9, 21, 21)
