import numpy as np
import pytest

from scout.models import Instance
from scout.vision.measure import compute_metrics, footprint_m2, species_key

SIZE = (100, 100)


def _inst(iid, role, box, species=None, prompt="plant", height=None):
    return Instance(id=iid, source="concept", prompt=prompt, box=box, area_px=0, role=role,
                    species_id=species, height_in=height)


def _mask(box):
    m = np.zeros((100, 100), bool)
    left, top, right, bottom = box
    m[top:bottom, left:right] = True
    return m


def _setup():
    boxes = {"c": (10, 10, 30, 30), "w1": (60, 60, 70, 70), "w2": (80, 10, 90, 20)}
    instances = [_inst("c", "crop", boxes["c"]),
                 _inst("w1", "weed", boxes["w1"], "waterhemp", height=3.0),
                 _inst("w2", "weed", boxes["w2"], None, prompt="grass")]
    return instances, {key: _mask(box) for key, box in boxes.items()}


def test_cover_and_counts_without_scale():
    instances, masks = _setup()
    m = compute_metrics(instances, masks, SIZE, "row")
    assert m.weed_cover_pct == 2.0 and m.crop_cover_pct == 4.0 and m.weed_count == 2
    assert m.species == {"waterhemp": 1, "unidentified_grass": 1}
    assert m.species_cover_pct == {"waterhemp": 1.0, "unidentified_grass": 1.0}
    assert m.density_per_m2 is None and m.species_density is None
    assert m.max_height_in == 3.0 and m.species_max_height_in == {"waterhemp": 3.0}


def test_sky_is_removed_from_ground():
    instances, masks = _setup()
    sky = np.zeros((100, 100), bool)
    sky[:50, :] = True
    m = compute_metrics(instances, masks, SIZE, "row", sky=sky)
    assert m.weed_cover_pct == 2.0  # 100 weed px on 5000 ground px
    assert m.weed_count == 1 and m.species == {"waterhemp": 1}


def test_top_down_density():
    instances, masks = _setup()
    m = compute_metrics(instances, masks, SIZE, "top_down")
    assert m.density_per_m2 == pytest.approx(0.735, abs=0.01)
    assert footprint_m2(SIZE, "top_down") == pytest.approx(2.72, abs=0.01)
    assert footprint_m2(SIZE, "row") is None


def test_in_row_share_and_approximate_count():
    instances, masks = _setup()
    band = np.zeros((100, 100), bool)
    band[:, 55:75] = True
    assert compute_metrics(instances, masks, SIZE, "row", band=band).in_row_weed_share == 0.5
    big = compute_metrics([_inst("w3", "weed", (0, 40, 50, 100))], {"w3": _mask((0, 40, 50, 100))}, SIZE, "row")
    assert big.count_approximate is True


def _boxed(iid, role, species=None, source="prompt", prompt="box", height=None):
    return Instance(id=iid, source=source, prompt=prompt, box=(0, 0, 1, 1), area_px=0, role=role,
                    species_id=species, height_in=height)


def test_footprint_is_the_same_portrait_or_landscape():
    assert footprint_m2((4032, 3024), "top_down") == pytest.approx(2.04, abs=0.01)
    assert footprint_m2((3024, 4032), "top_down") == pytest.approx(footprint_m2((4032, 3024), "top_down"))
    assert footprint_m2((0, 0), "top_down") is None


def test_boxed_weed_inside_row_canopy_wins_over_crop():
    row, weed = _mask((40, 0, 60, 100)), _mask((45, 40, 55, 50))
    m = compute_metrics([_boxed("c", "crop"), _boxed("w", "weed", "waterhemp", height=4.0)], {"c": row, "w": weed},
                        SIZE, "row", band=_mask((35, 0, 65, 100)))
    assert (m.weed_cover_pct, m.crop_cover_pct, m.weed_count) == (1.0, 19.0, 1)
    assert m.in_row_weed_share == 1.0 and m.max_height_in == 4.0


def test_weed_over_crop_edge_counts_once_as_weed():
    m = compute_metrics([_boxed("c", "crop"), _boxed("w", "weed", "waterhemp")],
                        {"c": _mask((0, 0, 10, 10)), "w": _mask((5, 0, 15, 10))}, SIZE, "row")
    assert (m.weed_cover_pct, m.crop_cover_pct) == (1.0, 0.5)


def test_overlapping_weeds_count_each_pixel_once():
    m = compute_metrics([_boxed("a", "weed", "waterhemp"), _boxed("b", "weed", "giant_ragweed")],
                        {"a": _mask((0, 0, 10, 10)), "b": _mask((5, 0, 15, 10))}, SIZE, "row")
    assert m.weed_cover_pct == 1.5 and m.species_cover_pct == {"waterhemp": 1.0, "giant_ragweed": 0.5}
    dup = compute_metrics([_boxed("a", "weed", "waterhemp"), _boxed("b", "weed", "waterhemp")],
                          {"a": _mask((0, 0, 10, 10)), "b": _mask((2, 2, 12, 12))}, SIZE, "row")
    assert dup.weed_count == 1  # a second box mostly on the same plant


def test_sky_rules():
    weed = {"w": _mask((0, 60, 10, 70))}
    half_sky = _mask((0, 0, 100, 50))
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "top_down", sky=half_sky).weed_cover_pct == 1.0
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "row", sky=half_sky).weed_cover_pct == 2.0
    all_sky = np.ones((100, 100), bool)
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "wide", sky=all_sky).weed_cover_pct == 1.0


def test_roles_missing_masks_and_density_semantics():
    instances = [_boxed("u", "unknown"), _boxed("i", "ignore"), _boxed("gone", "weed")]
    masks = {"u": _mask((0, 0, 10, 10)), "i": _mask((20, 0, 30, 10))}
    m = compute_metrics(instances, masks, SIZE, "top_down")
    assert m.weed_count == 0 and m.density_per_m2 == 0.0 and m.species_density == {}  # a top-down stop with no weeds
    assert m.in_row_weed_share is None
    assert compute_metrics(instances, masks, SIZE, "close_up").species_density is None


def test_approximate_only_for_big_pieces_outside_close_ups():
    small = compute_metrics([_boxed("w", "weed")], {"w": _mask((0, 0, 10, 10))}, SIZE, "row")
    assert small.count_approximate is False
    big = {"w": _mask((0, 0, 30, 30))}
    assert compute_metrics([_boxed("w", "weed")], big, SIZE, "row").count_approximate is True
    assert compute_metrics([_boxed("w", "weed")], big, SIZE, "close_up").count_approximate is False


def test_grass_words_in_labels_mean_unidentified_grass():
    assert species_key(_boxed("a", "weed", prompt="giant foxtail")) == "unidentified_grass"
    assert species_key(_boxed("b", "weed", prompt="pigweed seedling")) == "unidentified_broadleaf"


def test_a_weed_area_counts_its_pieces_and_makes_the_count_approximate():
    instances, masks = _setup()
    area = Instance(id="p", source="patch", prompt="horsetail", box=(40, 40, 50, 50), area_px=100, role="weed",
                    species_id="waterhemp", plant_count=7)
    masks["p"] = _mask((40, 40, 50, 50))
    m = compute_metrics(instances + [area], masks, SIZE, "close_up")
    assert m.weed_count == 9 and m.species["waterhemp"] == 8 and m.count_approximate is True
    assert m.weed_cover_pct == 3.0  # w1, w2 and the area: 100 px each


def test_a_weed_area_counts_only_the_pieces_no_box_already_counted():
    box = Instance(id="b", source="prompt", prompt="horsetail clump", box=(40, 40, 58, 70), area_px=540, role="weed",
                   species_id="waterhemp")
    area = Instance(id="p", source="patch", prompt="horsetail", box=(40, 40, 70, 70), area_px=900, role="weed",
                    species_id="waterhemp", plant_count=10)
    masks = {"b": _mask((40, 40, 58, 70)), "p": _mask((40, 40, 70, 70))}
    m = compute_metrics([box, area], masks, SIZE, "close_up")
    assert m.weed_count == 5 and m.weed_cover_pct == 9.0  # 1 boxed plant + 10 pieces x 360 of 900 px not yet counted
    m = compute_metrics([area, box], masks, SIZE, "close_up")
    assert m.weed_count == 5 and m.weed_cover_pct == 9.0  # order no longer matters: the box is measured first


def test_a_boxed_weed_inside_a_weed_area_keeps_its_species_and_height_in_any_order():
    ragweed = Instance(id="b", source="prompt", prompt="giant ragweed", box=(40, 40, 58, 70), area_px=540,
                       role="weed", species_id="giant_ragweed", height_in=30.0)
    area = Instance(id="p", source="patch", prompt="horsetail", box=(40, 40, 70, 70), area_px=900, role="weed",
                    species_id="field_horsetail", plant_count=10)
    masks = {"b": _mask((40, 40, 58, 70)), "p": _mask((40, 40, 70, 70))}
    for order in ([ragweed, area], [area, ragweed]):
        m = compute_metrics(order, masks, SIZE, "close_up")
        assert m.species == {"giant_ragweed": 1, "field_horsetail": 4} and m.max_height_in == 30.0
