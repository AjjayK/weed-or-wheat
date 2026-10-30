import numpy as np
import pytest

from scout.models import Instance
from scout.vision.measure import compute_metrics, footprint_m2

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
