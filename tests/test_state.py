import numpy as np
import pytest

from scout.agent.state import StopState, to_pixels, to_thousandths
from tests.helpers import field_scene, make_image, mask_record, rect


def _state(crop="soybean"):
    return StopState(make_image(100, 100), crop)


def test_boxes_convert_between_thousandths_and_pixels():
    assert to_pixels([100, 200, 300, 400], (100, 50)) == (10, 10, 30, 20)
    assert to_thousandths((10, 10, 30, 20), (100, 50)) == [100, 200, 300, 400]
    assert to_pixels([300, 400, 100, 200], (100, 50)) == (10, 10, 30, 20)  # corners in either order
    assert to_pixels([-50, 990, 1200, 1000], (100, 100)) == (0, 98, 100, 100)  # clamped, still 2 px tall
    assert to_pixels([500.4, 500, 500, 500], (100, 100)) == (50, 50, 52, 52)  # grown to 2 px a side
    with pytest.raises(ValueError):
        to_pixels([1, 2, 3], (100, 100))


def test_weed_on_a_crop_plant_is_dropped_but_weeds_in_a_row_canopy_are_flagged():
    state, scene = _state(), field_scene()
    state.add_mask(scene["row"], "prompt", "soybean row", role="crop")
    state.add_mask(scene["plant"], "prompt", "soybean plant", role="crop")
    assert state.add_weed(scene["plant"] & rect(42, 32, 50, 40), "prompt", "pigweed") is None
    in_row = state.add_weed(scene["row"] & rect(12, 50, 22, 60), "prompt", "leaf in the row")
    edge = state.add_weed(scene["edge_weed"], "prompt", "pigweed")
    clear = state.add_weed(scene["weed_a"], "prompt", "waterhemp")
    assert in_row.role == edge.role == clear.role == "weed" and edge.id_confidence is None
    assert in_row.check_this and edge.check_this and not clear.check_this
    assert edge.area_px == 100  # the whole mask is kept, including the part inside the row
    assert state.unclassified() == [in_row.id, edge.id, clear.id] == ["s0-i3", "s0-i4", "s0-i5"]


def test_large_weed_mask_is_kept_as_an_approximate_patch():
    state = _state()
    assert not state.count_approximate()
    patch = state.add_weed(field_scene()["patch"], "prompt", "foxtail patch")
    assert patch.role == "weed" and patch.check_this and state.is_patch(patch)
    assert state.has_patch() and state.count_approximate()


def test_row_band_from_a_row_canopy_leaves_out_sky():
    state = _state()
    assert state.row_band() is None  # no crop yet
    state.add_mask(field_scene()["row"], "prompt", "soybean row", role="crop")
    band = state.row_band()
    assert band[50, 8] and band[50, 31] and not band[50, 40]
    state.sky = rect(0, 0, 100, 20)
    band = state.row_band()
    assert not band[10, 20] and band[50, 20]


def _plants_in_two_rows(state):
    for x in (17, 67):
        for y in (10, 40, 70):
            state.add_mask(rect(x, y, x + 6, y + 6), "prompt", "soybean plant", role="crop")


def test_row_band_fits_lines_through_separate_plants():
    state = _state()
    _plants_in_two_rows(state)
    band = state.row_band()
    assert band[50, 20] and band[50, 70] and not band[50, 45]


def test_row_band_is_none_for_alfalfa():
    state = _state("alfalfa")
    _plants_in_two_rows(state)
    state.add_mask(field_scene()["row"], "prompt", "alfalfa", role="crop")
    assert state.row_band() is None


def test_overlay_paints_weeds_over_crop():
    state, scene = _state(), field_scene()
    state.add_mask(scene["row"], "prompt", "soybean row", role="crop")
    state.add_mask(scene["edge_weed"], "prompt", "pigweed", role="weed")
    pixels = np.asarray(state.overlay())
    # (45, 28) is in the row and the weed, (45, 31) only in the weed, (45, 20) only in the row; all are in the band
    assert tuple(pixels[45, 28]) == tuple(pixels[45, 31]) != tuple(pixels[45, 20])


def test_log_reports_steps():
    seen = []
    state = StopState(make_image(), "corn", on_step=seen.append, clock=lambda: 0.0)
    state.log("mark", "Mark crop: 2 of 2 boxes kept", ["s0-i1"])
    assert seen[0].summary.startswith("Mark") and seen[0].t_ms == 0


def test_overlay_renders_without_instances():
    assert StopState(make_image(40, 30), "corn").overlay().size == (40, 30)


def test_patch_is_approximate_except_in_a_close_up():
    state = _state()
    state.add_weed(field_scene()["patch"], "prompt", "foxtail patch")
    assert state.count_approximate() and state.count_approximate("row") and state.count_approximate("top_down")
    assert not state.count_approximate("close_up")  # a few big plants fill a close-up
    state.box_cap_hit = True
    assert state.count_approximate("close_up")  # a floor, whatever the view


class _SkySam:
    def __init__(self, records=None, error=None):
        self.records, self.error, self.calls = records or [], error, 0

    def segment_concept(self, image, phrase):
        self.calls += 1
        if self.error:
            raise self.error
        return self.records


def test_ensure_sky_asks_once():
    state, sam = _state(), _SkySam([mask_record("0", rect(0, 0, 100, 20))])
    state.ensure_sky(sam)
    state.ensure_sky(sam)
    assert sam.calls == 1 and state.sky[10, 50] and not state.sky[50, 50] and state.steps == []


def test_ensure_sky_is_best_effort_and_not_retried():
    state, sam = _state(), _SkySam(error=TimeoutError("timed out"))
    state.ensure_sky(sam)
    state.ensure_sky(sam)
    assert sam.calls == 1 and state.sky is None
    assert [(s.tool, s.ok) for s in state.steps] == [("sky", False)]
    assert "TimeoutError" in state.steps[0].summary
