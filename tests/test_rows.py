import numpy as np

from scout.vision.rows import band_mask, canopy_band, fit_rows


def _two_rows(jitter=False):
    left = [(100 + (i % 2 if jitter else 0), 20 + 30 * i) for i in range(5)]
    right = [(300, 20 + 30 * i) for i in range(5)]
    return left + right


def test_fits_two_vertical_rows():
    fit = fit_rows(_two_rows(jitter=True), [20] * 10)
    assert fit is not None and fit.orientation == "vertical" and len(fit.lines) == 2
    intercepts = sorted(round(b) for _, b in fit.lines)
    assert intercepts[0] in (100, 101) and intercepts[1] == 300


def test_too_few_plants_gives_no_rows():
    assert fit_rows([(10, 10), (20, 20), (30, 30)], [5, 5, 5]) is None


def test_band_mask_covers_rows_only():
    band = band_mask(fit_rows(_two_rows(), [20] * 10), (400, 200))
    assert band[50, 100] and band[50, 300] and not band[50, 200]


def _strip(x0, x1, shape=(100, 100)):
    mask = np.zeros(shape, bool)
    mask[:, x0:x1] = True
    return mask


def test_canopy_band_is_canopy_plus_margin():
    band = canopy_band(_strip(10, 30) | _strip(60, 80))
    assert band[50, 10] and band[50, 8] and band[50, 32]
    assert not band[50, 35] and not band[50, 45]


def test_canopy_band_fills_weed_holes_not_shut_in_furrows():
    crop = _strip(10, 40)
    crop[50:55, 20:25] = False
    shut = _strip(60, 70) | _strip(90, 100)
    shut[0:5, 60:100] = shut[95:100, 60:100] = True
    band = canopy_band(crop | shut)
    assert band[52, 22] and not band[50, 80]


def test_canopy_band_none_without_crop_or_between_row_ground():
    assert canopy_band(np.zeros((50, 50), bool)) is None
    assert canopy_band(np.ones((50, 50), bool)) is None
