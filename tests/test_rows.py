from scout.vision.rows import band_mask, fit_rows


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
