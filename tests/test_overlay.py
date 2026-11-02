import numpy as np
from PIL import Image

from scout.vision.overlay import AREA_RGB, FLAG_RGB, render_overlay


def test_overlay_tints_crop_green_weed_magenta_and_outlines_flags():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    crop = np.zeros((10, 10), bool)
    crop[0, 0] = True
    weed = np.zeros((10, 10), bool)
    weed[9, 9] = True
    out = render_overlay(image, crop, weed, flagged=[(2, 2, 9, 9)])
    r, g, b = out.getpixel((0, 0))
    assert g > r and g > b
    r, g, b = out.getpixel((9, 9))
    assert r > g and b > g
    assert out.getpixel((4, 4)) == FLAG_RGB
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_overlay_shades_band_under_plants_and_skips_empty_boxes():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    band = np.zeros((10, 10), bool)
    band[:, 0] = True
    weed = np.zeros((10, 10), bool)
    weed[0, 0] = True
    out = render_overlay(image, np.zeros((10, 10), bool), weed, band, flagged=[(0, 0, 0, 0)])
    assert out.getpixel((0, 5)) == (225, 225, 223)
    r, g, b = out.getpixel((0, 0))
    assert r > g and b > g
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_overlay_outlines_weed_areas_in_blue_under_flag_boxes():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    empty = np.zeros((10, 10), bool)
    out = render_overlay(image, empty, empty, areas=[(0, 0, 10, 10)], flagged=[(0, 0, 5, 5)])
    assert out.getpixel((9, 9)) == AREA_RGB
    assert out.getpixel((0, 0)) == FLAG_RGB  # a flag box drawn later stays visible
    assert out.getpixel((5, 5)) == (255, 255, 255)
