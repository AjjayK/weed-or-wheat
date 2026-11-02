import numpy as np
from PIL import Image, ImageDraw

from scout.vision.patches import green_mask, patch_mask

SOIL, LEAF, SHADOW, STRAW = (110, 80, 50), (60, 140, 40), (40, 40, 40), (200, 170, 120)


def _photo():
    """100x100 soil with two green shoots (5x20 px each), a shadow, straw, and a 9 px green speck."""
    image = Image.new("RGB", (100, 100), SOIL)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 24, 39), fill=LEAF)
    draw.rectangle((40, 20, 44, 39), fill=LEAF)
    draw.rectangle((60, 60, 69, 69), fill=SHADOW)
    draw.rectangle((70, 20, 79, 29), fill=STRAW)
    draw.rectangle((85, 85, 87, 87), fill=LEAF)
    return image


def test_green_mask_keeps_leaves_and_drops_soil_shadow_and_straw():
    green = green_mask(_photo())
    assert green[30, 22] and green[30, 42] and green[86, 86]
    assert not green[50, 50] and not green[65, 65] and not green[25, 75]


def test_patch_counts_green_pieces_inside_the_box():
    green = green_mask(_photo())
    no_crop = np.zeros((100, 100), bool)
    mask, pieces = patch_mask(green, (10, 10, 95, 95), no_crop)
    assert pieces == 2 and mask.sum() == 200  # the 9 px speck is noise
    mask, pieces = patch_mask(green, (10, 10, 30, 50), no_crop)
    assert pieces == 1 and mask.sum() == 100  # only the first shoot is inside


def test_patch_leaves_out_crop_and_a_margin_around_it():
    green = green_mask(_photo())
    crop = np.zeros((100, 100), bool)
    crop[20:40, 40:45] = True  # the second shoot is really a crop leaf
    mask, pieces = patch_mask(green, (10, 10, 95, 95), crop)
    assert pieces == 1 and mask.sum() == 100
    crop = np.zeros((100, 100), bool)
    crop[20:40, 46:50] = True  # crop 1 px right of the second shoot: the 3 px margin takes its columns 43-44
    mask, pieces = patch_mask(green, (10, 10, 95, 95), crop)
    assert pieces == 2 and mask.sum() == 160
