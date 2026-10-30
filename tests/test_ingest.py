import io

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from scout.app.ingest import gps_from_ifd, load_image
from tests.helpers import png_bytes


def test_load_image_makes_full_and_work_sizes():
    image = load_image(3, png_bytes(3000, 2000))
    assert image.index == 3
    assert max(image.full.size) == 2048 and max(image.work.size) == 1024
    assert image.size == image.work.size
    assert len(image.hash) == 16 and image.gps is None


def test_exif_orientation_is_applied():
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20)).save(buf, format="JPEG", exif=exif)
    assert load_image(0, buf.getvalue()).work.size == (20, 40)


def test_gps_from_ifd():
    assert gps_from_ifd({1: "N", 2: (41.0, 30.0, 0.0), 3: "W", 4: (93.0, 36.0, 0.0)}) == (41.5, -93.6)
    assert gps_from_ifd({}) is None


def test_work_data_url_is_jpeg():
    assert load_image(0, png_bytes(50, 50)).work_data_url.startswith("data:image/jpeg;base64,")


def test_load_image_reads_gps_from_exif():
    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: "S", 2: (IFDRational(41, 1), IFDRational(30, 1), IFDRational(0, 1)),
                                 3: "W", 4: (IFDRational(93, 1), IFDRational(36, 1), IFDRational(0, 1))})
    buf = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buf, format="JPEG", exif=exif)
    assert load_image(0, buf.getvalue()).gps == (-41.5, -93.6)


def test_gps_from_ifd_rejects_unknown_and_impossible_values():
    unknown = IFDRational(0, 0)  # Pillow reads 0/0 as NaN
    assert gps_from_ifd({1: "N", 2: (unknown,) * 3, 3: "W", 4: (unknown,) * 3}) is None
    assert gps_from_ifd({1: "N", 2: (0.0, 0.0, 0.0), 3: "E", 4: (0.0, 0.0, 0.0)}) is None
    assert gps_from_ifd({1: "N", 2: (95.0, 0.0, 0.0), 3: "W", 4: (93.0, 0.0, 0.0)}) is None


def test_heic_is_rotated_exactly_once():
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buf, format="HEIF", exif=exif.tobytes())
    assert load_image(0, buf.getvalue()).work.size == (20, 40)


def test_small_rgba_image_is_not_upscaled_and_becomes_rgb():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 30)).save(buf, format="PNG")
    image = load_image(0, buf.getvalue())
    assert image.full.size == image.work.size == (50, 30) and image.work.mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image", png_bytes(500, 500)[:100]])
def test_unreadable_bytes_raise(data):
    with pytest.raises((OSError, ValueError)):
        load_image(0, data)


def test_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setattr("scout.app.ingest.MAX_PIXELS", 99)
    with pytest.raises(ValueError):
        load_image(0, png_bytes(10, 10))


def test_rejects_formats_outside_jpeg_png_heic():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="TIFF")
    with pytest.raises(OSError):
        load_image(0, buf.getvalue())
