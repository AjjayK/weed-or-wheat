import io

from PIL import Image

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
