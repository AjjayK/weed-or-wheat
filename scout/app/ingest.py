"""Turn uploaded bytes into a StopImage: rotation applied, two sizes, GPS, and a content hash."""
import base64
import hashlib
import io
from dataclasses import dataclass
from functools import cached_property

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

GPS_IFD = 0x8825


def to_data_url(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class StopImage:
    index: int
    hash: str
    full: Image.Image  # long side <= full_max_px, for zoom crops
    work: Image.Image  # long side <= work_max_px, what SAM and the agent see
    gps: tuple[float, float] | None

    @property
    def size(self) -> tuple[int, int]:
        return self.work.size

    @cached_property
    def work_data_url(self) -> str:
        return to_data_url(self.work)


def _shrink(image: Image.Image, max_px: int) -> Image.Image:
    out = image.copy()
    out.thumbnail((max_px, max_px))
    return out


def _dms(value) -> float:
    degrees, minutes, seconds = (float(v) for v in value)
    return degrees + minutes / 60 + seconds / 3600


def gps_from_ifd(gps: dict) -> tuple[float, float] | None:
    try:
        lat, lon = _dms(gps[2]), _dms(gps[4])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return (round(lat, 6), round(lon, 6))


def load_image(index: int, data: bytes, full_max_px: int = 2048, work_max_px: int = 1024) -> StopImage:
    image = Image.open(io.BytesIO(data))
    gps = gps_from_ifd(dict(image.getexif().get_ifd(GPS_IFD)))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return StopImage(
        index=index,
        hash=hashlib.sha256(data).hexdigest()[:16],
        full=_shrink(image, full_max_px),
        work=_shrink(image, work_max_px),
        gps=gps,
    )
