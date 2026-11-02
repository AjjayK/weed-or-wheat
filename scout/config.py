"""Runtime settings, read from the environment (and .env when present)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.meta.ai/v1"
    agent_model: str = "muse-spark-1.3"
    sam_model: str = "sam-3.1"
    full_max_px: int = 2048
    work_max_px: int = 1024
    max_tool_calls: int = 30  # generous: marking every weed takes more turns than sampling them did
    stop_timeout_s: float = 360.0  # Muse Spark takes 13-16 s per turn; a thorough stop needs 2-4 minutes
    max_concurrent_stops: int = 10  # every photo of a walk at once: the walk takes as long as its slowest stop
    max_photos: int = 10
    max_upload_bytes: int = 25_000_000
    walks_per_ip_per_hour: int = 3
    walks_per_hour: int = 20  # site-wide; X-Forwarded-For can be faked, so this is the real cost guard
    offset_gps: bool = True  # public site: shift every walk's pins by one random offset


def get_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY", ""),
        agent_model=os.environ.get("AGENT_MODEL", "muse-spark-1.3"),
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
    )
