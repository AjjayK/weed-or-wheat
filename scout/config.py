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
    # Reasoning effort for every Muse Spark call: minimal, low, medium, high (the API's default), xhigh or max. Every
    # call runs at max: the agent loop, label extraction and the field explanation here, the naming panel below.
    agent_effort: str = "max"
    # Naming a weed: up to this many focused calls vote on each kind (see scout/agent/identify.py), and the panel
    # stops waiting once the votes still out can't change the winner. At max, three panels of 5 on burcucumber
    # close-ups all named it (13 votes of 15); 7 votes steady the split ones, such as velvetleaf against common mallow.
    id_samples: int = 7
    id_effort: str = "max"
    # A panel call at max took 22-138 s (median 40-105 s by photo); a call slower than this loses its vote.
    id_timeout_s: float = 150.0
    sam_model: str = "sam-3.1"
    full_max_px: int = 3072  # close-ups for naming a weed come from this size: leaf margins and hairs need pixels
    work_max_px: int = 1024
    max_tool_calls: int = 30  # generous: marking every weed takes more turns than sampling them did
    # At max effort a turn with pictures takes 30-160 s and a stop 6-10 minutes; the last turn may run past this.
    stop_timeout_s: float = 600.0
    # With this little time left, the agent is told to identify, classify, measure and finish in one turn. That turn
    # at max takes up to 160 s and the panel at max up to 150 s, so a smaller margin closed stops with weeds unnamed.
    wrap_up_s: float = 240.0
    max_concurrent_stops: int = 10  # every photo of a walk at once: the walk takes as long as its slowest stop
    max_photos: int = 10
    max_upload_bytes: int = 25_000_000
    walks_per_ip_per_hour: int = 10  # room for judges who share one office IP
    # Site-wide; X-Forwarded-For can be faked, so this is the real guard. Cloud Run's disk is the instance's
    # memory and a 10-photo walk saves ~30 MB, so this also keeps the one instance from filling its RAM.
    walks_per_hour: int = 30
    offset_gps: bool = True  # public site: shift every walk's pins by one random offset


def get_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY", ""),
        agent_model=os.environ.get("AGENT_MODEL", "muse-spark-1.3"),
        agent_effort=os.environ.get("AGENT_EFFORT", Settings.agent_effort),
        stop_timeout_s=float(os.environ.get("STOP_TIMEOUT_S", Settings.stop_timeout_s)),
        id_samples=int(os.environ.get("ID_SAMPLES", Settings.id_samples)),
        id_effort=os.environ.get("ID_EFFORT", Settings.id_effort),
        id_timeout_s=float(os.environ.get("ID_TIMEOUT_S", Settings.id_timeout_s)),
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
        walks_per_ip_per_hour=int(os.environ.get("WALKS_PER_IP_PER_HOUR", Settings.walks_per_ip_per_hour)),
        walks_per_hour=int(os.environ.get("WALKS_PER_HOUR", Settings.walks_per_hour)),
    )
