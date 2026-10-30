"""Weed species: names for identification, thresholds for decisions, and where each threshold comes from."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("weeds.yaml")


@dataclass
class WeedInfo:
    id: str
    common_name: str
    scientific_name: str
    type: str
    synonyms: list[str] = field(default_factory=list)
    action_density: float | None = None  # plants per m², sourced
    action_cover_pct: float | None = None  # % ground cover, sourced
    zero_tolerance: bool = False
    sources: list[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]+", " ", name.lower()).strip()


class WeedBook:
    def __init__(self, weeds: dict[str, WeedInfo]):
        self._weeds = weeds
        self._names: dict[str, str] = {}
        for weed in weeds.values():
            for name in (weed.id.replace("_", " "), weed.common_name, weed.scientific_name, *weed.synonyms):
                key = _norm(name)
                if key:
                    self._names.setdefault(key, weed.id)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "WeedBook":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls({key: WeedInfo(id=key, **value) for key, value in raw.items()})

    def ids(self) -> list[str]:
        return list(self._weeds)

    def get(self, species_id: str) -> WeedInfo:
        return self._weeds[species_id]

    def __contains__(self, species_id: object) -> bool:
        return species_id in self._weeds

    def resolve(self, name: str) -> str | None:
        key = _norm(name or "")
        return self._names.get(key) if key else None

    def validate(self) -> list[str]:
        """Every threshold or zero-tolerance flag must name its source."""
        problems = []
        for weed in self._weeds.values():
            claims = weed.zero_tolerance or weed.action_density is not None or weed.action_cover_pct is not None
            if claims and not weed.sources:
                problems.append(f"{weed.id}: threshold without a source")
        return problems
