"""Weed species: names for identification, thresholds for decisions, and where each threshold comes from."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("weeds.yaml")
TYPES = {"broadleaf", "grass", "sedge", "other"}
MAX_NAME_WORDS = 5


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
    """Lowercase letters and single spaces; apostrophes vanish so lamb's-quarters == lambs quarters."""
    return re.sub(r"[^a-z]+", " ", re.sub(r"['’]", "", str(name).lower())).strip()


class WeedBook:
    def __init__(self, weeds: dict[str, WeedInfo]):
        self._weeds = weeds
        self._owners: dict[str, set[str]] = {}  # name with spaces removed -> species ids
        for weed in weeds.values():
            for name in (weed.id.replace("_", " "), weed.common_name, weed.scientific_name, *weed.synonyms):
                key = _norm(name).replace(" ", "")
                if key:
                    self._owners.setdefault(key, set()).add(weed.id)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "WeedBook":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping of species id to entry")
        weeds = {}
        for key, value in raw.items():
            try:
                value = {k: ([] if v is None and k in ("synonyms", "sources") else v) for k, v in value.items()}
                info = WeedInfo(id=key, **value)
                if not (isinstance(info.synonyms, list) and isinstance(info.sources, list)):
                    raise TypeError("synonyms and sources must be lists")
                weeds[key] = info
            except (TypeError, AttributeError) as error:
                raise ValueError(f"{path}: entry '{key}': {error}") from error
        return cls(weeds)

    def ids(self) -> list[str]:
        return list(self._weeds)

    def get(self, species_id: str) -> WeedInfo:
        return self._weeds[species_id]

    def __contains__(self, species_id: object) -> bool:
        return isinstance(species_id, str) and species_id in self._weeds

    def _lookup(self, key: str) -> set[str]:
        for form in (key, key[:-1] if key.endswith("s") else "", key[:-3] + "y" if key.endswith("ies") else ""):
            if form in self._owners:
                return self._owners[form]
        return set()

    def resolve(self, name: str) -> str | None:
        """Species id for a model-written name, or None when unknown or ambiguous.

        Ignores case, punctuation and plurals, and looks for the longest known names inside the phrase, so
        "dandelion flower" and "waterhemp pigweed" resolve; "ragweed" or "giant or common ragweed" do not.
        """
        words = _norm(name if isinstance(name, str) else "").split()
        spans: list[tuple[int, int, set[str]]] = []
        for size in range(min(len(words), MAX_NAME_WORDS), 0, -1):
            for start in range(len(words) - size + 1):
                if any(a <= start and start + size <= b for a, b, _ in spans):
                    continue  # inside a longer match
                ids = self._lookup("".join(words[start:start + size]))
                if ids:
                    spans.append((start, start + size, ids))
        found = set().union(*(ids for _, _, ids in spans))
        return next(iter(found)) if len(found) == 1 else None

    def validate(self) -> list[str]:
        """Thresholds must be numbers and name their source; names must point at one species."""
        problems = []
        for weed in self._weeds.values():
            if weed.type not in TYPES:
                problems.append(f"{weed.id}: type {weed.type!r} not in {sorted(TYPES)}")
            for label, value in (("action_density", weed.action_density), ("action_cover_pct", weed.action_cover_pct)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
                    problems.append(f"{weed.id}: {label} must be a positive number, got {value!r}")
            claims = weed.zero_tolerance or weed.action_density is not None or weed.action_cover_pct is not None
            if claims and not any(str(s).strip() for s in weed.sources):
                problems.append(f"{weed.id}: threshold without a source")
        for key, ids in self._owners.items():
            if len(ids) > 1:
                problems.append(f"name {key!r} belongs to several species: {sorted(ids)}")
        return problems
