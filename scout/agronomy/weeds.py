"""Weed species: names for identification, thresholds for decisions, and where each threshold comes from.

The catalog in weeds.yaml holds the species with sourced thresholds. It is not a closed list: a weed the agent or a
label names that the catalog lacks is registered under its own name (and scientific name), and the rules grade it
with the tool default threshold.
"""
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("weeds.yaml")
TYPES = {"broadleaf", "grass", "sedge", "other"}
MAX_NAME_WORDS = 5
# Words that describe a weed without naming it: "small broadleaf seedling" or "unknown grass" is an unidentified weed.
GENERIC_WORDS = {
    "a", "an", "and", "annual", "broadleaf", "broadleafs", "broadleaves", "cluster", "dicot", "flowering", "grass",
    "grasses", "grassy", "group", "large", "leafy", "likely", "low", "mat", "monocot", "of", "or", "other", "patch",
    "perennial", "plant", "plants", "possible", "possibly", "probable", "probably", "sedge", "sedges", "seedling",
    "seedlings", "small", "sp", "species", "spp", "tall", "tiny", "unidentified", "unknown", "vine", "vines", "weed",
    "weeds", "weedy", "young"}
CROP_NAMES = {"corn", "maize", "soybean", "soybeans", "soy", "alfalfa"}
_NAME_CHARS = re.compile(r"^[A-Za-z][A-Za-z .'’-]*$")
_PARENTHESES = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


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
    catalog: bool = True  # False for a weed registered at run time (named by the agent or a label): no thresholds


def _norm(name: str) -> str:
    """Lowercase letters and single spaces; apostrophes vanish so lamb's-quarters == lambs quarters."""
    return re.sub(r"[^a-z]+", " ", re.sub(r"['’]", "", str(name).lower())).strip()


def _key(name: str) -> str:
    return _norm(name).replace(" ", "")


def is_generic(name: str | None) -> bool:
    """True for a phrase that describes a weed without naming one, such as "small broadleaf seedling"."""
    words = _norm(name if isinstance(name, str) else "").split()
    return all(word in GENERIC_WORDS for word in words)


def _display(name: str) -> str:
    """Whitespace collapsed; Title Case (as models often write names) becomes lower case, other capitals stay."""
    name = " ".join(str(name).split())
    return name.lower() if name == name.title() else name


class WeedBook:
    def __init__(self, weeds: dict[str, WeedInfo]):
        self._weeds = dict(weeds)
        self._owners: dict[str, set[str]] = {}  # name with spaces removed -> species ids
        self._lock = threading.RLock()  # one walk's stops register weeds from several threads
        for weed in self._weeds.values():
            self._index(weed)

    def _index(self, weed: WeedInfo) -> None:
        for name in (weed.id.replace("_", " "), weed.common_name, weed.scientific_name, *weed.synonyms):
            key = _key(name)
            if key:
                self._owners.setdefault(key, set()).add(weed.id)

    def copy(self) -> "WeedBook":
        """A book of its own, so weeds registered during one walk don't leak into the next."""
        with self._lock:
            return WeedBook(self._weeds)

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
        with self._lock:
            return list(self._weeds)

    def catalog_ids(self) -> list[str]:
        """The species from weeds.yaml, without the weeds registered at run time."""
        with self._lock:
            return [sid for sid, weed in self._weeds.items() if weed.catalog]

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

    def exact(self, name: str | None) -> str | None:
        """Species id whose whole name, synonym or scientific name is `name` (ignoring case, spaces and plurals);
        None when unknown or ambiguous."""
        key = _key(name if isinstance(name, str) else "")
        ids = self._lookup(key) if key else set()
        return next(iter(ids)) if len(ids) == 1 else None

    def identify(self, name: str | None, scientific_name: str | None = None, weed_type: str | None = None) -> str:
        """The species id for a weed the agent named, registering the weed when this book doesn't know it.

        A known id, name, synonym or scientific name maps to its species; an exact scientific name wins over a
        common name, being more precise. A phrase that only describes ("small broadleaf seedling"), or no name,
        maps to unidentified grass or broadleaf by weed_type. Any other name is a weed this book didn't know: it is
        registered without thresholds. Raises ValueError for a name that can't be a weed's.
        """
        name = name.strip() if isinstance(name, str) else ""
        scientific = scientific_name.strip() if isinstance(scientific_name, str) else ""
        if name in self:
            return name
        inside = _PARENTHESES.match(name)  # "burcucumber (Sicyos angulatus)"
        if inside:
            name, scientific = inside.group(1).strip(), scientific or inside.group(2).strip()
        found = self.exact(scientific) or self.exact(name) or self.resolve(name)
        if found:
            return found
        if is_generic(name):
            return "unidentified_grass" if weed_type in ("grass", "sedge") else "unidentified_broadleaf"
        return self.register(name, scientific, weed_type or "other")

    def register(self, common_name: str, scientific_name: str | None = "", weed_type: str = "other") -> str:
        """Add a weed this book doesn't know, or return the id of the one that already has this exact name."""
        common = _display(common_name)
        scientific = " ".join(str(scientific_name or "").split())
        if not _NAME_CHARS.match(common) or len(_norm(common).split()) > MAX_NAME_WORDS:
            raise ValueError(f"{common_name!r} is not a weed name: give a common name of 1 to {MAX_NAME_WORDS} words")
        if _norm(common) in CROP_NAMES:
            raise ValueError(f"{common_name!r} is a crop: use role crop, or name a volunteer crop as such")
        if not _NAME_CHARS.match(scientific):
            scientific = ""
        with self._lock:
            found = self.exact(scientific) or self.exact(common)
            if found:
                return found
            base = _norm(common).replace(" ", "_")
            sid, n = base, 2
            while sid in self._weeds:
                sid, n = f"{base}_{n}", n + 1
            weed = WeedInfo(id=sid, common_name=common, scientific_name=scientific,
                            type=weed_type if weed_type in TYPES else "other", catalog=False)
            self._weeds[sid] = weed
            self._index(weed)
            return sid

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
