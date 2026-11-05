"""Name a kind of weed from close-ups: a panel of focused Muse Spark calls, each describing the plant before naming
it, votes on the species.

The stop agent works with many pictures and tools in its context; asked there, it named a burcucumber vine
morningglory or velvetleaf in 3 runs of 3. The same close-ups in a call of their own, which must list the plant's
features and weigh the look-alikes before naming it, gave burcucumber in 8 calls of 10. A majority over several such
calls is steadier than any one of them. Any species can win: nothing here is limited to the catalog. The panel
stops waiting once the votes still out could not change the winner, so a clear verdict costs the time of its
fastest calls, not its slowest.
"""
import json
import logging
import math
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from PIL import Image

from scout.agent.llm import user_message
from scout.agronomy.weeds import WeedBook, _norm, is_generic
from scout.app.ingest import to_data_url

log = logging.getLogger(__name__)

ID_MAX_PX = 1024  # each close-up's long side
ID_MIN_PX = 768  # smaller close-ups are enlarged to this
MIN_NAMED_SHARE = 0.4  # a species named by this share of the panel beats any number of "can't tell" answers
WEED_TYPES = ("broadleaf", "grass", "sedge", "other")
CONFIDENCE_SCORE = {"high": 3, "medium": 2, "low": 1}

ID_PROMPT = """You identify weeds for a scout in a {crop} field. The pictures show one kind of weed, cut from one \
scouting photo: for each plant, a close-up of its leaves, then a wider view of the plant and its growth habit \
(other plants, {crop} leaves or soil may show around it). First describe what you see \
that tells species apart: for a broadleaf, the leaf shape, lobes, margin (smooth or toothed), veins, hairs, leaf \
arrangement, stem, tendrils and growth habit; for a grass, the ligule, collar and sheath hairs, leaf width, and \
whether the leaves are rolled or folded. Then list the look-alikes that could fit and the feature that rules each in \
or out. Then name the species from what you see, not from what is common in {crop}: any weed species may be the \
answer. common_name is null when the pictures don't show enough to name one species."""

ID_FORMAT = {
    "type": "json_schema",
    "name": "weed_identification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "features": {"type": "array", "items": {"type": "string"}},
            "look_alikes": {"type": "array", "items": {
                "type": "object",
                "properties": {"common_name": {"type": "string"}, "verdict": {"type": "string"}},
                "required": ["common_name", "verdict"], "additionalProperties": False}},
            "common_name": {"type": ["string", "null"]},
            "scientific_name": {"type": ["string", "null"]},
            "weed_type": {"type": "string", "enum": list(WEED_TYPES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_SCORE)},
        },
        "required": ["features", "look_alikes", "common_name", "scientific_name", "weed_type", "confidence"],
        "additionalProperties": False,
    },
}


@dataclass
class Vote:
    common_name: str | None
    scientific_name: str | None
    weed_type: str
    confidence: str
    features: list[str] = field(default_factory=list)


@dataclass
class Identification:
    """The panel's verdict on one kind of weed. species_id is None when nobody named a species."""
    species_id: str | None
    common_name: str | None
    scientific_name: str | None
    weed_type: str
    votes: int  # calls that named the winner
    answered: int  # calls that answered at all
    confidence: str
    others: list[str]  # the other names given, with their votes, e.g. "velvetleaf (1)"
    features: list[str]


def _vote_keys(vote: Vote, weeds: WeedBook) -> set[str]:
    """What a vote names: a known species by its id; an unknown one by its scientific name and its name without
    spaces; a vote that names nothing as unidentified grass or broadleaf."""
    if _unnamed(vote):
        return {"unidentified_grass" if vote.weed_type in ("grass", "sedge") else "unidentified_broadleaf"}
    known = weeds.exact(vote.scientific_name) or weeds.exact(vote.common_name) or weeds.resolve(vote.common_name)
    if known:
        return {known}
    keys = {"name:" + _norm(vote.common_name).replace(" ", "")}
    genus_species = _norm(vote.scientific_name or "").split()[:2]
    if len(genus_species) == 2:
        keys.add("sci:" + " ".join(genus_species))
    return keys


def _unnamed(vote: Vote) -> bool:
    return not vote.common_name or is_generic(vote.common_name)


def _group(votes: list[Vote], weeds: WeedBook) -> list[list[Vote]]:
    """Votes for one weed, however written: those sharing a name or a scientific name count together."""
    groups: list[tuple[set[str], list[Vote]]] = []
    for vote in votes:
        keys, members = _vote_keys(vote, weeds), [vote]
        for group in [g for g in groups if g[0] & keys]:
            groups.remove(group)
            keys, members = keys | group[0], group[1] + members
        groups.append((keys, members))
    return [members for _, members in groups]


def ask_once(llm, crop: str, urls: list[str], effort: str | None) -> Vote:
    prompt = ID_PROMPT.format(crop=crop)
    turn = llm.turn([user_message(prompt, urls)], text_format=ID_FORMAT, effort=effort)
    data = json.loads(turn.text)
    weed_type = data.get("weed_type") if data.get("weed_type") in WEED_TYPES else "other"
    confidence = data.get("confidence") if data.get("confidence") in CONFIDENCE_SCORE else "low"
    name = data.get("common_name")
    return Vote(name.strip() if isinstance(name, str) and name.strip() else None,
                data.get("scientific_name") or None, weed_type, confidence,
                [str(f) for f in data.get("features") or []])


def _ranked(votes: list[Vote], weeds: WeedBook) -> list[list[Vote]]:
    """The groups of votes, winner first: the most votes, then the surer voters, then the earlier vote. "Can't
    tell" is no vote for another weed: the most-named species wins over abstentions when enough of the panel named
    it."""
    order = {id(vote): n for n, vote in enumerate(votes)}
    ranked = sorted(_group(votes, weeds), key=lambda g: (-len(g), -sum(CONFIDENCE_SCORE[v.confidence] for v in g),
                                                        min(order[id(v)] for v in g)))
    named = [group for group in ranked if not _unnamed(group[0])]
    if named and len(named[0]) >= max(1, math.ceil(MIN_NAMED_SHARE * len(votes))):
        ranked = [named[0]] + [group for group in ranked if group is not named[0]]
    return ranked


def settled(votes: list[Vote], remaining: int, weeds: WeedBook) -> bool:
    """True when no `remaining` votes still to come could change the winner, however they fall: a rival given all
    of them, or a species no one has named yet, would still be behind."""
    if remaining <= 0:
        return True
    if not votes:
        return False
    winner, *rivals = _ranked(votes, weeds)
    lead = len(winner)
    species = max((len(group) for group in rivals if not _unnamed(group[0])), default=0)
    bar = MIN_NAMED_SHARE * (len(votes) + remaining)  # the share that beats abstentions, if every call answers
    if not _unnamed(winner[0]) and lead >= bar:
        return species + remaining < lead  # over the bar for good: only another species could pass it
    # otherwise it must stay first by count, and an abstaining winner must keep every species under the bar
    if max((len(group) for group in rivals), default=0) + remaining >= lead:
        return False
    return not _unnamed(winner[0]) or species + remaining < bar


def tally(votes: list[Vote], weeds: WeedBook) -> Identification:
    """The most-voted species wins; a tie goes to the one its voters were surer of. The winner is registered in the
    book when it is a weed the book didn't know. The winner's share of all answers sets the confidence."""
    if not votes:
        raise ValueError("no identification answered")
    ranked = _ranked(votes, weeds)
    winners = ranked[0]
    key = next(iter(_vote_keys(winners[0], weeds)))
    # the name most of the winning voters wrote, and a scientific name if any gave one
    name = Counter(v.common_name for v in winners if v.common_name).most_common(1)
    scientific = Counter(v.scientific_name for v in winners if v.scientific_name).most_common(1)
    weed_type = Counter(v.weed_type for v in winners).most_common(1)[0][0]
    common = name[0][0] if name else None
    if key.startswith("unidentified"):
        species_id, common, scientific_name = key, None, None
    else:
        scientific_name = scientific[0][0] if scientific else None
        try:
            species_id = weeds.identify(common, scientific_name, weed_type)
        except ValueError:  # a name that can't be a weed's: the vote stands, unnamed
            species_id = None
    share = len(winners) / len(votes)
    sure = sum(CONFIDENCE_SCORE[v.confidence] for v in winners) / len(winners)
    confidence = ("high" if share >= 0.8 and sure >= 2.5 else "medium" if share > 0.5 else "low")
    others = [f"{(group[0].common_name or 'unidentified')} ({len(group)})" for group in ranked[1:]]
    return Identification(species_id, common if species_id else None, scientific_name if species_id else None,
                          weed_type, len(winners), len(votes), confidence, others, winners[0].features[:6])


def fit(picture: Image.Image) -> Image.Image:
    """A close-up sized for the panel: a big one shrunk to ID_MAX_PX, a small one enlarged to ID_MIN_PX, so a
    seedling cut from a wide photo isn't seen as a few dozen pixels."""
    copy = picture.convert("RGB")
    long_side = max(copy.size)
    if long_side > ID_MAX_PX:
        copy.thumbnail((ID_MAX_PX, ID_MAX_PX))
    elif long_side < ID_MIN_PX:
        scale = ID_MIN_PX / long_side
        copy = copy.resize((round(copy.width * scale), round(copy.height * scale)), Image.Resampling.LANCZOS)
    return copy


def identify_kinds(llm, crop: str, kinds: list[list[Image.Image]], weeds: WeedBook, samples: int,
                   effort: str | None, workers: int | None = None,
                   timeout_s: float | None = None) -> list[Identification | None]:
    """One verdict per kind of weed (None when no call answered); every call of every kind runs at once. The panel
    stops waiting once every kind is settled: the votes still out could not change any winner. With timeout_s,
    the votes in by then are counted and later ones are dropped: one slow call mustn't stall the stop."""
    urls = [[to_data_url(fit(picture)) for picture in pictures] for pictures in kinds]
    jobs = [(k, n) for k in range(len(kinds)) for n in range(samples)]

    def one(job):
        try:
            return job[0], ask_once(llm, crop, urls[job[0]], effort)
        except Exception:  # one failed call costs one vote, not the identification
            log.warning("an identification call failed", exc_info=True)
            return job[0], None

    pool = ThreadPoolExecutor(max_workers=max(1, min(workers or len(jobs), len(jobs))))
    pending = {pool.submit(one, job) for job in jobs}
    votes: list[list[Vote]] = [[] for _ in kinds]
    left = [samples] * len(kinds)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while pending and not all(settled(votes[k], left[k], weeds) for k in range(len(kinds))):
        wait_s = None if deadline is None else max(0.0, deadline - time.monotonic())
        done, pending = wait(pending, timeout=wait_s, return_when=FIRST_COMPLETED)
        if not done:
            log.warning("%d of %d identification calls were too slow and were not counted", len(pending), len(jobs))
            break
        for future in done:
            kind, vote = future.result()
            left[kind] -= 1
            if vote is not None:
                votes[kind].append(vote)
    pool.shutdown(wait=False, cancel_futures=True)  # calls still out finish in the background, unheard
    return [tally(kind_votes, weeds) if kind_votes else None for kind_votes in votes]
