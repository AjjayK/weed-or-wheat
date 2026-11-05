"""The seven tools Muse Spark can call at a stop, and the code that runs them. Every box in or out is 0-1000."""
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from PIL import Image

from scout.agent.identify import identify_kinds
from scout.agent.state import (MAX_BOXES, MAX_MARK_BOXES, UNMARKED_SHARE, StopFailed, StopState, to_pixels,
                               to_thousandths)
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import to_data_url
from scout.models import Instance
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.measure import compute_metrics
from scout.vision.sam_parse import SamRecord

log = logging.getLogger(__name__)

VIEW_TYPES = ("row", "top_down", "close_up", "wide")
KINDS = ("crop", "weed")
MARK_KINDS = ("crop", "weed", "patch")
CROPS_SEEN = ("corn", "soybean", "alfalfa", "other", "unsure")
CONFIDENCES = ("high", "medium", "low")
WEED_TYPES = ("broadleaf", "grass", "sedge", "other")
ZOOM_MAX_PX = 1024  # leaf margins, hairs and ligules decide a species; 768 px lost some of them
MAX_LISTED = 100
MAX_ID_GROUPS = 6  # kinds of weed named in one identify call
MAX_ID_SAMPLES = 3  # plants of one kind
CONTEXT_MARGIN = 1.0  # the wider view of a plant for identify: its box plus this share of it on every side
SAM_WORKERS = 4  # SAM requests in flight per mark call
MAX_ESTIMATE = 1000  # the survey's weed estimate is capped here
OVERLAY_QUALITY = 80  # JPEG quality of the picture sent back after a mark
SAM_UNAVAILABLE_LIMIT = 3  # SAM busy or timing out this many calls in a row ends the stop
NOTHING = "nothing found"
MOSTLY_CROP_NOTE = "mostly crop"
PATCH_NOTE = "large patch; counted as one, approximate"
OVERLAY_CAPTION = ("What is marked now: green is crop, magenta is weed, gray is the in-row zone, "
                   "blue boxes are weed areas, yellow boxes need a check")


class SamUnavailable(Exception):
    """SAM did not answer (a rate limit that outlasted the retries, a server error, a timeout, a dropped connection)."""


def _number(value, name: str) -> float | None:
    """A model-written number (3, 3.5 or "3"); None stays None."""
    if value is None:
        return None
    try:
        number = math.nan if isinstance(value, bool) else float(value)
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a number or null, got {value!r}")
    return number


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _is_box_item(item) -> bool:
    """A mark item is a dict whose "box" is a list of four finite numbers (numeric strings like "600" are fine)."""
    box = item.get("box") if isinstance(item, dict) else None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        return all(not isinstance(v, bool) and math.isfinite(float(v)) for v in box)
    except (TypeError, ValueError, OverflowError):
        return False


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}


_BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
        "description": "[x1, y1, x2, y2] from 0 to 1000: x from the left edge, y from the top edge"}

TOOL_SCHEMAS = [
    _fn("survey", "Look over the photo first: the crop you see, and how many weed plants are visible.",
        {"crop_seen": {"type": "string", "enum": list(CROPS_SEEN)},
         "weeds_visible": {"type": "integer", "minimum": 0, "maximum": MAX_ESTIMATE,
                           "description": "Your estimate of separate weed plants visible, small ones included"}},
        ["crop_seen", "weeds_visible"]),
    _fn("segment", "Ask SAM 3.1 for every instance of one concept, a short noun phrase. It finds \"crop row\" "
        "(kind crop) and distinct things such as \"dandelion\" or \"yellow flower\" (kind weed). It does not find crop "
        "plants, leaves, or weeds in general: use mark for those.",
        {"phrase": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}},
        ["phrase", "kind"]),
    _fn("mark", f"Mark what you see with boxes. crop and weed boxes go to SAM 3.1, up to {MAX_MARK_BOXES} per call; "
        f"call again for more (code splits them into requests of {MAX_BOXES}). patch boxes are areas of weeds too "
        "dense or small to box one by one; code measures them as green pixels that are not crop. The answer comes "
        "with a picture of everything marked so far.",
        {"kind": {"type": "string", "enum": list(MARK_KINDS),
                  "description": "crop: boxes around crop rows or crop plants (one tall box per visible row is "
                                 "fine). weed: a tight box around each weed, one per plant. patch: a box around an "
                                 "area of dense or tiny weeds, with as little crop as possible."},
         "boxes": {"type": "array", "minItems": 1, "maxItems": MAX_MARK_BOXES, "items": {
             "type": "object",
             "properties": {"box": _BOX, "label": {"type": "string", "description": "What it is, in 2-3 words"}},
             "required": ["box", "label"], "additionalProperties": False}}},
        ["kind", "boxes"]),
    _fn("zoom", "Look closely at one instance or region. The close-up arrives in the next message.",
        {"instance_id": {"type": "string"}, "box": _BOX}, []),
    _fn("identify", "Name the weeds: one group per kind of weed you marked, listing every instance of that kind. "
        f"Close-ups of its {MAX_ID_SAMPLES} largest go to a panel of focused identifications that vote; the answer "
        "gives the species, the votes, the other names given, and the features seen. A listed instance you then "
        "classify with species_id null gets the panel's species.",
        {"groups": {"type": "array", "minItems": 1, "maxItems": MAX_ID_GROUPS, "items": {
            "type": "object",
            "properties": {"instance_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                           "label": {"type": "string", "description": "This kind as you see it, in 2-4 words"}},
            "required": ["instance_ids", "label"], "additionalProperties": False}}},
        ["groups"]),
    _fn("classify", "Record your verdict on one or more instances. Batch them.",
        {"items": {"type": "array", "items": {"type": "object", "properties": {
            "instance_id": {"type": "string"},
            "role": {"type": "string", "enum": ["crop", "weed", "ignore"]},
            "species_id": {"type": ["string", "null"],
                           "description": "An id from the species list when the weed is one of them; for any other "
                                          "weed, its common name (e.g. \"wild buckwheat\"); null when you cannot "
                                          "tell"},
            "scientific_name": {"type": ["string", "null"],
                                "description": "The weed's binomial, e.g. \"Polygonum convolvulus\"; null if unsure"},
            "weed_type": {"type": ["string", "null"], "enum": [*WEED_TYPES, None]},
            "height_in": {"type": ["number", "null"]},
            "leaf_count": {"type": ["integer", "null"]},
            "id_confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["instance_id", "role", "id_confidence"], "additionalProperties": False}}},
        ["items"]),
    _fn("measure", "Compute weed and crop measurements for this photo with code.",
        {"view_type": {"type": "string", "enum": list(VIEW_TYPES)}}, ["view_type"]),
    _fn("finish", "Close this stop after measure.",
        {"crop_stage": {"type": ["string", "null"], "description": "Crop growth stage such as V3, or null"},
         "notes": {"type": "string"},
         "all_weeds_marked": {"type": "boolean",
                              "description": "true only if every weed you can see is marked, as boxes or areas"}},
        ["crop_stage", "notes", "all_weeds_marked"]),
]


@dataclass
class ToolResult:
    output: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (caption, data URL)
    finished: bool = False


class ToolRunner:
    def __init__(self, state: StopState, sam, weeds: WeedBook, max_calls: int = 12,
                 time_limit_s: float | None = None, llm=None, id_samples: int = 5, id_effort: str | None = None,
                 id_timeout_s: float | None = None):
        self.state = state
        self.sam = sam
        self.weeds = weeds
        self.max_calls = max_calls
        self.time_limit_s = time_limit_s
        self.llm = llm  # for identify's panel; None leaves naming to the agent alone
        self.id_samples = id_samples
        self.id_effort = id_effort
        self.id_timeout_s = id_timeout_s

    def run(self, name: str, args: dict) -> ToolResult:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None or "__invalid__" in args:
            self.state.log(name, "Invalid tool call", ok=False)
            return ToolResult(f"error: unknown tool or malformed arguments for {name}")
        try:
            return handler(**args)
        except (TypeError, ValueError, KeyError) as error:
            self.state.log(name, f"Rejected: {error}", ok=False)
            return ToolResult(f"error: {error}")
        except BadRequest as error:
            count = self.state.bad_requests[name] = self.state.bad_requests.get(name, 0) + 1
            self.state.log(name, f"SAM rejected the request: {error}", ok=False)
            if count >= 2:
                raise StopFailed(f"SAM rejected {name} twice") from error
            return ToolResult(f"error: SAM rejected the request: {error}")
        except SamUnavailable as error:  # a busy SAM is no reason to lose the marks made so far: say so, try again
            self.state.sam_unavailable += 1
            self.state.log(name, f"SAM unavailable: {error}", ok=False)
            if self.state.sam_unavailable >= SAM_UNAVAILABLE_LIMIT:
                raise StopFailed(f"SAM was unavailable {SAM_UNAVAILABLE_LIMIT} times") from error
            return ToolResult(f"error: SAM is busy or timed out ({error}). Nothing from this call was marked: "
                              "send it again.")

    def _brief(self, inst: Instance) -> dict:
        width, height = self.state.size
        out = {"id": inst.id, "label": inst.prompt, "box": to_thousandths(inst.box, self.state.size),
               "area_pct": round(100 * inst.area_px / (width * height), 2)}
        if inst.sam_confidence is not None:
            out["conf"] = round(inst.sam_confidence, 2)
        if inst.check_this:
            out["check_this"] = True
        return out

    def _add(self, record: SamRecord, kind: str, source: str, label: str) -> tuple[Instance | None, str | None]:
        """One SAM record -> a crop or weed instance, plus a note for the agent when it was dropped or is a patch."""
        mask = record.full_mask(self.state.size)
        if not mask.any():
            return None, NOTHING
        if kind == "crop":
            return self.state.add_mask(mask, source, label, role="crop", confidence=record.confidence), None
        inst = self.state.add_weed(mask, source, label, confidence=record.confidence)
        if inst is None:
            return None, MOSTLY_CROP_NOTE
        return inst, PATCH_NOTE if self.state.is_patch(inst) else None

    def _sam_boxes(self, pixel_boxes: list) -> dict[str, SamRecord]:
        """Every box to SAM in requests of at most 16, sent in parallel. Keys are box numbers from 1."""
        starts = list(range(0, len(pixel_boxes), MAX_BOXES))
        chunks = [pixel_boxes[start:start + MAX_BOXES] for start in starts]

        def ask(start: int, chunk: list) -> list[SamRecord]:
            try:
                return self.sam.segment_boxes(self.state.image, chunk)
            except BadRequest as error:  # name the boxes at fault; the whole call fails, so all boxes must be resent
                raise BadRequest(f"boxes {start + 1}-{start + len(chunk)}: {error}. Nothing from this call was "
                                 "marked: send all the boxes again, changing those.") from error
            except Exception as error:  # a 429 that outlasted the retries, a 5xx, a timeout, a dropped connection
                log.warning("SAM did not answer boxes %d-%d", start + 1, start + len(chunk), exc_info=True)
                raise SamUnavailable(f"boxes {start + 1}-{start + len(chunk)}: {type(error).__name__}") from error

        with ThreadPoolExecutor(max_workers=min(SAM_WORKERS, len(chunks))) as pool:
            answers = list(pool.map(ask, starts, chunks))
        self.state.sam_unavailable = 0  # every request answered: only failures in a row count
        records: dict[str, SamRecord] = {}
        unexpected = []
        for start, chunk, answer in zip(starts, chunks, answers):
            for record in answer:  # object i + 1 is the chunk's box i
                n = int(record.object_id) if str(record.object_id).isdecimal() else 0
                if 1 <= n <= len(chunk):
                    records.setdefault(str(start + n), record)
                else:
                    unexpected.append(record.object_id)
        if unexpected:
            self.state.log("mark", f"SAM returned unexpected object ids {unexpected[:5]}", ok=False)
        return records

    def _render_overlay(self):
        """The marking picture, rendered once per mark: for the agent's feedback and for people watching."""
        try:
            return self.state.overlay()
        except Exception:  # the picture helps; the counts in the answer still stand without it
            log.warning("overlay picture failed", exc_info=True)
            return None

    def _feedback(self, out: dict, kind: str, picture) -> ToolResult:
        """Counts against the survey estimate, a hint when weed marks fall short of it, what is left of the budget, and
        a picture of the marks."""
        plants, areas = self.state.weeds_marked()
        out["weeds_marked"] = {"plants": plants, "weed_areas": areas}
        out["calls_left"] = max(self.max_calls - self.state.calls, 0)
        if self.time_limit_s is not None:
            out["seconds_left"] = max(round(self.time_limit_s - self.state.elapsed_s()), 0)
        estimate = self.state.weeds_estimate
        if estimate is not None:
            out["weeds_estimate"] = estimate
            counted = self.state.plants_marked()
            # like the final check, no hint once a weed area exists: its pieces are not plant counts
            if kind != "crop" and not self.state.has_weed_area() and counted < UNMARKED_SHARE * estimate:
                out["hint"] = (f"You estimated {estimate} weeds; your marks count as about {counted}: "
                               f"{_plural(plants, 'boxed plant')}. "
                               "Mark the rest: more boxes, or kind patch where they are too dense.")
        if picture is None:
            out["picture"] = "unavailable"
            return ToolResult(json.dumps(out))
        return ToolResult(json.dumps(out), images=[(OVERLAY_CAPTION, to_data_url(picture, quality=OVERLAY_QUALITY))])

    def _mark_patches(self, pixel_boxes: list, labels: list[str]) -> ToolResult:
        found, notes = [], []
        for n, (box, label) in enumerate(zip(pixel_boxes, labels), start=1):
            inst, note = self.state.add_patch(box, label)
            if inst is not None:
                found.append({"box_no": n, **self._brief(inst), "plants": inst.plant_count})
            if note:
                notes.append(f"box {n}: {note}")
        picture = self._render_overlay()
        self.state.log("mark", f"Mark patch: {len(found)} of {len(pixel_boxes)} areas kept", [f["id"] for f in found],
                       picture=picture)
        return self._feedback({"found": found, "notes": notes}, "patch", picture)

    def _tool_survey(self, crop_seen: str, weeds_visible) -> ToolResult:
        if crop_seen not in CROPS_SEEN:
            raise ValueError(f"crop_seen must be one of {list(CROPS_SEEN)}")
        count = _number(weeds_visible, "weeds_visible")
        if count is None:
            raise ValueError("weeds_visible must be a number")
        estimate = min(round(count), MAX_ESTIMATE)
        self.state.crop_seen, self.state.weeds_estimate = crop_seen, estimate
        look = "crop unclear" if crop_seen == "unsure" else f"looks like {crop_seen}"
        self.state.log("survey", f"Survey: {look}, about {estimate} weeds")
        return ToolResult(json.dumps({"ok": True}))

    def _tool_segment(self, phrase: str, kind: str) -> ToolResult:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        phrase = str(phrase).strip()
        if not phrase:
            raise ValueError("phrase is empty")
        found, notes = [], []
        try:
            records = self.sam.segment_concept(self.state.image, phrase)
        except BadRequest:
            raise  # run() handles a rejected request
        except Exception as error:
            log.warning("SAM did not answer segment %r", phrase, exc_info=True)
            raise SamUnavailable(type(error).__name__) from error
        self.state.sam_unavailable = 0
        for n, record in enumerate(records, start=1):
            inst, note = self._add(record, kind, "concept", phrase)
            if inst is not None:
                found.append(inst)
            if note and note != NOTHING:
                notes.append(f"match {n}: {note}")
        dropped = sum(note.endswith(MOSTLY_CROP_NOTE) for note in notes)
        summary = f'Segment "{phrase}": {len(found)} found'
        if dropped:
            summary += f", {dropped} dropped as crop"
        if kind == "crop":
            self.state.refresh_patches()
        self.state.log("segment", summary, [i.id for i in found])
        return ToolResult(json.dumps({"total_found": len(found), "found": [self._brief(i) for i in found[:MAX_LISTED]],
                                      "notes": notes[:MAX_LISTED]}))

    def _tool_mark(self, kind: str, boxes: list[dict]) -> ToolResult:
        if kind not in MARK_KINDS:
            raise ValueError(f"kind must be one of {list(MARK_KINDS)}")
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_MARK_BOXES:
            raise ValueError(f"give 1 to {MAX_MARK_BOXES} boxes")
        for n, item in enumerate(boxes, start=1):
            if not _is_box_item(item):
                raise ValueError(f"box {n}: needs a box of four numbers")
        pixel_boxes = [to_pixels(item["box"], self.state.size) for item in boxes]
        labels = [str(item.get("label") or kind).strip() for item in boxes]
        if kind == "patch":
            return self._mark_patches(pixel_boxes, labels)
        records = self._sam_boxes(pixel_boxes)
        found, notes = [], []
        for n, label in enumerate(labels, start=1):
            record = records.get(str(n))
            inst, note = self._add(record, kind, "prompt", label) if record else (None, NOTHING)
            if inst is not None:
                found.append({"box_no": n, **self._brief(inst)})
            if note:
                notes.append(f"box {n}: {note}")
        if kind == "weed" and len(boxes) == MAX_MARK_BOXES:
            self.state.box_cap_hit = True  # more weeds may be out there: the count becomes a floor
        if kind == "crop":
            self.state.refresh_patches()
        picture = self._render_overlay()
        self.state.log("mark", f"Mark {kind}: {len(found)} of {len(boxes)} boxes kept", [f["id"] for f in found],
                       picture=picture)
        return self._feedback({"found": found, "notes": notes}, kind, picture)

    def _close_up(self, box, margin: float = 0.15) -> Image.Image:
        """The full-size photo cut around a work-image box, with a margin (a share of the box) for context."""
        left, top, right, bottom = mk.pad_box(box, margin, self.state.size)
        scale = self.state.image.full.width / self.state.image.work.width
        return self.state.image.full.crop(tuple(int(v * scale) for v in (left, top, right, bottom)))

    def _tool_identify(self, groups: list[dict]) -> ToolResult:
        if self.llm is None or self.id_samples < 1:
            raise ValueError("identify is not available here: name the weeds in classify yourself")
        if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_ID_GROUPS:
            raise ValueError(f"give 1 to {MAX_ID_GROUPS} groups, one per kind of weed")
        kinds, ids, labels = [], [], []
        for n, group in enumerate(groups, start=1):
            members = group.get("instance_ids") if isinstance(group, dict) else None
            if not isinstance(members, list) or not members:
                raise ValueError(f"group {n}: list the instance_ids of this kind")
            for iid in members:
                if iid not in self.state.instances:
                    raise ValueError(f"group {n}: unknown instance {iid}")
            # The largest plants show the most of their features (the agent, asked to list them first, sometimes led
            # with a sliver of crop leaf), each twice: its leaves up close, then the plant in its surroundings.
            largest = sorted(members, key=lambda iid: -self.state.instances[iid].area_px)[:MAX_ID_SAMPLES]
            kinds.append([view for iid in largest for view in (
                self._close_up(self.state.instances[iid].box),
                self._close_up(self.state.instances[iid].box, CONTEXT_MARGIN))])
            ids.append(list(members))
            labels.append(str(group.get("label") or f"group {n}").strip())
        verdicts = identify_kinds(self.llm, self.state.crop, kinds, self.weeds, self.id_samples, self.id_effort,
                                  timeout_s=self.id_timeout_s)
        out, lines = [], []
        for members, label, verdict in zip(ids, labels, verdicts):
            if verdict is None:
                out.append({"instance_ids": members, "error": "no identification answered: name it yourself"})
                lines.append(f"{label}: no identification answered")
                continue
            name = (verdict.common_name or self.weeds.get(verdict.species_id).common_name) if verdict.species_id \
                else "unidentified"
            out.append({"instance_ids": members, "species_id": verdict.species_id, "common_name": verdict.common_name,
                        "scientific_name": verdict.scientific_name, "weed_type": verdict.weed_type,
                        "votes": f"{verdict.votes} of {verdict.answered}", "confidence": verdict.confidence,
                        "other_names": verdict.others, "features": verdict.features})
            for iid in members:
                self.state.identified[iid] = (verdict.species_id, verdict.confidence)
            others = f"; also {', '.join(verdict.others)}" if verdict.others else ""
            lines.append(f"{label}: {name}, {verdict.votes} of {verdict.answered} votes{others}")
        # one step per call, so people watching count it as one tool call; the first kind's close-up shows
        self.state.log("identify", " | ".join(lines), [i for members in ids for i in members],
                       ok=any(v is not None for v in verdicts), picture=kinds[0][0])
        return ToolResult(json.dumps({"identified": out}))

    def _tool_zoom(self, instance_id: str | None = None, box: list[int] | None = None) -> ToolResult:
        if instance_id:
            if instance_id not in self.state.instances:
                raise ValueError(f"unknown instance {instance_id}")
            region, label = self.state.instances[instance_id].box, instance_id
        elif box:
            region, label = to_pixels(box, self.state.size), f"box {list(box)}"
        else:
            raise ValueError("give instance_id or box")
        left, top, right, bottom = mk.pad_box(region, 0.15, self.state.size)
        if right - left < 2 or bottom - top < 2:
            raise ValueError("region is too small to zoom")
        crop = self._close_up(region)
        crop.thumbnail((ZOOM_MAX_PX, ZOOM_MAX_PX))
        self.state.log("zoom", f"Zoom on {label}", [instance_id] if instance_id else [], picture=crop)
        return ToolResult(f"The close-up of {label} follows in the next message.",
                          images=[(f"Close-up of {label}", to_data_url(crop))])

    def _tool_classify(self, items: list[dict]) -> ToolResult:
        resolved = []
        for item in items:
            inst = self.state.instances.get(item["instance_id"])
            if inst is None:
                raise ValueError(f"unknown instance {item['instance_id']}")
            role = item["role"]
            if role not in ("crop", "weed", "ignore"):
                raise ValueError(f"bad role {role}")
            species = None
            named, scientific, kind = item.get("species_id"), item.get("scientific_name"), item.get("weed_type")
            if kind is not None and kind not in WEED_TYPES:
                raise ValueError(f"weed_type must be one of {list(WEED_TYPES)} or null")
            # a crop's name is not a weed species: ignore it. Any weed name is accepted: one the book doesn't know
            # is registered under its own name, so a weed off the list is never forced onto a wrong species.
            if role == "weed" and (named or scientific or kind):
                species = self.weeds.identify(named, scientific, kind)
            panel = self.state.identified.get(inst.id)  # (species, confidence) from identify, if it saw this kind
            if role == "weed" and not named and not scientific and panel and panel[0]:
                species = panel[0]  # no name of its own: identify's verdict stands
            confidence = item["id_confidence"]
            if confidence not in CONFIDENCES:
                raise ValueError(f"id_confidence must be one of {list(CONFIDENCES)}")
            if role == "weed" and panel and species == panel[0] and CONFIDENCES.index(panel[1]) > \
                    CONFIDENCES.index(confidence):
                confidence = panel[1]  # a name from a split panel is no surer than the panel: the report flags it
            height = _number(item.get("height_in"), "height_in")
            leaves = _number(item.get("leaf_count"), "leaf_count")
            resolved.append((inst, role, species, confidence, height, None if leaves is None else int(leaves)))
        for inst, role, species, confidence, height, leaves in resolved:
            inst.role, inst.species_id, inst.id_confidence = role, species, confidence
            inst.height_in, inst.leaf_count = height, leaves
            # a confident verdict clears the flag; a big SAM mask may be crop, but a weed area is meant to be big
            inst.check_this = confidence == "low" or (inst.source != "patch" and self.state.is_patch(inst))
        weeds = [inst for inst, role, *_ in resolved if role == "weed"]
        if len(resolved) == 1 and weeds:
            inst = weeds[0]
            name = self.weeds.get(inst.species_id).common_name if inst.species_id else "unidentified weed"
            details = [f"~{inst.height_in:g} in" if inst.height_in is not None else "",
                       f"{inst.leaf_count} leaves" if inst.leaf_count is not None else ""]
            summary = "Identified: " + ", ".join([name] + [d for d in details if d])
        else:
            summary = f"Classified {len(resolved)} plants ({len(weeds)} weeds)"
        ids = [inst.id for inst, *_ in resolved]
        self.state.refresh_patches()  # an instance reclassified as crop changes what weed areas leave out
        self.state.log("classify", summary, ids)
        return ToolResult(json.dumps({"ok": ids}))

    def _tool_measure(self, view_type: str) -> ToolResult:
        if view_type not in VIEW_TYPES:
            raise ValueError(f"view_type must be one of {list(VIEW_TYPES)}")
        self.state.view_type = view_type
        if view_type in ("row", "wide"):
            self.state.ensure_sky(self.sam)
        metrics = compute_metrics(list(self.state.instances.values()), self.state.masks, self.state.size,
                                  view_type, self.state.sky, self.state.row_band())
        out = asdict(metrics)
        out["count_approximate"] = metrics.count_approximate or self.state.count_approximate(view_type)
        self.state.log("measure", f"Measure: weed cover {metrics.weed_cover_pct}%, {metrics.weed_count} weeds")
        return ToolResult(json.dumps(out))

    def _tool_finish(self, crop_stage: str | None, notes: str, all_weeds_marked=None) -> ToolResult:
        unclassified = self.state.unclassified()
        if unclassified and self.state.calls < self.max_calls:
            self.state.log("finish", "Refused: unclassified weeds remain", ok=False)
            return ToolResult(f"error: classify these first: {unclassified[:20]}")
        if self.state.view_type is None:
            self.state.log("finish", "Refused: measure has not run", ok=False)
            return ToolResult("error: call measure first")
        self.state.crop_stage = None if crop_stage is None else (str(crop_stage).strip() or None)
        self.state.notes = str(notes or "")
        self.state.all_weeds_marked = all_weeds_marked if isinstance(all_weeds_marked, bool) else None
        self.state.finished = True
        self.state.log("finish", "Stop closed")
        return ToolResult("ok", finished=True)
