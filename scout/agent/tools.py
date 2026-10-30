"""The six tools Muse Spark can call at a stop, and the code that runs them. Every box in or out is 0-1000."""
import json
import math
from dataclasses import asdict, dataclass, field

from scout.agent.state import MAX_BOXES, StopFailed, StopState, to_pixels, to_thousandths
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import to_data_url
from scout.models import Instance
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.measure import compute_metrics
from scout.vision.sam_parse import SamRecord

VIEW_TYPES = ("row", "top_down", "close_up", "wide")
KINDS = ("crop", "weed")
CONFIDENCES = ("high", "medium", "low")
ZOOM_MAX_PX = 768
MAX_LISTED = 100
NOTHING = "nothing found"
MOSTLY_CROP_NOTE = "mostly crop"
PATCH_NOTE = "large patch; counted as one, approximate"


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


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}


_BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
        "description": "[x1, y1, x2, y2] from 0 to 1000: x from the left edge, y from the top edge"}

TOOL_SCHEMAS = [
    _fn("segment", "Ask SAM 3.1 for every instance of one concept, a short noun phrase. It finds \"crop row\" "
        "(kind crop) and distinct things such as \"dandelion\" or \"yellow flower\" (kind weed). It does not find crop "
        "plants, leaves, or weeds in general: use mark for those.",
        {"phrase": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}},
        ["phrase", "kind"]),
    _fn("mark", "Segment plants you can see, one box per plant; all boxes go to SAM 3.1 in one request.",
        {"kind": {"type": "string", "enum": list(KINDS),
                  "description": "crop: boxes around crop rows or crop plants (one tall box per visible row is "
                                 "fine). weed: a tight box around each weed, one per plant."},
         "boxes": {"type": "array", "minItems": 1, "maxItems": MAX_BOXES, "items": {
             "type": "object",
             "properties": {"box": _BOX, "label": {"type": "string", "description": "What it is, in 2-3 words"}},
             "required": ["box", "label"], "additionalProperties": False}}},
        ["kind", "boxes"]),
    _fn("zoom", "Look closely at one instance or region. The close-up arrives in the next message.",
        {"instance_id": {"type": "string"}, "box": _BOX}, []),
    _fn("classify", "Record your verdict on one or more instances. Batch them.",
        {"items": {"type": "array", "items": {"type": "object", "properties": {
            "instance_id": {"type": "string"},
            "role": {"type": "string", "enum": ["crop", "weed", "ignore"]},
            "species_id": {"type": ["string", "null"]},
            "height_in": {"type": ["number", "null"]},
            "leaf_count": {"type": ["integer", "null"]},
            "id_confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["instance_id", "role", "id_confidence"], "additionalProperties": False}}},
        ["items"]),
    _fn("measure", "Compute weed and crop measurements for this photo with code.",
        {"view_type": {"type": "string", "enum": list(VIEW_TYPES)}}, ["view_type"]),
    _fn("finish", "Close this stop after measure.",
        {"crop_stage": {"type": ["string", "null"], "description": "Crop growth stage such as V3, or null"},
         "notes": {"type": "string"}},
        ["crop_stage", "notes"]),
]


@dataclass
class ToolResult:
    output: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (caption, data URL)
    finished: bool = False


class ToolRunner:
    def __init__(self, state: StopState, sam, weeds: WeedBook, max_calls: int = 12):
        self.state = state
        self.sam = sam
        self.weeds = weeds
        self.max_calls = max_calls

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

    def _tool_segment(self, phrase: str, kind: str) -> ToolResult:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        phrase = str(phrase).strip()
        if not phrase:
            raise ValueError("phrase is empty")
        found, notes = [], []
        for n, record in enumerate(self.sam.segment_concept(self.state.image, phrase), start=1):
            inst, note = self._add(record, kind, "concept", phrase)
            if inst is not None:
                found.append(inst)
            if note and note != NOTHING:
                notes.append(f"match {n}: {note}")
        dropped = sum(note.endswith(MOSTLY_CROP_NOTE) for note in notes)
        summary = f'Segment "{phrase}": {len(found)} found'
        if dropped:
            summary += f", {dropped} dropped as crop"
        self.state.log("segment", summary, [i.id for i in found])
        return ToolResult(json.dumps({"total_found": len(found), "found": [self._brief(i) for i in found[:MAX_LISTED]],
                                      "notes": notes[:MAX_LISTED]}))

    def _tool_mark(self, kind: str, boxes: list[dict]) -> ToolResult:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_BOXES:
            raise ValueError(f"give 1 to {MAX_BOXES} boxes")
        pixel_boxes = [to_pixels(item["box"], self.state.size) for item in boxes]
        labels = [str(item.get("label") or kind).strip() for item in boxes]
        records: dict[str, SamRecord] = {}
        for record in self.sam.segment_boxes(self.state.image, pixel_boxes):
            records.setdefault(record.object_id, record)  # object i + 1 is box i
        asked = {str(n) for n in range(1, len(boxes) + 1)}
        unexpected = [object_id for object_id in records if object_id not in asked]
        if unexpected:
            self.state.log("mark", f"SAM returned unexpected object ids {unexpected[:5]}", ok=False)
        found, notes = [], []
        for n, label in enumerate(labels, start=1):
            record = records.get(str(n))
            inst, note = self._add(record, kind, "prompt", label) if record else (None, NOTHING)
            if inst is not None:
                found.append({"box_no": n, **self._brief(inst)})
            if note:
                notes.append(f"box {n}: {note}")
        if kind == "weed" and len(boxes) == MAX_BOXES:
            self.state.box_cap_hit = True  # more weeds may be out there: the count becomes a floor
        self.state.log("mark", f"Mark {kind}: {len(found)} of {len(boxes)} boxes kept", [f["id"] for f in found])
        return ToolResult(json.dumps({"found": found, "notes": notes}))

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
        scale = self.state.image.full.width / self.state.image.work.width
        crop = self.state.image.full.crop(tuple(int(v * scale) for v in (left, top, right, bottom)))
        crop.thumbnail((ZOOM_MAX_PX, ZOOM_MAX_PX))
        self.state.log("zoom", f"Zoom on {label}", [instance_id] if instance_id else [])
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
            named = item.get("species_id")
            if role == "weed" and named not in (None, ""):  # a crop's name is not a weed species: ignore it
                species = named if named in self.weeds else self.weeds.resolve(named)
                if species is None:
                    raise ValueError(f"unknown species_id {named}; use an id from the allowed list or null")
            confidence = item["id_confidence"]
            if confidence not in CONFIDENCES:
                raise ValueError(f"id_confidence must be one of {list(CONFIDENCES)}")
            height = _number(item.get("height_in"), "height_in")
            leaves = _number(item.get("leaf_count"), "leaf_count")
            resolved.append((inst, role, species, confidence, height, None if leaves is None else int(leaves)))
        for inst, role, species, confidence, height, leaves in resolved:
            inst.role, inst.species_id, inst.id_confidence = role, species, confidence
            inst.height_in, inst.leaf_count = height, leaves
            inst.check_this = confidence == "low" or self.state.is_patch(inst)  # a confident verdict clears the flag
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

    def _tool_finish(self, crop_stage: str | None, notes: str) -> ToolResult:
        unclassified = self.state.unclassified()
        if unclassified and self.state.calls < self.max_calls:
            self.state.log("finish", "Refused: unclassified weeds remain", ok=False)
            return ToolResult(f"error: classify these first: {unclassified[:20]}")
        if self.state.view_type is None:
            self.state.log("finish", "Refused: measure has not run", ok=False)
            return ToolResult("error: call measure first")
        self.state.crop_stage = None if crop_stage is None else (str(crop_stage).strip() or None)
        self.state.notes = str(notes or "")
        self.state.finished = True
        self.state.log("finish", "Stop closed")
        return ToolResult("ok", finished=True)
