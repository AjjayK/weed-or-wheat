"""Find label-backed options for one species on one crop. A filter over a table, not search over text."""
import json
from pathlib import Path

from scout.models import LabelOption, UseRecord

DEFAULT_PATH = Path(__file__).with_name("use_records.json")
STAGES = ["VE", "VC"] + [f"V{n}" for n in range(1, 21)] + ["VT"] + [f"R{n}" for n in range(1, 9)]


def stage_index(stage: str | None) -> int | None:
    if not stage:
        return None
    key = stage.strip().upper().replace(" ", "")
    return STAGES.index(key) if key in STAGES else None


def load_records(path: Path = DEFAULT_PATH) -> list[UseRecord]:
    path = Path(path)
    if not path.exists():
        return []
    return [UseRecord.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def find_label_options(records: list[UseRecord], crop: str, species_id: str, weed_height_in: float | None,
                       crop_stage: str | None) -> list[LabelOption]:
    stage = stage_index(crop_stage)
    matches = []
    for record in records:
        if record.crop != crop or record.species_id != species_id or record.timing != "postemergence":
            continue
        low, high = stage_index(record.crop_stage_min), stage_index(record.crop_stage_max)
        if stage is not None and low is not None and high is not None and not low <= stage <= high:
            continue
        over = (weed_height_in is not None and record.max_weed_height_in is not None
                and record.max_weed_height_in < weed_height_in)
        matches.append(LabelOption(record=record, over_size_limit=over))
    matches.sort(key=lambda o: (o.over_size_limit, -(o.record.max_weed_height_in or 0)))
    best_per_product: dict[str, LabelOption] = {}
    for option in matches:
        best_per_product.setdefault(option.record.reg_no, option)
    ordered, rest, seen_groups = [], [], set()
    for option in best_per_product.values():
        groups = set(option.record.herbicide_groups)
        if option.over_size_limit or groups <= seen_groups:
            rest.append(option)
        else:
            ordered.append(option)
            seen_groups |= groups
    return ordered + rest
