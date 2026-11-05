"""Shared data shapes.

Boxes are (left, top, right, bottom) in work-image pixels, with right and bottom exclusive.
"""
from typing import Literal

from pydantic import BaseModel, Field

Box = tuple[int, int, int, int]
Role = Literal["crop", "weed", "ignore", "unknown"]
Outcome = Literal["no_action", "scout_again", "treat"]
ViewType = Literal["row", "top_down", "close_up", "wide"]
Crop = Literal["corn", "soybean", "alfalfa"]


class Instance(BaseModel):
    id: str
    source: Literal["concept", "prompt", "candidate", "patch"]  # patch: a weed area measured from green pixels
    prompt: str
    box: Box
    sam_confidence: float | None = None
    area_px: int
    role: Role = "unknown"
    species_id: str | None = None
    height_in: float | None = None
    leaf_count: int | None = None
    id_confidence: Literal["high", "medium", "low"] | None = None
    check_this: bool = False
    plant_count: int = 1  # a weed area holds several plants: its separate green pieces


class AgentStep(BaseModel):
    t_ms: int
    tool: str
    summary: str
    instance_ids: list[str] = Field(default_factory=list)
    ok: bool = True
    detail: list[str] = Field(default_factory=list)  # a "think" step's reasoning lines
    picture: str | None = None  # file name of the picture saved for this step, if any


class StopResult(BaseModel):
    stop_index: int
    image_hash: str
    view_type: ViewType = "row"
    gps: tuple[float, float] | None = None
    crop_stage: str | None = None
    weed_cover_pct: float = 0.0
    crop_cover_pct: float = 0.0
    weed_count: int = 0
    count_approximate: bool = False
    density_per_m2: float | None = None
    in_row_weed_share: float | None = None
    species: dict[str, int] = Field(default_factory=dict)
    species_cover_pct: dict[str, float] = Field(default_factory=dict)
    species_density: dict[str, float] | None = None
    species_max_height_in: dict[str, float] = Field(default_factory=dict)
    max_height_in: float | None = None
    flags: list[str] = Field(default_factory=list)
    notes: str = ""
    crop_seen: str | None = None  # the crop the agent saw: corn, soybean, alfalfa, other or unsure
    weeds_estimate: int | None = None  # how many weed plants the agent said it saw, before marking
    all_weeds_marked: bool | None = None  # the agent's own word at finish; None when it never finished
    steps: list[AgentStep] = Field(default_factory=list)
    status: Literal["ok", "no_crop_found", "failed"] = "ok"


class SpeciesSummary(BaseModel):
    species_id: str
    # Saved with the report: a weed named at run time is not in the catalog when the report is opened again.
    common_name: str | None = None
    scientific_name: str | None = None
    stops_present: int
    frequency: float
    mean_cover_pct: float
    mean_density_per_m2: float | None
    max_height_in: float | None
    outcome: Outcome


class UseRecord(BaseModel):
    reg_no: str
    product_name: str
    active_ingredients: list[str]
    herbicide_groups: list[str]
    crop: Crop
    species_id: str
    scientific_name: str | None = None  # given for weeds off the catalog, to match the name the agent gives
    weed_label_text: str
    timing: Literal["preemergence", "postemergence", "burndown"]
    max_weed_height_in: float | None = None
    crop_stage_min: str | None = None
    crop_stage_max: str | None = None
    rate_min: float | None = None
    rate_max: float | None = None
    rate_unit: str | None = None
    phi_days: int | None = None
    rei_hours: int | None = None
    page: int
    quote: str
    pdf_file: str | None = None
    requires_trait: str | None = None  # seed trait the crop must carry (e.g. glyphosate-tolerant); from focus_products.csv


class LabelOption(BaseModel):
    record: UseRecord
    over_size_limit: bool = False


class FieldReport(BaseModel):
    walk_id: str
    crop: Crop
    stops: list[StopResult]
    species_summary: list[SpeciesSummary]
    decision: Outcome
    treat_mode: Literal["spot", "broadcast"] | None = None
    urgency: str | None = None
    rule_trace: list[str]
    explanation: str
    options: list[LabelOption]
    crop_mismatch: str | None = None  # a warning when photos show another crop than the walk was scouted as
