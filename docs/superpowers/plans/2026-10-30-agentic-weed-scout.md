# Agentic Weed Scout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and ship (by Wed 2026-11-04, 9 PM ET) a web app where a farmer uploads the photos from one field walk. A Muse Spark agent drives SAM 3.1 to separate weeds from crop, code measures weed intensity and applies threshold rules, and the report lists herbicide options backed by EPA labels with page citations.

**Architecture:**
- A Python package `scout/` with small focused modules: vision (SAM client, mask math, rows, measurement), agent (Muse Spark tool loop per stop, field step), agronomy (weed data, decision rules), labels (EPA fetch, PDF extraction, lookup), report (HTML/CSV), app (ingest, pipeline, FastAPI).
- The agent decides what to segment and how to classify each plant. Every number in the report is computed by code.
- Stops run concurrently in threads, and progress streams to the browser over Server-Sent Events.

**Tech Stack:**
- Python 3.10, FastAPI, Uvicorn, Pydantic v2, NumPy, SciPy, Pillow + pillow-heif.
- OpenAI SDK pointed at the Meta Model API (`https://api.meta.ai/v1`), `meta-sam-parser`, pypdf, Jinja2, PyYAML, requests, pytest.
- Hugging Face Spaces (Docker) for hosting.

**Spec:** `docs/superpowers/specs/2026-10-30-agentic-scout-design.md`

---

## Conventions

- **Python version:** run everything in a venv with Python 3.10+, from the repo root.
- **Commands:** commands are written for Git Bash. In PowerShell, activate with `.venv\Scripts\Activate.ps1` instead of `source .venv/Scripts/activate`.
- **Commits:** end every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The commit steps below show the subject line only.
- **User tasks:** tasks marked **[USER]** need the human: an account, a choice, photos, or a recording. An agent executing the plan stops at these and asks.
- **API keys:** tests never call the network. Only `scripts/` and the running app use real API keys.

## Facts verified while writing this plan (2026-10-30)

- **Model ids:** `muse-spark-1.3` (latest Muse Spark) and `sam-3.1`, both on the Responses API. `client.responses.create(model=..., input=...)`, with the key from `MODEL_API_KEY`.
- **Tool calls:**
  - The model returns `function_call` items in `response.output`, each with `call_id`, `name`, and `arguments` (a JSON string).
  - You answer with `{"type": "function_call_output", "call_id": ..., "output": "<string>"}` plus `previous_response_id`.
  - Images are processed **only in user-role messages**, so zoom crops are sent as a follow-up user message, never inside a tool output.
- **Structured output:** `text={"format": {"type": "json_schema", "name": ..., "schema": ..., "strict": true}}`. In strict mode every property is required and every object needs `additionalProperties: false`.
- **PDF input:** `{"type": "input_file", "filename": ..., "file_data": "data:application/pdf;base64,..."}`. Up to 50 MB inline; text is read from the first 100 pages.
- **SAM output:**
  - Format: `<0f>ID<|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|><|mask;x=0;y=0[;c=];data=H,W,<payload>|>`, where `x2`/`y2` are inclusive.
  - The mask raster covers the **box**, not the whole frame.
  - The payload is base85 and can contain `,` `;` `<` `>`, but never `|`.
- **`meta-sam-parser` 0.1.0:**
  - `decode_mask_to_raster(SegmentationMask(encoding, payload, width, height)) -> bytes` (row-major 0/1). The `payload` includes its `!`/`~` marker.
  - The public stream parser is async-only, so we parse the final text ourselves with a regex and decode masks with the library.
  - A 1024×768 mask takes about 4 s to decode (pure Python), so SAM gets a working copy at most 1024 px on the long side.
- **`build_object_prompt` output:**
  - `build_object_prompt(size=(100, 80), objects=[PromptObject(id=1, prompts=[FramePrompt(frame=0, points=[PromptPoint(10, 20), PromptPoint(30, 40, label="negative")])])])` returns `<0f>1<|point;x=10;y=20;w=100;h=80|>-<|point;x=30;y=40;w=100;h=80|>`.
  - `PromptBox(left, top, right, bottom)` treats right/bottom as exclusive.
- **EPA PPLS:**
  - `GET https://ordspub.epa.gov/ords/pesticides/cswu/ppls/<reg_no>` returns `items[0]` with `productname` and these lists: `active_ingredients[].active_ing`, `sites[].site`, `pests[].pest`, `types[].type`, `pdffiles[].pdffile` (named `NNNNNN-NNNNN-YYYYMMDD.pdf`).
  - PDFs live at `https://www3.epa.gov/pesticides/chem_search/ppls/<pdffile>`.
  - Verified TLS works today; a legacy-renegotiation flag is added for OpenSSL 3 environments.

## Refinements to the spec made in this plan

1. All code lives under one package, `scout/`, so imports are unambiguous: `scout/app`, `scout/vision`, `scout/agent`, `scout/agronomy`, `scout/labels`, `scout/report`, `scout/web`. The evaluation folder is `evaluation/`, because `eval` shadows a builtin.
2. `Instance.source` gains `"candidate"` for pieces derived as plant minus crop.
3. `segment` takes a `kind` (`crop`, `all_plants`, or `weed`); `segment_at` takes a `kind` too. `measure(view_type)` sets the view type, and `finish(crop_stage, notes)` no longer takes one. `classify` accepts a batch of items.
4. `StopResult` gains `species_cover_pct`, `species_density`, and `species_max_height_in`. The rules need per-species values.
5. `UseRecord` gains `pdf_file`, so the report can link straight to the label page. `FieldReport.options` is `list[LabelOption]`, where `LabelOption` wraps a record with an `over_size_limit` flag.
6. Zoom crops always go in a user message, since the docs say images are only processed there.
7. Offline tests build realistic SAM text with the parser's own encoder instead of recorded fixtures. Spike outputs go to the git-ignored `spike_out/`.
8. Sample walks are run once, then committed as cached reports in `demo/`. The live site can then show them without spending API credits. GPS in samples is shifted by a random offset.

## File map

```
requirements.txt, pyproject.toml, .env.example, Dockerfile, .dockerignore, README.md
scout/config.py              Settings from env
scout/models.py              Pydantic data shapes
scout/retry.py               Shared retry policy and BadRequest
scout/app/ingest.py          Upload -> StopImage (HEIC, EXIF rotation, GPS, sizes, hash)
scout/app/pipeline.py        Walk runner: stops in threads -> field report; cache; samples
scout/app/main.py            FastAPI routes + SSE
scout/vision/masks.py        Mask paste/union/pieces/bbox/pad
scout/vision/sam_parse.py    SAM text -> SamRecord with box-local mask
scout/vision/sam_client.py   SAM requests (concept + point/box), retry, cache
scout/vision/overlay.py      Colored overlay image
scout/vision/rows.py         Crop-row fitting and in-row band
scout/vision/measure.py      Metrics from instances + masks
scout/agent/llm.py           Responses API wrapper: LlmTurn/ToolCall, user_message
scout/agent/state.py         StopState: instances, masks, candidates, step log, overlay
scout/agent/tools.py         Tool schemas + ToolRunner
scout/agent/prompts.py       Agent instructions
scout/agent/stop_agent.py    Tool loop for one stop + finalize
scout/agent/field_agent.py   Field decision, label options, checked explanation
scout/agronomy/weeds.yaml    35 species with names, synonyms, thresholds, sources
scout/agronomy/weeds.py      WeedBook loader/resolver
scout/agronomy/rules.py      Decision rules and urgency
scout/labels/epa.py          EPA PPLS fetch + PDF download
scout/labels/extract.py      Muse Spark PDF -> UseRecords; quote verification
scout/labels/build_labels.py CLI: focus_products.csv -> use_records.json
scout/labels/lookup.py       find_label_options
scout/report/render.py       Report HTML + CSV
scout/report/templates/report.html
scout/web/index.html, app.js, style.css
scripts/spike_sam.py, spike_agent.py, run_stop.py, make_sample.py
evaluation/run_eval.py, evaluation/labels.csv
tests/helpers.py, tests/test_*.py
docs/spike-notes.md, docs/demo-script.md
```

---

# Day 1: Fri Oct 30. Foundations and API test

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.env.example`, `scout/__init__.py`, `scout/app/__init__.py`, `scout/vision/__init__.py`, `scout/agent/__init__.py`, `scout/agronomy/__init__.py`, `scout/labels/__init__.py`, `scout/report/__init__.py`, `tests/__init__.py`, `scout/config.py`
- Modify: `.gitignore`
- Test: `tests/test_config.py`

- [ ] **Step 1: Create the venv and dependency files**

`requirements.txt`:
```
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
pydantic>=2.6
numpy>=1.26
scipy>=1.11
pillow>=10.3
pillow-heif>=0.16
openai>=1.66,<2
meta-sam-parser>=0.1.0
pypdf>=4.2
pyyaml>=6.0
jinja2>=3.1
requests>=2.31
python-dotenv>=1.0
httpx>=0.27
pytest>=8.0
```

`pyproject.toml`:
```toml
[project]
name = "weed-or-wheat"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

`.env.example`:
```
MODEL_API_KEY=your-meta-model-api-key
AGENT_MODEL=muse-spark-1.3
SAM_MODEL=sam-3.1
```

Append to `.gitignore`:
```
.pytest_cache/
store/
spike_out/
run_out/
scout/labels/pdfs/
evaluation/results.csv
```

Create empty `__init__.py` files at: `scout/`, `scout/app/`, `scout/vision/`, `scout/agent/`, `scout/agronomy/`, `scout/labels/`, `scout/report/`, `tests/`.

Run:
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```
Expected: installs without errors.

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from scout.config import Settings, get_settings


def test_get_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k-123")
    monkeypatch.setenv("AGENT_MODEL", "muse-spark-1.2")
    settings = get_settings()
    assert settings.api_key == "k-123"
    assert settings.agent_model == "muse-spark-1.2"
    assert settings.sam_model == "sam-3.1"


def test_settings_defaults():
    settings = Settings(api_key="x")
    assert settings.base_url == "https://api.meta.ai/v1"
    assert settings.work_max_px == 1024 and settings.max_tool_calls == 12
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.config'`

- [ ] **Step 4: Write the implementation**

`scout/config.py`:
```python
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
    max_tool_calls: int = 12
    stop_timeout_s: float = 90.0
    max_concurrent_stops: int = 4
    max_photos: int = 10
    max_upload_bytes: int = 25_000_000
    walks_per_ip_per_hour: int = 3


def get_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY", ""),
        agent_model=os.environ.get("AGENT_MODEL", "muse-spark-1.3"),
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml .env.example .gitignore scout tests
git commit -m "chore: scaffold scout package and settings"
```

### Task 2: Data models

**Files:**
- Create: `scout/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from scout.models import FieldReport, Instance, StopResult


def test_instance_defaults():
    inst = Instance(id="s0-i1", source="concept", prompt="plant", box=(0, 0, 10, 10), area_px=100)
    assert inst.role == "unknown" and inst.check_this is False


def test_stop_result_round_trip():
    stop = StopResult(stop_index=2, image_hash="abc", species={"waterhemp": 3}, species_cover_pct={"waterhemp": 1.5})
    again = StopResult.model_validate_json(stop.model_dump_json())
    assert again == stop and again.status == "ok"


def test_field_report_minimal():
    report = FieldReport(walk_id="w", crop="soybean", stops=[], species_summary=[], decision="no_action",
                         rule_trace=[], explanation="", options=[])
    assert report.treat_mode is None and report.urgency is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.models'`

- [ ] **Step 3: Write the implementation**

`scout/models.py`:
```python
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
    source: Literal["concept", "prompt", "candidate"]
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


class AgentStep(BaseModel):
    t_ms: int
    tool: str
    summary: str
    instance_ids: list[str] = Field(default_factory=list)
    ok: bool = True


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
    steps: list[AgentStep] = Field(default_factory=list)
    status: Literal["ok", "no_crop_found", "failed"] = "ok"


class SpeciesSummary(BaseModel):
    species_id: str
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scout/models.py tests/test_models.py
git commit -m "feat: add shared data models"
```

### Task 3: Retry policy

**Files:**
- Create: `scout/retry.py`
- Test: `tests/test_retry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retry.py`:
```python
import pytest

from scout.retry import BadRequest, with_retry


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_retries_transient_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise StatusError(429)
        return "ok"

    assert with_retry(fn, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_bad_request_is_not_retried():
    calls = []

    def fn():
        calls.append(1)
        raise StatusError(400)

    with pytest.raises(BadRequest):
        with_retry(fn, sleep=lambda s: None)
    assert len(calls) == 1


def test_non_transient_error_raises_immediately():
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with_retry(fn, sleep=lambda s: None)


def test_gives_up_after_retries():
    sleeps = []

    def fn():
        raise StatusError(503)

    with pytest.raises(StatusError):
        with_retry(fn, retries=2, sleep=sleeps.append)
    assert sleeps == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.retry'`

- [ ] **Step 3: Write the implementation**

`scout/retry.py`:
```python
"""Retry policy shared by the SAM and Muse Spark clients."""
import time
from typing import Callable, TypeVar

T = TypeVar("T")

RETRY_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_ERRORS = {"APIConnectionError", "APITimeoutError"}


class BadRequest(Exception):
    """The API rejected the request (HTTP 400). Retrying the same request will not help."""


def with_retry(fn: Callable[[], T], retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status == 400:
                raise BadRequest(str(error)) from error
            transient = status in RETRY_STATUS or type(error).__name__ in TRANSIENT_ERRORS
            if not transient or attempt == retries:
                raise
            sleep(2 ** attempt)
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/retry.py tests/test_retry.py
git commit -m "feat: add retry policy with BadRequest"
```

### Task 4: Mask helpers

**Files:**
- Create: `scout/vision/masks.py`
- Test: `tests/test_masks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_masks.py`:
```python
import numpy as np

from scout.vision import masks as mk


def test_paste_clips_to_canvas():
    canvas = mk.paste(np.ones((4, 4), dtype=bool), (-2, 8, 2, 12), (10, 10))
    assert canvas.sum() == 4
    assert canvas[8:10, 0:2].all()


def test_union_and_bbox():
    a = mk.paste(np.ones((2, 2), bool), (1, 1, 3, 3), (10, 10))
    b = mk.paste(np.ones((2, 2), bool), (6, 5, 8, 7), (10, 10))
    both = mk.union([a, b], (10, 10))
    assert both.sum() == 8
    assert mk.bbox(both) == (1, 1, 8, 7)


def test_bbox_of_empty_mask():
    assert mk.bbox(np.zeros((5, 5), bool)) == (0, 0, 0, 0)


def test_pieces_splits_blobs_and_drops_small():
    canvas = np.zeros((40, 40), bool)
    canvas[2:12, 2:12] = True
    canvas[20:30, 25:35] = True
    canvas[35, 35] = True
    found = mk.pieces(canvas, min_area=20)
    assert sorted(int(p.sum()) for p in found) == [100, 100]


def test_pad_box_clamps():
    assert mk.pad_box((0, 0, 10, 10), 0.5, (12, 12)) == (0, 0, 12, 12)
    assert mk.pad_box((10, 10, 20, 20), 0.1, (100, 100)) == (9, 9, 21, 21)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_masks.py -v`
Expected: FAIL with `ImportError` for `scout.vision.masks`

- [ ] **Step 3: Write the implementation**

`scout/vision/masks.py`:
```python
"""Boolean mask helpers. Masks are numpy bool arrays shaped (height, width); sizes are (width, height)."""
import numpy as np
from scipy import ndimage

from scout.models import Box


def paste(local: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """Place a box-local mask onto a full canvas, clipping anything outside it."""
    width, height = size
    canvas = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    l, t = max(left, 0), max(top, 0)
    r, b = min(right, width), min(bottom, height)
    if r <= l or b <= t:
        return canvas
    canvas[t:b, l:r] = local[t - top:b - top, l - left:r - left]
    return canvas


def union(masks: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    out = np.zeros((height, width), dtype=bool)
    for mask in masks:
        out |= mask
    return out


def pieces(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Split a mask into connected pieces, after an opening that removes slivers along crop edges."""
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndimage.label(cleaned)
    if count == 0:
        return []
    areas = ndimage.sum(cleaned, labels, index=range(1, count + 1))
    return [labels == index + 1 for index, area in enumerate(areas) if area >= min_area]


def bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pad_box(box: Box, frac: float, size: tuple[int, int]) -> Box:
    left, top, right, bottom = box
    pad_x, pad_y = int((right - left) * frac), int((bottom - top) * frac)
    width, height = size
    return (max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_masks.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/masks.py tests/test_masks.py
git commit -m "feat: add mask helpers"
```

### Task 5: SAM output parser

**Files:**
- Create: `scout/vision/sam_parse.py`, `tests/helpers.py`
- Test: `tests/test_sam_parse.py`

- [ ] **Step 1: Write the test helpers and the failing test**

`tests/helpers.py`:
```python
"""Shared test helpers."""
import numpy as np
from meta_sam_parser._mask_codec import _encode_raster  # private encoder; only used to build realistic SAM text


def sam_record_text(object_id, box, mask, frame_size, confidence=None):
    left, top, right, bottom = box
    width, height = frame_size
    payload = _encode_raster(mask.astype(np.uint8).ravel().tolist(), mask.shape[1], mask.shape[0])
    conf = f";c={confidence}" if confidence is not None else ""
    return (f"{object_id}<|box;x1={left};y1={top};x2={right - 1};y2={bottom - 1};w={width};h={height}{conf}|>"
            f"<|mask;x=0;y=0{conf};data={mask.shape[0]},{mask.shape[1]},{payload}|>")


def sam_text(records, frame_size):
    """records: list of (object_id, box, mask) or (object_id, box, mask, confidence)."""
    return "<0f>" + ",".join(sam_record_text(r[0], r[1], r[2], frame_size, *r[3:]) for r in records)
```

`tests/test_sam_parse.py`:
```python
import numpy as np

from scout.vision.sam_parse import parse_sam_output
from tests.helpers import sam_text


def test_parses_two_records_with_confidence():
    first = np.zeros((4, 6), bool)
    first[1:3, 2:5] = True
    second = np.ones((3, 3), bool)
    text = sam_text([("0", (10, 20, 16, 24), first, 0.91), ("1", (0, 0, 3, 3), second, 0.5)], (100, 80))
    records = parse_sam_output(text)
    assert [r.object_id for r in records] == ["0", "1"]
    assert records[0].box == (10, 20, 16, 24)
    assert records[0].confidence == 0.91
    assert np.array_equal(records[0].mask, first)
    full = records[0].full_mask((100, 80))
    assert full.shape == (80, 100) and full.sum() == first.sum() and full[21, 12]


def test_resizes_mask_that_does_not_match_box():
    text = sam_text([("0", (0, 0, 4, 4), np.ones((2, 2), bool))], (10, 10))
    record = parse_sam_output(text)[0]
    assert record.mask.shape == (4, 4) and record.mask.all()


def test_empty_output_gives_no_records():
    assert parse_sam_output("") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.sam_parse'`

- [ ] **Step 3: Write the implementation**

`scout/vision/sam_parse.py`:
```python
"""Parse SAM 3.1 output text into records with box-local boolean masks.

Record grammar: ID<|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|><|mask;x=0;y=0[;c=];data=H,W,<payload>|>
x2/y2 are inclusive. The mask raster covers the box. The base85 payload never contains '|'.
"""
import re
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from PIL import Image

from scout.models import Box
from scout.vision.masks import paste

_RECORD = re.compile(r"([A-Za-z0-9_-]+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_DATA = re.compile(r"(\d+),(\d+),([!~][^|]+)")


@dataclass
class SamRecord:
    object_id: str
    box: Box
    confidence: float | None
    mask: np.ndarray  # bool, shaped (box height, box width)

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        return paste(self.mask, self.box, size)


def _fields(body: str) -> dict[str, str]:
    out = {}
    for part in body.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def parse_sam_output(text: str) -> list[SamRecord]:
    records = []
    for match in _RECORD.finditer(text):
        object_id, box_body, mask_body = match.groups()
        box = _fields(box_body)
        head, _, data = mask_body.partition("data=")  # data is last, and its payload may contain ';'
        parsed = _DATA.fullmatch(data)
        if parsed is None:
            continue
        height, width, payload = int(parsed[1]), int(parsed[2]), parsed[3]
        encoding = "lossless" if payload.startswith("~") else "one_bit"
        raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))
        local = np.frombuffer(raster, dtype=np.uint8).reshape(height, width).astype(bool)
        left, top = int(box["x1"]), int(box["y1"])
        right, bottom = int(box["x2"]) + 1, int(box["y2"]) + 1
        if local.shape != (bottom - top, right - left):
            local = np.array(Image.fromarray(local).resize((right - left, bottom - top), Image.NEAREST)).astype(bool)
        confidence = box.get("c") or _fields(head).get("c")
        records.append(SamRecord(object_id, (left, top, right, bottom), float(confidence) if confidence else None, local))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam_parse.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/sam_parse.py tests/helpers.py tests/test_sam_parse.py
git commit -m "feat: parse SAM 3.1 output into box-local masks"
```

### Task 6: Image ingest

**Files:**
- Create: `scout/app/ingest.py`
- Modify: `tests/helpers.py` (append)
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Append helpers and write the failing test**

Append to `tests/helpers.py`:
```python
import io

from PIL import Image

from scout.app.ingest import load_image


def png_bytes(width=100, height=100, color=(90, 140, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_image(width=100, height=100, index=0, color=(90, 140, 60)):
    return load_image(index, png_bytes(width, height, color))
```

`tests/test_ingest.py`:
```python
import io

from PIL import Image

from scout.app.ingest import gps_from_ifd, load_image
from tests.helpers import png_bytes


def test_load_image_makes_full_and_work_sizes():
    image = load_image(3, png_bytes(3000, 2000))
    assert image.index == 3
    assert max(image.full.size) == 2048 and max(image.work.size) == 1024
    assert image.size == image.work.size
    assert len(image.hash) == 16 and image.gps is None


def test_exif_orientation_is_applied():
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20)).save(buf, format="JPEG", exif=exif)
    assert load_image(0, buf.getvalue()).work.size == (20, 40)


def test_gps_from_ifd():
    assert gps_from_ifd({1: "N", 2: (41.0, 30.0, 0.0), 3: "W", 4: (93.0, 36.0, 0.0)}) == (41.5, -93.6)
    assert gps_from_ifd({}) is None


def test_work_data_url_is_jpeg():
    assert load_image(0, png_bytes(50, 50)).work_data_url.startswith("data:image/jpeg;base64,")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.app.ingest'`

- [ ] **Step 3: Write the implementation**

`scout/app/ingest.py`:
```python
"""Turn uploaded bytes into a StopImage: rotation applied, two sizes, GPS, and a content hash."""
import base64
import hashlib
import io
from dataclasses import dataclass
from functools import cached_property

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

GPS_IFD = 0x8825


def to_data_url(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class StopImage:
    index: int
    hash: str
    full: Image.Image  # long side <= full_max_px, for zoom crops
    work: Image.Image  # long side <= work_max_px, what SAM and the agent see
    gps: tuple[float, float] | None

    @property
    def size(self) -> tuple[int, int]:
        return self.work.size

    @cached_property
    def work_data_url(self) -> str:
        return to_data_url(self.work)


def _shrink(image: Image.Image, max_px: int) -> Image.Image:
    out = image.copy()
    out.thumbnail((max_px, max_px))
    return out


def _dms(value) -> float:
    degrees, minutes, seconds = (float(v) for v in value)
    return degrees + minutes / 60 + seconds / 3600


def gps_from_ifd(gps: dict) -> tuple[float, float] | None:
    try:
        lat, lon = _dms(gps[2]), _dms(gps[4])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return (round(lat, 6), round(lon, 6))


def load_image(index: int, data: bytes, full_max_px: int = 2048, work_max_px: int = 1024) -> StopImage:
    image = Image.open(io.BytesIO(data))
    gps = gps_from_ifd(dict(image.getexif().get_ifd(GPS_IFD)))
    image = ImageOps.exif_transpose(image).convert("RGB")
    return StopImage(
        index=index,
        hash=hashlib.sha256(data).hexdigest()[:16],
        full=_shrink(image, full_max_px),
        work=_shrink(image, work_max_px),
        gps=gps,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/app/ingest.py tests/helpers.py tests/test_ingest.py
git commit -m "feat: ingest photos with rotation, GPS, and two sizes"
```

### Task 7: SAM client

**Files:**
- Create: `scout/vision/sam_client.py`
- Test: `tests/test_sam_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sam_client.py`:
```python
import numpy as np
import pytest

from scout.retry import BadRequest
from scout.vision.sam_client import SamClient
from tests.helpers import make_image, sam_text


def test_segment_concept_builds_request_and_parses():
    image = make_image(64, 48)
    sent = []

    def send(payload):
        sent.append(payload)
        return sam_text([("0", (1, 2, 5, 6), np.ones((4, 4), bool), 0.8)], image.size)

    records = SamClient(send, model="sam-3.1", sleep=lambda s: None).segment_concept(image, "  plant ")
    assert len(records) == 1 and records[0].confidence == 0.8
    content = sent[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "plant"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert sent[0]["model"] == "sam-3.1"
    assert sent[0]["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}


def test_repeated_prompt_uses_cache():
    image = make_image(32, 32)
    calls = []
    sam = SamClient(lambda p: calls.append(p) or "", sleep=lambda s: None)
    sam.segment_concept(image, "sky")
    sam.segment_concept(image, "sky")
    assert len(calls) == 1


def test_bad_request_raises():
    class Rejected(Exception):
        status_code = 400

    def send(payload):
        raise Rejected("no")

    with pytest.raises(BadRequest):
        SamClient(send, sleep=lambda s: None).segment_concept(make_image(32, 32), "plant")


def test_segment_at_sends_object_prompt():
    image = make_image(100, 80)
    sent = []
    sam = SamClient(lambda p: sent.append(p) or "", sleep=lambda s: None)
    sam.segment_at(image, points=[(10, 20, True), (30, 40, False)])
    text = sent[0]["input"][0]["content"][0]["text"]
    assert text == "<0f>1<|point;x=10;y=20;w=100;h=80|>-<|point;x=30;y=40;w=100;h=80|>"
    sam.segment_at(image, box=(5, 6, 50, 60))
    assert sent[1]["input"][0]["content"][0]["text"] == "<0f>1<|box;x1=5;y1=6;x2=49;y2=59;w=100;h=80|>"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.sam_client'`

- [ ] **Step 3: Write the implementation**

`scout/vision/sam_client.py`:
```python
"""SAM 3.1 requests over the Meta Model API: concept prompts and box/point prompts, cached per image."""
import time
from typing import Callable, Sequence

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt

from scout.app.ingest import StopImage
from scout.models import Box
from scout.retry import with_retry
from scout.vision.sam_parse import SamRecord, parse_sam_output


class SamClient:
    def __init__(self, send: Callable[[dict], str], model: str = "sam-3.1", sleep: Callable[[float], None] = time.sleep):
        self._send = send
        self._model = model
        self._sleep = sleep
        self._cache: dict[tuple[str, str], str] = {}

    @classmethod
    def from_settings(cls, settings) -> "SamClient":
        from openai import OpenAI

        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=120.0)
        return cls(lambda payload: client.responses.create(**payload).output_text, model=settings.sam_model)

    def fetch_raw(self, image: StopImage, text: str) -> str:
        key = (image.hash, text)
        if key not in self._cache:
            payload = {
                "model": self._model,
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": image.work_data_url},
                    ],
                }],
                "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            }
            self._cache[key] = with_retry(lambda: self._send(payload), sleep=self._sleep)
        return self._cache[key]

    def segment_concept(self, image: StopImage, phrase: str) -> list[SamRecord]:
        return parse_sam_output(self.fetch_raw(image, phrase.strip()))

    def segment_at(self, image: StopImage, box: Box | None = None,
                   points: Sequence[tuple[int, int, bool]] = ()) -> list[SamRecord]:
        prompt = FramePrompt(
            frame=0,
            box=PromptBox(*box) if box else None,
            points=[PromptPoint(int(x), int(y), label="positive" if positive else "negative") for x, y, positive in points],
        )
        text = build_object_prompt(size=image.size, objects=[PromptObject(id=1, prompts=[prompt])])
        return parse_sam_output(self.fetch_raw(image, text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam_client.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/sam_client.py tests/test_sam_client.py
git commit -m "feat: add SAM 3.1 client with cache and retries"
```

### Task 8: Overlay renderer

**Files:**
- Create: `scout/vision/overlay.py`
- Test: `tests/test_overlay.py`

- [ ] **Step 1: Write the failing test**

`tests/test_overlay.py`:
```python
import numpy as np
from PIL import Image

from scout.vision.overlay import render_overlay


def test_overlay_tints_crop_green_weed_red_and_outlines_flags():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    crop = np.zeros((10, 10), bool)
    crop[0, 0] = True
    weed = np.zeros((10, 10), bool)
    weed[9, 9] = True
    out = render_overlay(image, crop, weed, flagged=[(2, 2, 9, 9)])
    r, g, b = out.getpixel((0, 0))
    assert g > r and g > b
    r, g, b = out.getpixel((9, 9))
    assert r > g and r > b
    assert out.getpixel((4, 4)) == (163, 45, 45)
    assert out.getpixel((5, 5)) == (255, 255, 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.overlay'`

- [ ] **Step 3: Write the implementation**

`scout/vision/overlay.py`:
```python
"""Draw crop (green), weeds (red), the in-row band (gray), and boxes around plants that need a check."""
import numpy as np
from PIL import Image, ImageDraw

from scout.models import Box

CROP_RGB = (99, 153, 34)
WEED_RGB = (226, 75, 74)
BAND_RGB = (136, 135, 128)
FLAG_RGB = (163, 45, 45)


def render_overlay(image: Image.Image, crop: np.ndarray, weed: np.ndarray,
                   band: np.ndarray | None = None, flagged: list[Box] = ()) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, rgb, alpha in ((band, BAND_RGB, 0.25), (crop, CROP_RGB, 0.55), (weed, WEED_RGB, 0.65)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for left, top, right, bottom in flagged:
        draw.rectangle((left, top, right - 1, bottom - 1), outline=FLAG_RGB, width=3)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/overlay.py tests/test_overlay.py
git commit -m "feat: render crop and weed overlays"
```

### Task 9: [USER] API access and SAM test (go/no-go)

**Files:**
- Create: `scripts/spike_sam.py`, `docs/spike-notes.md`, `.env` (not committed)

- [ ] **Step 1: [USER] Get API access**
  1. Claim the $150 credits with the hackathon credit form. The link is in the participant guide, section 5.
  2. Add a card on the Meta Model API Billing page, and set a spend-limit notification.
  3. Create an API key.
  4. Copy `.env.example` to `.env` and paste the key in as `MODEL_API_KEY`.

- [ ] **Step 2: Confirm the model ids**

Run:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); from openai import OpenAI; c = OpenAI(base_url='https://api.meta.ai/v1', api_key=os.environ['MODEL_API_KEY']); print(sorted(m.id for m in c.models.list()))"
```
Expected: the list includes `muse-spark-1.3` and `sam-3.1`. If `models.list()` errors, skip this step; Step 4 will fail loudly if an id is wrong.

- [ ] **Step 3: Write the test script**

`scripts/spike_sam.py`:
```python
"""Day-1 test: run SAM 3.1 concept prompts on photos and save overlays.

Usage:
  python scripts/spike_sam.py --crop-prompt "soybean plant" field-images/IMG_4219.JPG field-images/IMG_3926.JPG
Writes spike_out/<photo>__<prompt>.png and spike_out/<photo>__candidates.png, and prints timings.
"""
import argparse
import re
import time
from pathlib import Path

from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.sam_client import SamClient
from scout.vision.sam_parse import parse_sam_output

OUT = Path("spike_out")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop-prompt", required=True)
    parser.add_argument("--extra", nargs="*", default=["grass", "sky"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    settings = get_settings()
    sam = SamClient.from_settings(settings)
    for photo in args.photos:
        image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
        stem = Path(photo).stem
        unions = {}
        for prompt in [args.crop_prompt, "plant", *args.extra]:
            started = time.time()
            raw = sam.fetch_raw(image, prompt)
            fetched = time.time()
            records = parse_sam_output(raw)
            decoded = time.time()
            unions[prompt] = mk.union([r.full_mask(image.size) for r in records], image.size)
            mismatched = sum(1 for r in records if r.mask.shape != (r.box[3] - r.box[1], r.box[2] - r.box[0]))
            confs = [round(r.confidence, 2) for r in records if r.confidence is not None]
            print(f"{stem} | {prompt!r}: {len(records)} found, api {fetched - started:.1f}s, "
                  f"decode {decoded - fetched:.1f}s, conf {confs[:8]}, mask/box size mismatches {mismatched}")
            render_overlay(image.work, unions[prompt], unions[prompt] & False).save(OUT / f"{stem}__{slug(prompt)}.png")
        crop = unions[args.crop_prompt]
        candidates = mk.pieces(unions["plant"] & ~crop, 0.0005 * image.size[0] * image.size[1])
        weed = mk.union(candidates, image.size)
        print(f"{stem} | plant minus crop: {len(candidates)} weed candidates")
        render_overlay(image.work, crop, weed).save(OUT / f"{stem}__candidates.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it on 6 photos**

Run (crop prompt per crop):
```bash
python scripts/spike_sam.py --crop-prompt "corn plant" field-images/IMG_4214.JPG field-images/IMG_4215.JPG
python scripts/spike_sam.py --crop-prompt "soybean plant" field-images/IMG_4219.JPG field-images/IMG_4220.JPG
python scripts/spike_sam.py --crop-prompt "alfalfa plant" field-images/IMG_3922.JPG field-images/IMG_3926.JPG --extra dandelion sky
```
Expected: one line per prompt with counts and timings, and PNGs in `spike_out/`.

- [ ] **Step 5: Decide go/no-go and write it down**

Open the `*__candidates.png` files. **Go** if the red pieces mostly sit on weeds rather than on crop, on at least 4 of the 6 photos. **No-go:** try `--extra "weed" "seedling" "green plant"` on the failures.

If the "plant" concept is still poor, adopt the fallback (spec §8.2.3). The agent marks points on weeds in the full photo and calls `segment_at`. The tools in Task 16 already support this; strengthen `prompts.py` step 3 wording in Task 17.

Create `docs/spike-notes.md` with what was observed:
```markdown
# Spike notes (2026-10-30)

## SAM 3.1
- Model ids confirmed: (list)
- Per photo: prompt -> instances found, API seconds, decode seconds
- Mask raster equals box size: yes/no (mismatch count)
- "plant" minus crop: go / no-go, with photo names
- Useful extra prompts: (e.g. "grass", "dandelion")

## Muse Spark (Task 11)
- Tool call round trip: works / issues
- Image in follow-up user message: works / issues
- Structured output: works / issues
- Typical latency per turn
```
Fill each bullet with the actual result.

- [ ] **Step 6: Commit**

```bash
git add scripts/spike_sam.py docs/spike-notes.md
git commit -m "chore: SAM 3.1 test script and notes"
```

### Task 10: Muse Spark wrapper

**Files:**
- Create: `scout/agent/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

`tests/test_llm.py`:
```python
from types import SimpleNamespace as NS

from scout.agent.llm import Llm, user_message


def _response(output, text="", rid="resp_1"):
    return NS(id=rid, output=output, output_text=text)


def test_turn_parses_function_calls():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _response([NS(type="reasoning"),
                          NS(type="function_call", call_id="c1", name="segment",
                             arguments='{"phrase": "plant", "kind": "all_plants"}')])

    turn = Llm(create, "muse-spark-1.3").turn([user_message("hi")], instructions="be brief",
                                             tools=[{"type": "function", "name": "segment"}],
                                             previous_response_id="resp_0")
    assert turn.response_id == "resp_1"
    assert turn.calls[0].name == "segment"
    assert turn.calls[0].arguments == {"phrase": "plant", "kind": "all_plants"}
    assert seen["previous_response_id"] == "resp_0" and seen["instructions"] == "be brief"
    assert seen["model"] == "muse-spark-1.3"


def test_invalid_arguments_are_flagged():
    create = lambda **k: _response([NS(type="function_call", call_id="c1", name="zoom", arguments="{not json")])
    assert "__invalid__" in Llm(create, "m").turn("x").calls[0].arguments


def test_text_format_is_passed_as_text_param():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _response([], text='{"records": []}')

    turn = Llm(create, "m").turn("x", text_format={"type": "json_schema", "name": "r", "schema": {}})
    assert seen["text"] == {"format": {"type": "json_schema", "name": "r", "schema": {}}}
    assert turn.text == '{"records": []}' and turn.calls == []


def test_user_message_with_images_and_file():
    msg = user_message("look", ["data:image/jpeg;base64,AAA"], file=("label.pdf", "data:application/pdf;base64,BBB"))
    assert [part["type"] for part in msg["content"]] == ["input_text", "input_image", "input_file"]
    assert msg["role"] == "user" and msg["type"] == "message"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.llm'`

- [ ] **Step 3: Write the implementation**

`scout/agent/llm.py`:
```python
"""Thin wrapper over the Responses API for Muse Spark: one call = one LlmTurn."""
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from scout.retry import with_retry


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class LlmTurn:
    response_id: str
    calls: list[ToolCall]
    text: str


def user_message(text: str, image_urls: Sequence[str] = (), file: tuple[str, str] | None = None) -> dict:
    """A user-role message. Images and files are only processed in user-role messages."""
    content: list[dict] = [{"type": "input_text", "text": text}]
    content += [{"type": "input_image", "image_url": url} for url in image_urls]
    if file is not None:
        content.append({"type": "input_file", "filename": file[0], "file_data": file[1]})
    return {"type": "message", "role": "user", "content": content}


class Llm:
    def __init__(self, create: Callable[..., Any], model: str, sleep: Callable[[float], None] = time.sleep):
        self._create = create
        self.model = model
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings) -> "Llm":
        from openai import OpenAI

        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=180.0)
        return cls(client.responses.create, settings.agent_model)

    def turn(self, input, instructions: str | None = None, tools: list[dict] | None = None,
             previous_response_id: str | None = None, text_format: dict | None = None) -> LlmTurn:
        kwargs: dict[str, Any] = {"model": self.model, "input": input}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if text_format:
            kwargs["text"] = {"format": text_format}
        response = with_retry(lambda: self._create(**kwargs), sleep=self._sleep)
        calls = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__invalid__": item.arguments}
            calls.append(ToolCall(item.call_id, item.name, arguments))
        return LlmTurn(response.id, calls, getattr(response, "output_text", "") or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/llm.py tests/test_llm.py
git commit -m "feat: add Muse Spark Responses wrapper"
```

### Task 11: Muse Spark test (tool call, image follow-up, structured output)

**Files:**
- Create: `scripts/spike_agent.py`
- Modify: `docs/spike-notes.md`

- [ ] **Step 1: Write the script**

`scripts/spike_agent.py`:
```python
"""Day-1 test: Muse Spark tool calling, an image in a follow-up user message, and structured output.

Usage: python scripts/spike_agent.py field-images/IMG_4215.JPG
"""
import json
import sys
import time
from pathlib import Path

from scout.agent.llm import Llm, user_message
from scout.app.ingest import load_image, to_data_url
from scout.config import get_settings

TOOL = {
    "type": "function",
    "name": "report_rows",
    "description": "Report how many crop rows are visible.",
    "parameters": {"type": "object", "properties": {"count": {"type": "integer"}, "notes": {"type": "string"}},
                   "required": ["count", "notes"], "additionalProperties": False},
}
SCHEMA = {
    "type": "json_schema", "name": "plants", "strict": True,
    "schema": {"type": "object", "properties": {"crop": {"type": "string"}, "weeds_visible": {"type": "boolean"}},
               "required": ["crop", "weeds_visible"], "additionalProperties": False},
}


def main(photo: str) -> None:
    llm = Llm.from_settings(get_settings())
    image = load_image(0, Path(photo).read_bytes())
    started = time.time()
    first = llm.turn([user_message("How many crop rows are visible? Use the tool.", [image.work_data_url])], tools=[TOOL])
    print(f"1) {time.time() - started:.1f}s calls={[(c.name, c.arguments) for c in first.calls]} text={first.text[:200]!r}")
    if not first.calls:
        sys.exit("No tool call came back. Check the model id and tool format before continuing.")
    outputs = [{"type": "function_call_output", "call_id": c.call_id, "output": "recorded"} for c in first.calls]
    corner = image.full.crop((0, 0, image.full.width // 2, image.full.height // 2))
    started = time.time()
    second = llm.turn(outputs + [user_message("Close-up of the top-left quarter. Describe the plants in one sentence.",
                                              [to_data_url(corner)])],
                      tools=[TOOL], previous_response_id=first.response_id)
    print(f"2) {time.time() - started:.1f}s follow-up with image: {second.text[:300]!r}")
    third = llm.turn([user_message("What crop is this and are weeds visible?", [image.work_data_url])], text_format=SCHEMA)
    print(f"3) structured output: {json.loads(third.text)}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run it**

Run: `python scripts/spike_agent.py field-images/IMG_4215.JPG`
Expected:
- Line 1 shows a `report_rows` call with an integer count.
- Line 2 describes the close-up.
- Line 3 prints a dict with `crop` and `weeds_visible`.

If line 1 has no calls, try `AGENT_MODEL=muse-spark-1.2` in `.env` and record the result.

- [ ] **Step 3: Record results in `docs/spike-notes.md` (Muse Spark section) and commit**

```bash
git add scripts/spike_agent.py docs/spike-notes.md
git commit -m "chore: Muse Spark tool-calling test and notes"
```

---

# Day 2: Sat Oct 31. Stop agent

### Task 12: Crop-row fitting

**Files:**
- Create: `scout/vision/rows.py`
- Test: `tests/test_rows.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rows.py`:
```python
from scout.vision.rows import band_mask, fit_rows


def _two_rows(jitter=False):
    left = [(100 + (i % 2 if jitter else 0), 20 + 30 * i) for i in range(5)]
    right = [(300, 20 + 30 * i) for i in range(5)]
    return left + right


def test_fits_two_vertical_rows():
    fit = fit_rows(_two_rows(jitter=True), [20] * 10)
    assert fit is not None and fit.orientation == "vertical" and len(fit.lines) == 2
    intercepts = sorted(round(b) for _, b in fit.lines)
    assert intercepts[0] in (100, 101) and intercepts[1] == 300


def test_too_few_plants_gives_no_rows():
    assert fit_rows([(10, 10), (20, 20), (30, 30)], [5, 5, 5]) is None


def test_band_mask_covers_rows_only():
    band = band_mask(fit_rows(_two_rows(), [20] * 10), (400, 200))
    assert band[50, 100] and band[50, 300] and not band[50, 200]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.rows'`

- [ ] **Step 3: Write the implementation**

`scout/vision/rows.py`:
```python
"""Fit crop rows as lines through crop-plant centers (RANSAC), and build an in-row band mask."""
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RowFit:
    orientation: Literal["vertical", "horizontal"]
    lines: list[tuple[float, float]]  # v = a*u + b; vertical rows: u=y, v=x; horizontal rows: u=x, v=y
    half_width: float


def _fit(points, tol, max_lines, min_inliers, rng):
    remaining = list(points)
    lines, total = [], 0
    while len(remaining) >= min_inliers and len(lines) < max_lines:
        best, best_line = [], None
        for _ in range(200):
            (u1, v1), (u2, v2) = rng.sample(remaining, 2)
            if abs(u2 - u1) < 1e-6:
                continue
            a = (v2 - v1) / (u2 - u1)
            b = v1 - a * u1
            inliers = [(u, v) for u, v in remaining if abs(v - (a * u + b)) <= tol]
            if len(inliers) > len(best):
                best, best_line = inliers, (a, b)
        if len(best) < min_inliers:
            break
        us = np.array([p[0] for p in best], dtype=float)
        vs = np.array([p[1] for p in best], dtype=float)
        a, b = np.polyfit(us, vs, 1) if np.ptp(us) > 0 else best_line
        lines.append((float(a), float(b)))
        total += len(best)
        taken = set(best)
        remaining = [p for p in remaining if p not in taken]
    return lines, total


def fit_rows(centroids: list[tuple[float, float]], widths: list[float], max_lines: int = 6,
             min_inliers: int = 3, seed: int = 0) -> RowFit | None:
    """Return rows only when at least two lines each hold at least `min_inliers` crop plants."""
    if len(centroids) < 2 * min_inliers or not widths:
        return None
    half = float(np.median(widths)) / 2
    tol = max(half * 1.2, 4.0)
    best = None
    for orientation in ("vertical", "horizontal"):
        points = [(y, x) if orientation == "vertical" else (x, y) for x, y in centroids]
        lines, total = _fit(points, tol, max_lines, min_inliers, random.Random(seed))
        if len(lines) >= 2 and (best is None or total > best[2]):
            best = (orientation, lines, total)
    if best is None:
        return None
    return RowFit(orientation=best[0], lines=best[1], half_width=half)


def band_mask(fit: RowFit, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width]
    u, v = (ys, xs) if fit.orientation == "vertical" else (xs, ys)
    band = np.zeros((height, width), dtype=bool)
    for a, b in fit.lines:
        band |= np.abs(v - (a * u + b)) <= fit.half_width
    return band
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rows.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/rows.py tests/test_rows.py
git commit -m "feat: fit crop rows and in-row band"
```

### Task 13: Measurements

**Files:**
- Create: `scout/vision/measure.py`
- Test: `tests/test_measure.py`

- [ ] **Step 1: Write the failing test**

`tests/test_measure.py`:
```python
import numpy as np
import pytest

from scout.models import Instance
from scout.vision.measure import compute_metrics, footprint_m2

SIZE = (100, 100)


def _inst(iid, role, box, species=None, prompt="plant", height=None):
    return Instance(id=iid, source="concept", prompt=prompt, box=box, area_px=0, role=role,
                    species_id=species, height_in=height)


def _mask(box):
    m = np.zeros((100, 100), bool)
    left, top, right, bottom = box
    m[top:bottom, left:right] = True
    return m


def _setup():
    boxes = {"c": (10, 10, 30, 30), "w1": (60, 60, 70, 70), "w2": (80, 10, 90, 20)}
    instances = [_inst("c", "crop", boxes["c"]),
                 _inst("w1", "weed", boxes["w1"], "waterhemp", height=3.0),
                 _inst("w2", "weed", boxes["w2"], None, prompt="grass")]
    return instances, {key: _mask(box) for key, box in boxes.items()}


def test_cover_and_counts_without_scale():
    instances, masks = _setup()
    m = compute_metrics(instances, masks, SIZE, "row")
    assert m.weed_cover_pct == 2.0 and m.crop_cover_pct == 4.0 and m.weed_count == 2
    assert m.species == {"waterhemp": 1, "unidentified_grass": 1}
    assert m.species_cover_pct == {"waterhemp": 1.0, "unidentified_grass": 1.0}
    assert m.density_per_m2 is None and m.species_density is None
    assert m.max_height_in == 3.0 and m.species_max_height_in == {"waterhemp": 3.0}


def test_sky_is_removed_from_ground():
    instances, masks = _setup()
    sky = np.zeros((100, 100), bool)
    sky[:50, :] = True
    m = compute_metrics(instances, masks, SIZE, "row", sky=sky)
    assert m.weed_cover_pct == 2.0  # 100 weed px on 5000 ground px
    assert m.weed_count == 1 and m.species == {"waterhemp": 1}


def test_top_down_density():
    instances, masks = _setup()
    m = compute_metrics(instances, masks, SIZE, "top_down")
    assert m.density_per_m2 == pytest.approx(0.735, abs=0.01)
    assert footprint_m2(SIZE, "top_down") == pytest.approx(2.72, abs=0.01)
    assert footprint_m2(SIZE, "row") is None


def test_in_row_share_and_approximate_count():
    instances, masks = _setup()
    band = np.zeros((100, 100), bool)
    band[:, 55:75] = True
    assert compute_metrics(instances, masks, SIZE, "row", band=band).in_row_weed_share == 0.5
    big = compute_metrics([_inst("w3", "weed", (0, 40, 50, 100))], {"w3": _mask((0, 40, 50, 100))}, SIZE, "row")
    assert big.count_approximate is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.measure'`

- [ ] **Step 3: Write the implementation**

`scout/vision/measure.py`:
```python
"""Per-stop measurements, computed only from masks and classifications (never by the model)."""
import math
from dataclasses import dataclass

import numpy as np

from scout.models import Instance

APPROX_PIECE_FRAC = 0.02  # one weed piece over 2% of the ground likely merges several plants
CAPTURE_HEIGHT_M = 1.2
PHONE_HFOV_DEG = 69.0


@dataclass
class Metrics:
    weed_cover_pct: float
    crop_cover_pct: float
    weed_count: int
    count_approximate: bool
    density_per_m2: float | None
    in_row_weed_share: float | None
    species: dict[str, int]
    species_cover_pct: dict[str, float]
    species_density: dict[str, float] | None
    species_max_height_in: dict[str, float]
    max_height_in: float | None


def footprint_m2(size: tuple[int, int], view_type: str, capture_height_m: float = CAPTURE_HEIGHT_M,
                 hfov_deg: float = PHONE_HFOV_DEG) -> float | None:
    """Ground area of a straight-down photo; None for any other view."""
    if view_type != "top_down":
        return None
    width, height = size
    ground_width = 2 * capture_height_m * math.tan(math.radians(hfov_deg / 2))
    return ground_width * ground_width * height / width


def species_key(instance: Instance) -> str:
    if instance.species_id:
        return instance.species_id
    return "unidentified_grass" if "grass" in instance.prompt.lower() else "unidentified_broadleaf"


def compute_metrics(instances: list[Instance], masks: dict[str, np.ndarray], size: tuple[int, int],
                    view_type: str, sky: np.ndarray | None = None, band: np.ndarray | None = None) -> Metrics:
    width, height = size
    ground = ~sky if sky is not None else np.ones((height, width), dtype=bool)
    ground_px = max(int(ground.sum()), 1)
    crop = np.zeros((height, width), dtype=bool)
    for inst in instances:
        if inst.role == "crop":
            crop |= masks[inst.id]
    crop &= ground
    weed_mask = np.zeros((height, width), dtype=bool)
    species_masks: dict[str, np.ndarray] = {}
    species_counts: dict[str, int] = {}
    species_heights: dict[str, float] = {}
    heights: list[float] = []
    approximate = False
    for inst in instances:
        if inst.role != "weed":
            continue
        mask = masks[inst.id] & ground & ~crop
        if not mask.any():
            continue
        key = species_key(inst)
        weed_mask |= mask
        species_masks[key] = species_masks.get(key, np.zeros_like(mask)) | mask
        species_counts[key] = species_counts.get(key, 0) + 1
        if inst.height_in is not None:
            heights.append(inst.height_in)
            species_heights[key] = max(species_heights.get(key, 0.0), inst.height_in)
        if mask.sum() > APPROX_PIECE_FRAC * ground_px:
            approximate = True
    weed_px = int(weed_mask.sum())
    count = sum(species_counts.values())
    area = footprint_m2(size, view_type)
    return Metrics(
        weed_cover_pct=round(100 * weed_px / ground_px, 2),
        crop_cover_pct=round(100 * int(crop.sum()) / ground_px, 2),
        weed_count=count,
        count_approximate=approximate,
        density_per_m2=round(count / area, 2) if area else None,
        in_row_weed_share=round(float((weed_mask & band).sum()) / weed_px, 3) if band is not None and weed_px else None,
        species=species_counts,
        species_cover_pct={key: round(100 * int(m.sum()) / ground_px, 2) for key, m in species_masks.items()},
        species_density={key: round(c / area, 2) for key, c in species_counts.items()} if area else None,
        species_max_height_in=species_heights,
        max_height_in=max(heights) if heights else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/vision/measure.py tests/test_measure.py
git commit -m "feat: compute weed cover, counts, density, and in-row share"
```

### Task 14: Weed book

**Files:**
- Create: `scout/agronomy/weeds.yaml`, `scout/agronomy/weeds.py`
- Test: `tests/test_weeds.py`

- [ ] **Step 1: Write the failing test**

`tests/test_weeds.py`:
```python
from scout.agronomy.weeds import WeedBook


def test_default_book_loads_all_species():
    book = WeedBook.load()
    assert len(book.ids()) == 35
    assert "unidentified_broadleaf" in book and "unidentified_grass" in book


def test_resolve_names_and_synonyms():
    book = WeedBook.load()
    assert book.resolve("Marestail") == "horseweed"
    assert book.resolve("Amaranthus tuberculatus") == "waterhemp"
    assert book.resolve("giant foxtail") == "giant_foxtail"
    assert book.resolve("not a weed") is None
    assert book.resolve("") is None


def test_shipped_thresholds_have_sources():
    assert WeedBook.load().validate() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_weeds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agronomy.weeds'`

- [ ] **Step 3: Write the data file**

`scout/agronomy/weeds.yaml`. Thresholds start empty; Task 19 fills them with sources.
```yaml
waterhemp: {common_name: waterhemp, scientific_name: Amaranthus tuberculatus, type: broadleaf, synonyms: [common waterhemp, tall waterhemp]}
palmer_amaranth: {common_name: Palmer amaranth, scientific_name: Amaranthus palmeri, type: broadleaf, synonyms: [palmer pigweed, careless weed]}
redroot_pigweed: {common_name: redroot pigweed, scientific_name: Amaranthus retroflexus, type: broadleaf, synonyms: [redroot, rough pigweed, pigweed]}
common_lambsquarters: {common_name: common lambsquarters, scientific_name: Chenopodium album, type: broadleaf, synonyms: [lambsquarters, lamb's-quarters]}
giant_ragweed: {common_name: giant ragweed, scientific_name: Ambrosia trifida, type: broadleaf, synonyms: [buffaloweed, great ragweed]}
common_ragweed: {common_name: common ragweed, scientific_name: Ambrosia artemisiifolia, type: broadleaf, synonyms: [annual ragweed, short ragweed]}
velvetleaf: {common_name: velvetleaf, scientific_name: Abutilon theophrasti, type: broadleaf, synonyms: [buttonweed, butterprint]}
horseweed: {common_name: horseweed, scientific_name: Conyza canadensis, type: broadleaf, synonyms: [marestail, Erigeron canadensis]}
kochia: {common_name: kochia, scientific_name: Bassia scoparia, type: broadleaf, synonyms: [fireweed, burningbush, Kochia scoparia]}
common_cocklebur: {common_name: common cocklebur, scientific_name: Xanthium strumarium, type: broadleaf, synonyms: [cocklebur]}
pennsylvania_smartweed: {common_name: Pennsylvania smartweed, scientific_name: Persicaria pensylvanica, type: broadleaf, synonyms: [smartweed, Polygonum pensylvanicum]}
morningglory: {common_name: morningglory, scientific_name: Ipomoea spp., type: broadleaf, synonyms: [ivyleaf morningglory, tall morningglory, entireleaf morningglory, morning glory]}
jimsonweed: {common_name: jimsonweed, scientific_name: Datura stramonium, type: broadleaf, synonyms: [jimson weed, thornapple]}
eastern_black_nightshade: {common_name: eastern black nightshade, scientific_name: Solanum ptycanthum, type: broadleaf, synonyms: [black nightshade, nightshade]}
giant_foxtail: {common_name: giant foxtail, scientific_name: Setaria faberi, type: grass, synonyms: [foxtail]}
green_foxtail: {common_name: green foxtail, scientific_name: Setaria viridis, type: grass, synonyms: []}
yellow_foxtail: {common_name: yellow foxtail, scientific_name: Setaria pumila, type: grass, synonyms: [Setaria glauca]}
barnyardgrass: {common_name: barnyardgrass, scientific_name: Echinochloa crus-galli, type: grass, synonyms: [barnyard grass]}
large_crabgrass: {common_name: large crabgrass, scientific_name: Digitaria sanguinalis, type: grass, synonyms: [crabgrass]}
fall_panicum: {common_name: fall panicum, scientific_name: Panicum dichotomiflorum, type: grass, synonyms: []}
shattercane: {common_name: shattercane, scientific_name: Sorghum bicolor, type: grass, synonyms: [wild cane]}
woolly_cupgrass: {common_name: woolly cupgrass, scientific_name: Eriochloa villosa, type: grass, synonyms: []}
volunteer_corn: {common_name: volunteer corn, scientific_name: Zea mays, type: grass, synonyms: []}
canada_thistle: {common_name: Canada thistle, scientific_name: Cirsium arvense, type: broadleaf, synonyms: [creeping thistle]}
dandelion: {common_name: dandelion, scientific_name: Taraxacum officinale, type: broadleaf, synonyms: [common dandelion]}
common_milkweed: {common_name: common milkweed, scientific_name: Asclepias syriaca, type: broadleaf, synonyms: [milkweed]}
wild_mustard: {common_name: wild mustard, scientific_name: Sinapis arvensis, type: broadleaf, synonyms: [charlock, mustard]}
field_pennycress: {common_name: field pennycress, scientific_name: Thlaspi arvense, type: broadleaf, synonyms: [pennycress, fanweed]}
henbit: {common_name: henbit, scientific_name: Lamium amplexicaule, type: broadleaf, synonyms: []}
purple_deadnettle: {common_name: purple deadnettle, scientific_name: Lamium purpureum, type: broadleaf, synonyms: [red deadnettle]}
common_chickweed: {common_name: common chickweed, scientific_name: Stellaria media, type: broadleaf, synonyms: [chickweed]}
yellow_nutsedge: {common_name: yellow nutsedge, scientific_name: Cyperus esculentus, type: sedge, synonyms: [nutsedge, nutgrass]}
quackgrass: {common_name: quackgrass, scientific_name: Elymus repens, type: grass, synonyms: [quack grass, couchgrass]}
unidentified_broadleaf: {common_name: unidentified broadleaf, scientific_name: "", type: broadleaf, synonyms: []}
unidentified_grass: {common_name: unidentified grass, scientific_name: "", type: grass, synonyms: []}
```

- [ ] **Step 4: Write the loader**

`scout/agronomy/weeds.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_weeds.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add scout/agronomy/weeds.yaml scout/agronomy/weeds.py tests/test_weeds.py
git commit -m "feat: add weed book with names and synonyms"
```

### Task 15: Stop state

**Files:**
- Create: `scout/agent/state.py`
- Modify: `tests/helpers.py` (append)
- Test: `tests/test_state.py`

- [ ] **Step 1: Append helpers and write the failing test**

Append to `tests/helpers.py`:
```python
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.retry import BadRequest
from scout.vision.sam_parse import SamRecord


def square_record(object_id, left, top, size, confidence=0.9):
    return SamRecord(object_id, (left, top, left + size, top + size), confidence, np.ones((size, size), bool))


def field_scene():
    """Concept answers for a 100x100 photo: one soybean plant, two weeds, empty sky."""
    return {
        "soybean plant": [square_record("0", 10, 10, 20)],
        "plant": [square_record("0", 10, 10, 20), square_record("1", 60, 60, 10), square_record("2", 80, 10, 10)],
        "sky": [],
    }


class FakeSam:
    def __init__(self, concepts=None, at=None, fail=None):
        self.concepts = concepts or {}
        self.at = at or []
        self.fail = fail or set()
        self.calls = []

    def segment_concept(self, image, phrase):
        self.calls.append(("concept", phrase))
        if phrase in self.fail:
            raise BadRequest(f"rejected {phrase}")
        return list(self.concepts.get(phrase, []))

    def segment_at(self, image, box=None, points=()):
        self.calls.append(("at", box, tuple(points)))
        return list(self.at)


def small_weedbook():
    return WeedBook({
        "waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="Amaranthus tuberculatus",
                              type="broadleaf", synonyms=["common waterhemp"]),
        "horseweed": WeedInfo(id="horseweed", common_name="horseweed", scientific_name="Conyza canadensis",
                              type="broadleaf", synonyms=["marestail"]),
        "giant_foxtail": WeedInfo(id="giant_foxtail", common_name="giant foxtail", scientific_name="Setaria faberi",
                                  type="grass"),
        "unidentified_broadleaf": WeedInfo(id="unidentified_broadleaf", common_name="unidentified broadleaf",
                                           scientific_name="", type="broadleaf"),
        "unidentified_grass": WeedInfo(id="unidentified_grass", common_name="unidentified grass",
                                       scientific_name="", type="grass"),
    })
```

`tests/test_state.py`:
```python
from scout.agent.state import StopState
from tests.helpers import make_image, square_record


def test_candidates_are_plants_minus_crop():
    state = StopState(make_image(100, 100), "soybean")
    state.add_records([square_record("0", 10, 10, 20)], "concept", "soybean plant", role="crop", kind="crop")
    state.add_records([square_record("0", 10, 10, 20), square_record("1", 60, 60, 10), square_record("2", 80, 10, 10)],
                      "concept", "plant", role="ignore", kind="all_plants")
    candidates = state.refresh_candidates()
    assert [c.box for c in candidates] == [(80, 10, 90, 20), (60, 60, 70, 70)]
    assert [c.id for c in candidates] == ["s0-i5", "s0-i6"]


def test_refresh_keeps_classified_and_replaces_unclassified():
    state = StopState(make_image(100, 100), "soybean")
    state.add_records([square_record("0", 60, 60, 10), square_record("1", 80, 10, 10)], "concept", "plant",
                      role="ignore", kind="all_plants")
    first = state.refresh_candidates()
    first[0].role = "weed"
    second = state.refresh_candidates()
    assert len(second) == 1 and second[0].box == first[1].box
    assert first[0].id in state.instances and first[1].id not in state.instances


def test_log_reports_steps():
    seen = []
    state = StopState(make_image(), "corn", on_step=seen.append, clock=lambda: 0.0)
    state.log("segment", 'Segment "corn plant": 3 found', ["s0-i1"])
    assert seen[0].summary.startswith("Segment") and seen[0].t_ms == 0


def test_overlay_renders_without_instances():
    assert StopState(make_image(40, 30), "corn").overlay().size == (40, 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.state'`

- [ ] **Step 3: Write the implementation**

`scout/agent/state.py`:
```python
"""Everything known about one stop while the agent works on it."""
import time
from typing import Callable

import numpy as np
from PIL import Image

from scout.app.ingest import StopImage
from scout.models import AgentStep, Instance
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.rows import band_mask, fit_rows
from scout.vision.sam_parse import SamRecord

CANDIDATE_MIN_AREA_FRAC = 0.0005
LOW_SAM_CONFIDENCE = 0.5


class StopFailed(Exception):
    """The stop cannot continue (e.g. SAM rejected the same tool twice)."""


class StopState:
    def __init__(self, image: StopImage, crop: str, on_step: Callable[[AgentStep], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.image = image
        self.crop = crop
        self.instances: dict[str, Instance] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.kinds: dict[str, str] = {}
        self.sky: np.ndarray | None = None
        self.view_type: str | None = None
        self.crop_stage: str | None = None
        self.notes = ""
        self.steps: list[AgentStep] = []
        self.calls = 0
        self.bad_requests: dict[str, int] = {}
        self.finished = False
        self._next = 0
        self._on_step = on_step
        self._clock = clock
        self.started = clock()

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def log(self, tool: str, summary: str, instance_ids=(), ok: bool = True) -> AgentStep:
        step = AgentStep(t_ms=int((self._clock() - self.started) * 1000), tool=tool, summary=summary,
                         instance_ids=list(instance_ids), ok=ok)
        self.steps.append(step)
        if self._on_step:
            self._on_step(step)
        return step

    def add_mask(self, mask: np.ndarray, source: str, prompt: str, role: str = "unknown",
                 confidence: float | None = None, kind: str | None = None) -> Instance:
        self._next += 1
        inst = Instance(id=f"s{self.image.index}-i{self._next}", source=source, prompt=prompt, box=mk.bbox(mask),
                        sam_confidence=confidence, area_px=int(mask.sum()), role=role,
                        check_this=confidence is not None and confidence < LOW_SAM_CONFIDENCE)
        self.instances[inst.id] = inst
        self.masks[inst.id] = mask
        if kind:
            self.kinds[inst.id] = kind
        return inst

    def add_records(self, records: list[SamRecord], source: str, prompt: str, role: str = "unknown",
                    kind: str | None = None) -> list[Instance]:
        return [self.add_mask(r.full_mask(self.size), source, prompt, role, r.confidence, kind) for r in records]

    def union_of(self, predicate: Callable[[Instance], bool]) -> np.ndarray:
        return mk.union([self.masks[i.id] for i in self.instances.values() if predicate(i)], self.size)

    def refresh_candidates(self) -> list[Instance]:
        """Weed candidates = all-plant masks minus crop and already-classified weeds, split into pieces."""
        pool = self.union_of(lambda i: self.kinds.get(i.id) == "all_plants")
        if not pool.any():
            return []
        for iid in [i.id for i in self.instances.values() if i.source == "candidate" and i.role == "unknown"]:
            del self.instances[iid]
            del self.masks[iid]
        taken = self.union_of(lambda i: i.role in ("crop", "weed"))
        min_area = CANDIDATE_MIN_AREA_FRAC * self.size[0] * self.size[1]
        return [self.add_mask(piece, "candidate", "plant minus crop") for piece in mk.pieces(pool & ~taken, min_area)]

    def row_band(self) -> np.ndarray | None:
        crops = [i for i in self.instances.values() if i.role == "crop"]
        centers = [((i.box[0] + i.box[2]) / 2, (i.box[1] + i.box[3]) / 2) for i in crops]
        widths = [((i.box[2] - i.box[0]) + (i.box[3] - i.box[1])) / 2 for i in crops]
        fit = fit_rows(centers, widths)
        return band_mask(fit, self.size) if fit else None

    def overlay(self) -> Image.Image:
        crop = self.union_of(lambda i: i.role == "crop")
        weed = self.union_of(lambda i: i.role == "weed") & ~crop
        flagged = [i.box for i in self.instances.values() if i.role == "weed" and i.check_this]
        return render_overlay(self.image.work, crop, weed, self.row_band(), flagged)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/state.py tests/helpers.py tests/test_state.py
git commit -m "feat: add stop state with weed candidates"
```

### Task 16: Agent tools

**Files:**
- Create: `scout/agent/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import json

import pytest

from scout.agent.state import StopFailed, StopState
from scout.agent.tools import TOOL_SCHEMAS, ToolRunner
from tests.helpers import FakeSam, field_scene, make_image, small_weedbook


def _runner(sam=None, max_calls=12):
    state = StopState(make_image(100, 100), "soybean")
    return ToolRunner(state, sam or FakeSam(field_scene()), small_weedbook(), max_calls=max_calls), state


def _segment_both(runner):
    runner.run("segment", {"phrase": "soybean plant", "kind": "crop"})
    return runner.run("segment", {"phrase": "plant", "kind": "all_plants"})


def test_schemas_name_six_tools():
    assert [t["name"] for t in TOOL_SCHEMAS] == ["segment", "segment_at", "zoom", "classify", "measure", "finish"]


def test_segment_crop_then_plants_gives_candidates():
    runner, state = _runner()
    out = json.loads(_segment_both(runner).output)
    assert out["total_found"] == 3
    assert [c["id"] for c in out["weed_candidates"]] == ["s0-i5", "s0-i6"]
    assert state.steps[-1].summary == 'Segment "plant": 3 found, 2 weed candidates'


def test_classify_resolves_synonyms_and_rejects_unknown():
    runner, state = _runner()
    _segment_both(runner)
    ok = runner.run("classify", {"items": [{"instance_id": "s0-i6", "role": "weed", "species_id": "Marestail",
                                            "height_in": 2.5, "leaf_count": 6, "id_confidence": "low"}]})
    assert json.loads(ok.output) == {"ok": ["s0-i6"]}
    inst = state.instances["s0-i6"]
    assert inst.species_id == "horseweed" and inst.check_this is True
    bad = runner.run("classify", {"items": [{"instance_id": "s0-i5", "role": "weed", "species_id": "dragonweed",
                                             "id_confidence": "high"}]})
    assert bad.output.startswith("error:") and state.instances["s0-i5"].role == "unknown"


def test_zoom_returns_image():
    runner, _ = _runner()
    runner.run("segment", {"phrase": "soybean plant", "kind": "crop"})
    result = runner.run("zoom", {"instance_id": "s0-i1"})
    assert result.images and result.images[0][1].startswith("data:image/jpeg;base64,")


def test_segment_at_needs_box_or_points():
    runner, _ = _runner()
    assert runner.run("segment_at", {"kind": "weed"}).output.startswith("error:")


def test_unknown_tool_and_bad_arguments():
    runner, _ = _runner()
    assert runner.run("explode", {}).output.startswith("error:")
    assert runner.run("zoom", {"__invalid__": "{"}).output.startswith("error:")
    assert runner.run("measure", {"view": "row"}).output.startswith("error:")


def test_measure_then_finish():
    runner, state = _runner()
    _segment_both(runner)
    refused = runner.run("finish", {"crop_stage": None, "notes": ""})
    assert refused.output.startswith("error:") and not refused.finished
    runner.run("classify", {"items": [{"instance_id": i, "role": "weed", "species_id": None, "height_in": None,
                                       "leaf_count": None, "id_confidence": "medium"} for i in ("s0-i5", "s0-i6")]})
    metrics = json.loads(runner.run("measure", {"view_type": "row"}).output)
    assert metrics["weed_count"] == 2
    assert runner.run("finish", {"crop_stage": "V3", "notes": "ok"}).finished
    assert state.crop_stage == "V3" and state.finished


def test_two_bad_requests_fail_the_stop():
    runner, _ = _runner(FakeSam(field_scene(), fail={"plant"}))
    assert runner.run("segment", {"phrase": "plant", "kind": "all_plants"}).output.startswith("error:")
    with pytest.raises(StopFailed):
        runner.run("segment", {"phrase": "plant", "kind": "all_plants"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.tools'`

- [ ] **Step 3: Write the implementation**

`scout/agent/tools.py`:
```python
"""The six tools Muse Spark can call at a stop, and the code that runs them."""
import json
from dataclasses import asdict, dataclass, field

from scout.agent.state import StopFailed, StopState
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import to_data_url
from scout.models import Instance
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.measure import compute_metrics

VIEW_TYPES = ("row", "top_down", "close_up", "wide")
SEGMENT_ROLES = {"crop": "crop", "all_plants": "ignore", "weed": "weed"}
SEGMENT_AT_ROLES = {"crop": "crop", "weed": "weed", "unknown": "unknown"}
ZOOM_MAX_PX = 768


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}


_BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4,
        "description": "[left, top, right, bottom] in photo pixels"}

TOOL_SCHEMAS = [
    _fn("segment", "Ask SAM 3.1 for every instance of one concept. Use a short noun phrase naming one thing.",
        {"phrase": {"type": "string"},
         "kind": {"type": "string", "enum": ["crop", "all_plants", "weed"],
                  "description": "crop: the crop plants. all_plants: every plant; weed candidates are derived "
                                 "from it minus crop. weed: a phrase you know names weeds, e.g. grass in soybean."}},
        ["phrase", "kind"]),
    _fn("segment_at", "Segment one plant you can point at, with a box or with points.",
        {"box": _BOX,
         "points": {"type": "array", "items": {"type": "object", "properties": {
             "x": {"type": "integer"}, "y": {"type": "integer"}, "positive": {"type": "boolean"}},
             "required": ["x", "y", "positive"], "additionalProperties": False}},
         "kind": {"type": "string", "enum": ["crop", "weed", "unknown"]}},
        ["kind"]),
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
        except (TypeError, ValueError) as error:
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
        out = {"id": inst.id, "box": list(inst.box), "area_pct": round(100 * inst.area_px / (width * height), 2)}
        if inst.sam_confidence is not None:
            out["conf"] = round(inst.sam_confidence, 2)
        return out

    def _tool_segment(self, phrase: str, kind: str) -> ToolResult:
        if kind not in SEGMENT_ROLES:
            raise ValueError(f"kind must be one of {sorted(SEGMENT_ROLES)}")
        records = self.sam.segment_concept(self.state.image, phrase)
        added = self.state.add_records(records, "concept", phrase.strip(), role=SEGMENT_ROLES[kind], kind=kind)
        candidates = self.state.refresh_candidates() if kind in ("crop", "all_plants") else []
        summary = f'Segment "{phrase.strip()}": {len(added)} found'
        if candidates:
            summary += f", {len(candidates)} weed candidates"
        self.state.log("segment", summary, [i.id for i in added])
        return ToolResult(json.dumps({"total_found": len(added), "found": [self._brief(i) for i in added[:30]],
                                      "weed_candidates": [self._brief(i) for i in candidates[:30]]}))

    def _tool_segment_at(self, kind: str, box: list[int] | None = None, points: list[dict] | None = None) -> ToolResult:
        if kind not in SEGMENT_AT_ROLES:
            raise ValueError(f"kind must be one of {sorted(SEGMENT_AT_ROLES)}")
        if not box and not points:
            raise ValueError("give a box or at least one point")
        point_list = [(p["x"], p["y"], p["positive"]) for p in points or []]
        records = self.sam.segment_at(self.state.image, box=tuple(box) if box else None, points=point_list)
        added = self.state.add_records(records[:1], "prompt", "box" if box else "point",
                                       role=SEGMENT_AT_ROLES[kind], kind=kind)
        if kind == "crop":
            self.state.refresh_candidates()
        self.state.log("segment_at", f"Segment at a {'box' if box else 'point'}: {len(added)} found",
                       [i.id for i in added])
        return ToolResult(json.dumps({"found": [self._brief(i) for i in added]}))

    def _tool_zoom(self, instance_id: str | None = None, box: list[int] | None = None) -> ToolResult:
        if instance_id:
            if instance_id not in self.state.instances:
                raise ValueError(f"unknown instance {instance_id}")
            region = self.state.instances[instance_id].box
        elif box:
            region = tuple(box)
        else:
            raise ValueError("give instance_id or box")
        left, top, right, bottom = mk.pad_box(region, 0.15, self.state.size)
        if right - left < 2 or bottom - top < 2:
            raise ValueError("region is too small to zoom")
        scale = self.state.image.full.width / self.state.image.work.width
        crop = self.state.image.full.crop(tuple(int(v * scale) for v in (left, top, right, bottom)))
        crop.thumbnail((ZOOM_MAX_PX, ZOOM_MAX_PX))
        label = instance_id or f"box {list(region)}"
        self.state.log("zoom", f"Zoom on {label}", [instance_id] if instance_id else [])
        return ToolResult(f"The close-up of {label} follows in the next message.",
                          images=[(f"Close-up of {label}", to_data_url(crop))])

    def _tool_classify(self, items: list[dict]) -> ToolResult:
        resolved = []
        for item in items:
            inst = self.state.instances.get(item["instance_id"])
            if inst is None:
                raise ValueError(f"unknown instance {item['instance_id']}")
            if item["role"] not in ("crop", "weed", "ignore"):
                raise ValueError(f"bad role {item['role']}")
            species = item.get("species_id")
            if species is not None and species not in self.weeds:
                species = self.weeds.resolve(species)
                if species is None:
                    raise ValueError(f"unknown species_id {item['species_id']}; use an id from the allowed list or null")
            resolved.append((inst, item, species))
        for inst, item, species in resolved:
            inst.role = item["role"]
            inst.species_id = species if item["role"] == "weed" else None
            inst.height_in = item.get("height_in")
            inst.leaf_count = item.get("leaf_count")
            inst.id_confidence = item["id_confidence"]
            inst.check_this = inst.check_this or item["id_confidence"] == "low"
        weeds = [inst for inst, _, _ in resolved if inst.role == "weed"]
        if len(resolved) == 1 and weeds:
            inst = weeds[0]
            name = self.weeds.get(inst.species_id).common_name if inst.species_id else "unidentified weed"
            details = [f"~{inst.height_in:g} in" if inst.height_in is not None else "",
                       f"{inst.leaf_count} leaves" if inst.leaf_count is not None else ""]
            summary = "Identified: " + ", ".join([name] + [d for d in details if d])
        else:
            summary = f"Classified {len(resolved)} plants ({len(weeds)} weeds)"
        self.state.log("classify", summary, [inst.id for inst, _, _ in resolved])
        return ToolResult(json.dumps({"ok": [inst.id for inst, _, _ in resolved]}))

    def _tool_measure(self, view_type: str) -> ToolResult:
        if view_type not in VIEW_TYPES:
            raise ValueError(f"view_type must be one of {list(VIEW_TYPES)}")
        self.state.view_type = view_type
        if view_type in ("row", "wide") and self.state.sky is None:
            records = self.sam.segment_concept(self.state.image, "sky")
            self.state.sky = mk.union([r.full_mask(self.state.size) for r in records], self.state.size)
        metrics = compute_metrics(list(self.state.instances.values()), self.state.masks, self.state.size,
                                  view_type, self.state.sky, self.state.row_band())
        self.state.log("measure", f"Measure: weed cover {metrics.weed_cover_pct}%, {metrics.weed_count} weeds")
        return ToolResult(json.dumps(asdict(metrics)))

    def _tool_finish(self, crop_stage: str | None, notes: str) -> ToolResult:
        unknown = [i.id for i in self.state.instances.values() if i.role == "unknown"]
        if unknown and self.state.calls < self.max_calls:
            self.state.log("finish", "Refused: unclassified candidates remain", ok=False)
            return ToolResult(f"error: classify these first: {unknown[:20]}")
        if self.state.view_type is None:
            self.state.log("finish", "Refused: measure has not run", ok=False)
            return ToolResult("error: call measure first")
        self.state.crop_stage = crop_stage
        self.state.notes = notes
        self.state.finished = True
        self.state.log("finish", "Stop closed")
        return ToolResult("ok", finished=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tools.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/tools.py tests/test_tools.py
git commit -m "feat: add the six stop-agent tools"
```

### Task 17: Prompts and the stop agent loop

**Files:**
- Create: `scout/agent/prompts.py`, `scout/agent/stop_agent.py`
- Modify: `tests/helpers.py` (append)
- Test: `tests/test_stop_agent.py`

- [ ] **Step 1: Append the fake LLM and write the failing test**

Append to `tests/helpers.py`:
```python
from scout.agent.llm import LlmTurn


class FakeLlm:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def turn(self, input, **kwargs):
        self.requests.append({"input": input, **kwargs})
        return self.turns.pop(0) if self.turns else LlmTurn(f"r-end{len(self.requests)}", [], "")
```

`tests/test_stop_agent.py`:
```python
from dataclasses import replace

from scout.agent.llm import LlmTurn, ToolCall
from scout.agent.stop_agent import run_stop
from scout.config import Settings
from tests.helpers import FakeLlm, FakeSam, field_scene, make_image, small_weedbook

SETTINGS = Settings(api_key="test")
SEGMENT_BOTH = [ToolCall("c1", "segment", {"phrase": "soybean plant", "kind": "crop"}),
                ToolCall("c2", "segment", {"phrase": "plant", "kind": "all_plants"})]


def test_full_stop():
    llm = FakeLlm([
        LlmTurn("r1", SEGMENT_BOTH, ""),
        LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i6"})], ""),
        LlmTurn("r3", [ToolCall("c4", "classify", {"items": [
            {"instance_id": "s0-i5", "role": "weed", "species_id": "waterhemp", "height_in": 3, "leaf_count": 5,
             "id_confidence": "high"},
            {"instance_id": "s0-i6", "role": "weed", "species_id": "marestail", "height_in": None, "leaf_count": None,
             "id_confidence": "low"}]})], ""),
        LlmTurn("r4", [ToolCall("c5", "measure", {"view_type": "top_down"}),
                       ToolCall("c6", "finish", {"crop_stage": "V3", "notes": "Two weeds between rows."})], ""),
    ])
    steps = []
    result, _ = run_stop(make_image(100, 100), "soybean", llm, FakeSam(field_scene()), small_weedbook(), SETTINGS,
                         on_step=steps.append)
    assert result.status == "ok" and result.weed_count == 2
    assert result.species == {"waterhemp": 1, "horseweed": 1}
    assert result.crop_stage == "V3" and result.view_type == "top_down"
    assert result.density_per_m2 is not None
    assert any("s0-i6" in flag for flag in result.flags)
    assert len(steps) == len(result.steps) >= 6
    third = llm.requests[2]
    assert third["previous_response_id"] == "r2"
    assert any(part.get("type") == "message" for part in third["input"])  # zoom image rides in a user message


def test_call_limit_closes_stop_and_flags_leftovers():
    llm = FakeLlm([LlmTurn("r1", SEGMENT_BOTH, ""),
                   LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i5"})], "")])
    result, _ = run_stop(make_image(100, 100), "soybean", llm, FakeSam(field_scene()), small_weedbook(),
                         replace(SETTINGS, max_tool_calls=3))
    assert result.status == "ok" and result.weed_count == 2
    assert any(step.tool == "limit" for step in result.steps)
    assert len([f for f in result.flags if f.startswith("Check")]) == 2


def test_repeated_bad_request_fails_stop():
    call = [ToolCall("c1", "segment", {"phrase": "plant", "kind": "all_plants"})]
    llm = FakeLlm([LlmTurn("r1", call, ""), LlmTurn("r2", call, "")])
    result, _ = run_stop(make_image(100, 100), "soybean", llm, FakeSam(field_scene(), fail={"plant"}),
                         small_weedbook(), SETTINGS)
    assert result.status == "failed"


def test_prose_reply_is_nudged_then_closed():
    llm = FakeLlm([LlmTurn("r1", [], "I think there are weeds."), LlmTurn("r2", [], "Still prose.")])
    result, _ = run_stop(make_image(100, 100), "soybean", llm, FakeSam(field_scene()), small_weedbook(), SETTINGS)
    assert result.status == "no_crop_found" and len(llm.requests) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stop_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.stop_agent'`

- [ ] **Step 3: Write the prompts**

`scout/agent/prompts.py`:
```python
"""Instructions for the stop agent and the field explanation."""

CROP_PHRASES = {"corn": "corn plant", "soybean": "soybean plant", "alfalfa": "alfalfa plant"}


def stop_instructions(crop: str, size: tuple[int, int], species_ids: list[str], max_calls: int) -> str:
    width, height = size
    return f"""You are an agronomist scouting a {crop} field. You see one photo ({width}x{height} pixels; every \
coordinate you give is in these pixels). Find every weed, identify it, then have code measure it.

Procedure:
1. segment the crop with kind "crop" (for example "{CROP_PHRASES[crop]}"). If nothing is found, try one other \
phrase such as "{crop} seedling".
2. segment "plant" with kind "all_plants". Pieces of plant that are not crop come back as weed_candidates.
3. Review the candidates. Zoom on any you are unsure of. If you see a weed that no candidate covers, call \
segment_at with a point on it and kind "weed". If a whole class of weeds is present (for example grass in \
soybean), segment that phrase with kind "weed".
4. classify every candidate, batching many items into one call. Use a species_id from this list only: \
{", ".join(species_ids)}. Use null when you cannot tell. Give height_in and leaf_count when visible. Use role \
"crop" for crop pieces and "ignore" for soil, residue, or shadows.
5. measure with the view_type that fits: row (looking along rows), top_down (camera pointing straight down), \
close_up (a few plants fill the frame), wide (field to the horizon).
6. finish with the crop growth stage if you can tell (for example V3) and one sentence of notes.

Rules: never invent numbers; measurements come only from measure. You have at most {max_calls} tool calls. \
Always respond with tool calls, not prose."""


FIELD_INSTRUCTIONS = """You are an agronomist writing the reasoning for a field scouting report. You receive JSON \
with the species summary, the rule trace that produced the decision, and label-backed product options. Write at \
most 120 words for a farmer: what was found, why the decision follows, and what to weigh among the options (for \
example rotating herbicide groups). Use only numbers that appear in the JSON. Do not recommend a rate. End with: \
"Read the label before use." """
```

- [ ] **Step 4: Write the stop agent**

`scout/agent/stop_agent.py`:
```python
"""Run Muse Spark's tool loop for one stop, then turn the state into a StopResult."""
import time
from typing import Callable

from scout.agent.llm import Llm, user_message
from scout.agent.prompts import stop_instructions
from scout.agent.state import StopFailed, StopState
from scout.agent.tools import TOOL_SCHEMAS, ToolRunner
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.models import AgentStep, StopResult
from scout.vision import masks as mk
from scout.vision.measure import compute_metrics


def run_stop(image: StopImage, crop: str, llm: Llm, sam, weeds: WeedBook, settings: Settings,
             on_step: Callable[[AgentStep], None] | None = None,
             clock: Callable[[], float] = time.monotonic) -> tuple[StopResult, StopState]:
    state = StopState(image, crop, on_step=on_step, clock=clock)
    runner = ToolRunner(state, sam, weeds, max_calls=settings.max_tool_calls)
    instructions = stop_instructions(crop, image.size, weeds.ids(), settings.max_tool_calls)
    width, height = image.size
    pending: list[dict] = [user_message(f"Stop {image.index + 1}: the photo is {width}x{height} pixels.",
                                        [image.work_data_url])]
    previous = None
    nudged = False
    try:
        while not state.finished:
            if state.calls >= settings.max_tool_calls or clock() - state.started > settings.stop_timeout_s:
                state.log("limit", "Tool-call or time limit reached; closing the stop", ok=False)
                break
            turn = llm.turn(pending, instructions=instructions, tools=TOOL_SCHEMAS, previous_response_id=previous)
            previous = turn.response_id
            if not turn.calls:
                if nudged:
                    break
                nudged = True
                pending = [user_message("Call a tool. If you are done, call measure and then finish.")]
                continue
            pending, images = [], []
            for call in turn.calls:
                state.calls += 1
                result = runner.run(call.name, call.arguments)
                pending.append({"type": "function_call_output", "call_id": call.call_id, "output": result.output})
                images += result.images
            if images:
                pending.append(user_message("Close-ups you asked for: " + "; ".join(c for c, _ in images),
                                            [url for _, url in images]))
    except StopFailed as error:
        state.log("error", str(error), ok=False)
        return StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, steps=state.steps,
                          flags=["This stop could not be analyzed"], status="failed"), state
    return finalize(state, sam), state


def finalize(state: StopState, sam) -> StopResult:
    """Unclassified candidates count as unidentified weeds and are flagged; measurements are recomputed."""
    for inst in state.instances.values():
        if inst.role == "unknown":
            inst.role = "weed"
            inst.check_this = True
    view = state.view_type or "row"
    if view in ("row", "wide") and state.sky is None:
        try:
            records = sam.segment_concept(state.image, "sky")
            state.sky = mk.union([r.full_mask(state.size) for r in records], state.size)
        except Exception:  # sky removal is best effort; cover is then measured over the whole photo
            state.sky = None
    m = compute_metrics(list(state.instances.values()), state.masks, state.size, view, state.sky, state.row_band())
    has_crop = any(i.role == "crop" for i in state.instances.values())
    flags = [f"Check {i.id}: low confidence" for i in state.instances.values() if i.role == "weed" and i.check_this]
    if m.count_approximate:
        flags.append("Weed count is approximate: some weeds merged into patches")
    return StopResult(
        stop_index=state.image.index, image_hash=state.image.hash, view_type=view, gps=state.image.gps,
        crop_stage=state.crop_stage, weed_cover_pct=m.weed_cover_pct, crop_cover_pct=m.crop_cover_pct,
        weed_count=m.weed_count, count_approximate=m.count_approximate, density_per_m2=m.density_per_m2,
        in_row_weed_share=m.in_row_weed_share, species=m.species, species_cover_pct=m.species_cover_pct,
        species_density=m.species_density, species_max_height_in=m.species_max_height_in,
        max_height_in=m.max_height_in, flags=flags, notes=state.notes, steps=state.steps,
        status="ok" if has_crop else "no_crop_found",
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_stop_agent.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass

```bash
git add scout/agent/prompts.py scout/agent/stop_agent.py tests/helpers.py tests/test_stop_agent.py
git commit -m "feat: add stop agent tool loop"
```

### Task 18: Run the stop agent on real photos

**Files:**
- Create: `scripts/run_stop.py`

- [ ] **Step 1: Write the CLI**

`scripts/run_stop.py`:
```python
"""Run the stop agent on photos and save overlays plus JSON results.

Usage: python scripts/run_stop.py --crop soybean field-images/IMG_4219.JPG field-images/IMG_4220.JPG
"""
import argparse
from pathlib import Path

from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision.sam_client import SamClient

OUT = Path("run_out")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()
    for photo in args.photos:
        stem = Path(photo).stem
        print(f"== {stem}")
        image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
        result, state = run_stop(image, args.crop, llm, sam, weeds, settings,
                                 on_step=lambda s: print(f"  [{s.tool}] {s.summary}"))
        state.overlay().save(OUT / f"{stem}.png")
        (OUT / f"{stem}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        print(f"  -> {result.status}: cover {result.weed_cover_pct}%, {result.weed_count} weeds, "
              f"species {result.species}, flags {len(result.flags)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on 10 photos**

Run:
```bash
python scripts/run_stop.py --crop corn field-images/IMG_4214.JPG field-images/IMG_4215.JPG field-images/IMG_6564.HEIC field-images/IMG_4547.JPG
python scripts/run_stop.py --crop soybean field-images/IMG_4219.JPG field-images/IMG_4220.JPG field-images/3AEE1239-C89C-4BD5-905B-C2FCBC01E161.JPG
python scripts/run_stop.py --crop alfalfa field-images/IMG_3922.JPG field-images/IMG_3926.JPG field-images/IMG_6942.HEIC
```
Expected: each photo prints its steps and a summary line, and `run_out/` holds overlays and JSON.

- [ ] **Step 3: Tune**

Open the overlays and look for three common problems:
- **Crop marked as weed:** strengthen step 1 or add the crop phrase variant to `CROP_PHRASES`.
- **Grass mats missed:** tell the agent to segment "grass" with kind "weed" when crop is corn or soybean.
- **Too many calls:** reinforce batching in step 4.

Change only `scout/agent/prompts.py`, then rerun the failing photos. Note the changes in `docs/spike-notes.md`.

- [ ] **Step 4: [USER] Choose the demo walks**

Copy 1–2 field walks of 5–10 photos each from the phone **as originals** (USB DCIM copy or iCloud download), so GPS survives. Put them in `field-images/walk1/` and `field-images/walk2/`. These stay git-ignored.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_stop.py scout/agent/prompts.py docs/spike-notes.md
git commit -m "chore: run stop agent on real photos and tune prompts"
```

---

# Day 3: Sun Nov 1. Agronomy and labels

### Task 19: [USER] Thresholds from extension sources

**Files:**
- Modify: `scout/agronomy/weeds.yaml`

- [ ] **Step 1: Set zero tolerance where a source supports it**

For `waterhemp` and `palmer_amaranth`, find a university extension or Take Action (United Soybean Board) publication that recommends zero tolerance for herbicide-resistant amaranths. For each, set `zero_tolerance: true` and `sources: ["<publication title> — <URL>"]`. Use the real title and URL you read; if you find none, leave the species unchanged.

Example of the final form of one line:
```yaml
waterhemp: {common_name: waterhemp, scientific_name: Amaranthus tuberculatus, type: broadleaf, synonyms: [common waterhemp, tall waterhemp], zero_tolerance: true, sources: ["Title of the publication you read — https://its-real-url"]}
```

- [ ] **Step 2: Add numeric thresholds only where published**

For the species most likely in the demo walks, search extension publications for action thresholds (e.g. Iowa State ICM, Nebraska CropWatch, Purdue, Ohio State, Illinois). Set:
- `action_density` in plants/m². Convert plants per ft² by multiplying by 10.764.
- or `action_cover_pct`.

Add each value's source. Species without a published number keep null; the 5% cover tool default applies to them.

- [ ] **Step 3: Verify and commit**

Run: `python -m pytest tests/test_weeds.py -v`
Expected: 3 passed (`validate()` returns `[]`)

```bash
git add scout/agronomy/weeds.yaml
git commit -m "data: add sourced weed thresholds"
```

### Task 20: Decision rules

**Files:**
- Create: `scout/agronomy/rules.py`
- Test: `tests/test_rules.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rules.py`:
```python
from scout.agronomy.rules import decide, urgency
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import LabelOption, StopResult, UseRecord

BOOK = WeedBook({
    "waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="A. tuberculatus",
                          type="broadleaf", zero_tolerance=True, sources=["src"]),
    "velvetleaf": WeedInfo(id="velvetleaf", common_name="velvetleaf", scientific_name="A. theophrasti",
                           type="broadleaf", action_density=2.0, sources=["src"]),
    "giant_foxtail": WeedInfo(id="giant_foxtail", common_name="giant foxtail", scientific_name="S. faberi",
                              type="grass"),
})


def _stop(i, species=None, cover=None, density=None, heights=None, status="ok"):
    species, cover = species or {}, cover or {}
    return StopResult(stop_index=i, image_hash=f"h{i}", species=species, species_cover_pct=cover,
                      species_density=density, species_max_height_in=heights or {},
                      weed_cover_pct=sum(cover.values()), weed_count=sum(species.values()), status=status)


def test_zero_tolerance_species_triggers_spot_treatment():
    d = decide([_stop(0, {"waterhemp": 1}, {"waterhemp": 0.2}), _stop(1), _stop(2), _stop(3)], BOOK)
    assert d.decision == "treat" and d.treat_mode == "spot"
    assert "zero tolerance" in d.rule_trace[0]


def test_density_threshold_uses_top_down_stops():
    stops = [_stop(0, {"velvetleaf": 3}, {"velvetleaf": 2.0}, density={"velvetleaf": 3.0}),
             _stop(1, {"velvetleaf": 1}, {"velvetleaf": 1.0}, density={"velvetleaf": 1.0})]
    d = decide(stops, BOOK)
    assert d.summaries[0].mean_density_per_m2 == 2.0 and d.summaries[0].outcome == "treat"
    assert d.treat_mode == "broadcast"


def test_default_cover_threshold_gives_scout_again():
    stops = [_stop(0, {"giant_foxtail": 4}, {"giant_foxtail": 3.0}),
             _stop(1, {"giant_foxtail": 2}, {"giant_foxtail": 2.0})]
    d = decide(stops, BOOK)
    assert d.summaries[0].mean_cover_pct == 2.5 and d.decision == "scout_again"
    assert "tool default" in d.rule_trace[0]


def test_no_weeds_and_no_ok_stops():
    assert decide([_stop(0), _stop(1)], BOOK).decision == "no_action"
    failed = decide([_stop(0, status="failed")], BOOK)
    assert failed.decision == "no_action" and failed.rule_trace == ["No stop could be analyzed"]


def _option(limit):
    record = UseRecord(reg_no="1-1", product_name="P", active_ingredients=[], herbicide_groups=["14"], crop="soybean",
                       species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                       max_weed_height_in=limit, page=1, quote="q")
    return LabelOption(record=record)


def test_urgency_branches():
    assert urgency(4.0, [_option(4.0)]).startswith("Beyond label size limit")
    assert urgency(3.0, [_option(4.0)]).startswith("Treat within ~3 days")
    assert urgency(1.0, [_option(4.0)]).startswith("Treat within ~7 days")
    assert urgency(None, [_option(4.0)]) is None and urgency(2.0, [_option(None)]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agronomy.rules'`

- [ ] **Step 3: Write the implementation**

`scout/agronomy/rules.py`:
```python
"""Field decision rules (spec §6). Code decides; the model only explains."""
from dataclasses import dataclass

from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import LabelOption, Outcome, SpeciesSummary, StopResult

SCOUT_AGAIN_FACTOR = 0.5
URGENT_HEIGHT_FACTOR = 0.75
SPOT_MAX_FREQUENCY = 1 / 3
SPOT_MAX_COVER_PCT = 5.0
DEFAULT_ACTION_COVER_PCT = 5.0  # tool default, not an agronomic source
SEVERITY = {"no_action": 0, "scout_again": 1, "treat": 2}


@dataclass
class Decision:
    summaries: list[SpeciesSummary]
    decision: Outcome
    treat_mode: str | None
    rule_trace: list[str]


def _species_outcome(info: WeedInfo, present: int, n_ok: int, mean_cover: float,
                     mean_density: float | None) -> tuple[Outcome, str]:
    name = info.common_name
    if info.zero_tolerance and present:
        return "treat", f"{name}: zero tolerance; present at {present} of {n_ok} stops -> treat"
    if info.action_density is not None and mean_density is not None:
        value, limit, unit, source = mean_density, info.action_density, "plants/m²", "sourced threshold"
    elif info.action_cover_pct is not None:
        value, limit, unit, source = mean_cover, info.action_cover_pct, "% cover", "sourced threshold"
    else:
        value, limit, unit, source = mean_cover, DEFAULT_ACTION_COVER_PCT, "% cover", "tool default, not from a source"
    if value >= limit:
        outcome: Outcome = "treat"
    elif value >= SCOUT_AGAIN_FACTOR * limit:
        outcome = "scout_again"
    else:
        outcome = "no_action"
    return outcome, f"{name}: {value:g} {unit} vs threshold {limit:g} ({source}) -> {outcome.replace('_', ' ')}"


def decide(stops: list[StopResult], weeds: WeedBook) -> Decision:
    ok = [s for s in stops if s.status == "ok"]
    if not ok:
        return Decision([], "no_action", None, ["No stop could be analyzed"])
    n = len(ok)
    top_down = [s for s in ok if s.species_density is not None]
    summaries, trace = [], []
    for sid in sorted({sid for s in ok for sid in s.species}):
        present = sum(1 for s in ok if sid in s.species)
        mean_cover = round(sum(s.species_cover_pct.get(sid, 0.0) for s in ok) / n, 2)
        mean_density = (round(sum(s.species_density.get(sid, 0.0) for s in top_down) / len(top_down), 2)
                        if top_down else None)
        heights = [s.species_max_height_in[sid] for s in ok if sid in s.species_max_height_in]
        outcome, line = _species_outcome(weeds.get(sid), present, n, mean_cover, mean_density)
        summaries.append(SpeciesSummary(species_id=sid, stops_present=present, frequency=round(present / n, 2),
                                        mean_cover_pct=mean_cover, mean_density_per_m2=mean_density,
                                        max_height_in=max(heights) if heights else None, outcome=outcome))
        trace.append(line)
    if not summaries:
        trace.append("No weeds found at any stop -> no action")
    decision = max((s.outcome for s in summaries), key=SEVERITY.get, default="no_action")
    treat_mode = None
    if decision == "treat":
        top_frequency = max(s.frequency for s in summaries if s.outcome == "treat")
        field_cover = sum(s.weed_cover_pct for s in ok) / n
        treat_mode = "spot" if top_frequency <= SPOT_MAX_FREQUENCY and field_cover < SPOT_MAX_COVER_PCT else "broadcast"
        trace.append(f"Treat mode: {treat_mode} (treated species at {top_frequency:.0%} of stops, "
                     f"mean weed cover {field_cover:.1f}%)")
    return Decision(summaries, decision, treat_mode, trace)


def urgency(max_height_in: float | None, options: list[LabelOption]) -> str | None:
    limits = [o.record.max_weed_height_in for o in options if o.record.max_weed_height_in is not None]
    if max_height_in is None or not limits:
        return None
    limit = min(limits)
    if max_height_in >= limit:
        return f"Beyond label size limit ({max_height_in:g} in vs {limit:g} in)"
    if max_height_in >= URGENT_HEIGHT_FACTOR * limit:
        return f"Treat within ~3 days ({max_height_in:g} in, label limit {limit:g} in)"
    return f"Treat within ~7 days ({max_height_in:g} in, label limit {limit:g} in)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rules.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agronomy/rules.py tests/test_rules.py
git commit -m "feat: add field decision rules and urgency"
```

### Task 21: EPA PPLS client

**Files:**
- Create: `scout/labels/epa.py`
- Test: `tests/test_epa.py`

- [ ] **Step 1: Write the failing test**

`tests/test_epa.py`:
```python
from scout.labels.epa import latest_pdf, parse_product

ITEM = {
    "eparegno": "524-549", "productname": "RD 1617 HERBICIDE",
    "active_ingredients": [{"pc_code": "103613", "active_ing": "Glyphosate, potassium salt", "active_ing_percent": 48.7}],
    "sites": [{"site": "CORN, FIELD"}], "pests": [{"pest": "KOCHIA"}], "types": [{"type": "HERBICIDE"}],
    "pdffiles": [{"pdffile": "000524-00549-20200225.pdf"}, {"pdffile": "000524-00549-20230406.pdf"},
                 {"pdffile": "odd-name.pdf"}],
}


def test_parse_product():
    facts = parse_product(ITEM)
    assert facts.reg_no == "524-549" and facts.product_name == "RD 1617 HERBICIDE"
    assert facts.active_ingredients == ["Glyphosate, potassium salt"]
    assert facts.sites == ["CORN, FIELD"] and facts.pests == ["KOCHIA"]
    assert facts.is_herbicide and facts.latest_pdf == "000524-00549-20230406.pdf"


def test_latest_pdf_handles_missing_dates():
    assert latest_pdf([{"pdffile": "odd.pdf"}]) is None
    assert latest_pdf([]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_epa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.labels.epa'`

- [ ] **Step 3: Write the implementation**

`scout/labels/epa.py`:
```python
"""EPA Pesticide Product Label System (PPLS): product facts and the latest label PDF."""
import re
import ssl
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

PPLS_URL = "https://ordspub.epa.gov/ords/pesticides/cswu/ppls/{reg_no}"
PDF_BASE = "https://www3.epa.gov/pesticides/chem_search/ppls/"
OP_LEGACY_SERVER_CONNECT = 0x4
_PDF_DATE = re.compile(r"-(\d{8})\.pdf$", re.IGNORECASE)


class LegacyTLSAdapter(HTTPAdapter):
    """Keeps certificate checks on, but allows the legacy renegotiation some EPA hosts need under OpenSSL 3."""

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def epa_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://ordspub.epa.gov", LegacyTLSAdapter())
    session.mount("https://www3.epa.gov", LegacyTLSAdapter())
    return session


@dataclass
class ProductFacts:
    reg_no: str
    product_name: str
    active_ingredients: list[str]
    sites: list[str]
    pests: list[str]
    types: list[str]
    latest_pdf: str | None

    @property
    def is_herbicide(self) -> bool:
        return "HERBICIDE" in self.types


def latest_pdf(pdffiles: list[dict]) -> str | None:
    dated = [(m.group(1), f["pdffile"]) for f in pdffiles if (m := _PDF_DATE.search(f.get("pdffile", "")))]
    return max(dated)[1] if dated else None


def parse_product(item: dict) -> ProductFacts:
    return ProductFacts(
        reg_no=item["eparegno"],
        product_name=item["productname"],
        active_ingredients=[a["active_ing"] for a in item.get("active_ingredients") or []],
        sites=[s["site"] for s in item.get("sites") or []],
        pests=[p["pest"] for p in item.get("pests") or []],
        types=[t["type"] for t in item.get("types") or []],
        latest_pdf=latest_pdf(item.get("pdffiles") or []),
    )


def fetch_product(reg_no: str, session: requests.Session) -> ProductFacts:
    response = session.get(PPLS_URL.format(reg_no=reg_no), timeout=60)
    response.raise_for_status()
    items = response.json().get("items") or []
    if not items:
        raise LookupError(f"EPA has no product {reg_no}")
    return parse_product(items[0])


def download_pdf(filename: str, dest_dir: Path, session: requests.Session) -> Path:
    path = Path(dest_dir) / filename
    if not path.exists():
        response = session.get(PDF_BASE + filename, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_epa.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add scout/labels/epa.py tests/test_epa.py
git commit -m "feat: add EPA PPLS client"
```

### Task 22: Label extraction and quote verification

**Files:**
- Create: `scout/labels/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

`tests/test_extract.py`:
```python
import json

from scout.agent.llm import LlmTurn
from scout.labels.epa import ProductFacts
from scout.labels.extract import USE_RECORDS_FORMAT, extract_use_records, verify
from scout.models import UseRecord
from tests.helpers import FakeLlm, small_weedbook

FACTS = ProductFacts(reg_no="100-1", product_name="Test Herbicide", active_ingredients=["fomesafen"], sites=[],
                     pests=[], types=["HERBICIDE"], latest_pdf="000100-00001-20240101.pdf")


def _item(**over):
    base = {"crop": "soybean", "species_id": "waterhemp", "weed_label_text": "Waterhemp", "timing": "postemergence",
            "herbicide_groups": ["14"], "max_weed_height_in": 4, "crop_stage_min": None, "crop_stage_max": "R1",
            "rate_min": 1.0, "rate_max": 1.5, "rate_unit": "pt/A", "phi_days": None, "rei_hours": 24, "page": 3,
            "quote": "Apply to waterhemp up to 4 inches tall"}
    base.update(over)
    return base


def test_extract_maps_species_and_fills_product_facts(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    llm = FakeLlm([LlmTurn("r1", [], json.dumps({"records": [
        _item(), _item(species_id="Marestail"), _item(species_id="dragonweed")]}))])
    records = extract_use_records(pdf, FACTS, llm, small_weedbook())
    assert [r.species_id for r in records] == ["waterhemp", "horseweed"]
    assert records[0].reg_no == "100-1" and records[0].pdf_file == "000100-00001-20240101.pdf"
    request = llm.requests[0]
    assert request["text_format"] == USE_RECORDS_FORMAT
    assert request["input"][0]["content"][1]["type"] == "input_file"


def _record(page, quote):
    return UseRecord(reg_no="1", product_name="P", active_ingredients=[], herbicide_groups=[], crop="soybean",
                     species_id="waterhemp", weed_label_text="w", timing="postemergence", page=page, quote=quote)


def test_verify_keeps_found_quotes_and_fixes_adjacent_page():
    texts = ["Cover page", "Apply to  waterhemp up to\n4 inches tall.", "Other text"]
    kept, dropped = verify([_record(2, "apply to waterhemp up to 4 inches tall"),
                            _record(3, "Apply to waterhemp up to 4 inches tall"),
                            _record(1, "never printed anywhere on label")], texts)
    assert [r.page for r in kept] == [2, 2]
    assert len(dropped) == 1


def test_verify_drops_short_quotes():
    kept, dropped = verify([_record(1, "Cover")], ["Cover page"])
    assert kept == [] and len(dropped) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.labels.extract'`

- [ ] **Step 3: Write the implementation**

`scout/labels/extract.py`:
```python
"""Read a label PDF with Muse Spark into UseRecords, then keep only records whose quote is really on the page."""
import base64
import json
import re
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from scout.agent.llm import user_message
from scout.agronomy.weeds import WeedBook
from scout.labels.epa import ProductFacts
from scout.models import UseRecord

MIN_QUOTE_CHARS = 15


def _nullable(kind: str) -> dict:
    return {"type": [kind, "null"]}


_RECORD_PROPERTIES = {
    "crop": {"type": "string", "enum": ["corn", "soybean", "alfalfa"]},
    "species_id": {"type": "string"},
    "weed_label_text": {"type": "string"},
    "timing": {"type": "string", "enum": ["preemergence", "postemergence", "burndown"]},
    "herbicide_groups": {"type": "array", "items": {"type": "string"}},
    "max_weed_height_in": _nullable("number"),
    "crop_stage_min": _nullable("string"),
    "crop_stage_max": _nullable("string"),
    "rate_min": _nullable("number"),
    "rate_max": _nullable("number"),
    "rate_unit": _nullable("string"),
    "phi_days": _nullable("integer"),
    "rei_hours": _nullable("integer"),
    "page": {"type": "integer"},
    "quote": {"type": "string"},
}
USE_RECORDS_FORMAT = {
    "type": "json_schema",
    "name": "use_records",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"records": {"type": "array", "items": {
            "type": "object", "properties": _RECORD_PROPERTIES,
            "required": list(_RECORD_PROPERTIES), "additionalProperties": False}}},
        "required": ["records"],
        "additionalProperties": False,
    },
}

EXTRACT_PROMPT = """You are reading the EPA-registered label for {product} (EPA Reg. No. {reg_no}).
Extract one record per (crop, weed, timing) for these crops only: corn (field corn), soybean, alfalfa.
species_id must be one of these ids; map label weed names to them and skip weeds that do not map:
{species}
For each record give: the maximum weed height in inches the label allows (null if none); crop stage limits such as \
V2 or V8 (null if none); the rate range and unit as printed; the pre-harvest interval in days; the restricted-entry \
interval in hours; the herbicide group number(s) printed on the label.
page is the 1-based page index within this PDF file where the record's facts appear. quote is a verbatim snippet \
of 15 to 200 characters copied from that page that supports the record.
Return only records the label clearly supports."""


def pdf_data_url(path: Path) -> str:
    return "data:application/pdf;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def page_texts(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _norm(text: str) -> str:
    text = text.lower()
    for old, new in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def verify(records: list[UseRecord], texts: list[str]) -> tuple[list[UseRecord], list[UseRecord]]:
    """Keep a record only if its quote appears on its page (or the page next to it, which then becomes its page)."""
    normed = [_norm(t) for t in texts]
    kept, dropped = [], []
    for record in records:
        quote = _norm(record.quote)
        found = None
        if len(quote) >= MIN_QUOTE_CHARS:
            for page in (record.page, record.page - 1, record.page + 1):
                if 1 <= page <= len(normed) and quote in normed[page - 1]:
                    found = page
                    break
        if found is None:
            dropped.append(record)
        else:
            kept.append(record.model_copy(update={"page": found}))
    return kept, dropped


def extract_use_records(pdf_path: Path, facts: ProductFacts, llm, weeds: WeedBook) -> list[UseRecord]:
    species = [sid for sid in weeds.ids() if not sid.startswith("unidentified")]
    listing = "\n".join(f"- {sid}: {', '.join([weeds.get(sid).common_name, *weeds.get(sid).synonyms])}"
                        for sid in species)
    prompt = EXTRACT_PROMPT.format(product=facts.product_name, reg_no=facts.reg_no, species=listing)
    turn = llm.turn([user_message(prompt, file=(Path(pdf_path).name, pdf_data_url(pdf_path)))],
                    text_format=USE_RECORDS_FORMAT)
    records = []
    for item in json.loads(turn.text)["records"]:
        sid = item["species_id"] if item["species_id"] in weeds else weeds.resolve(item["species_id"])
        if sid is None or sid.startswith("unidentified"):
            continue
        try:
            records.append(UseRecord(reg_no=facts.reg_no, product_name=facts.product_name,
                                     active_ingredients=facts.active_ingredients, pdf_file=facts.latest_pdf,
                                     **{**item, "species_id": sid}))
        except ValidationError:
            continue
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scout/labels/extract.py tests/test_extract.py
git commit -m "feat: extract label use records and verify quotes"
```

### Task 23: [USER] Build the label table

**Files:**
- Create: `scout/labels/focus_products.csv`, `scout/labels/build_labels.py`, `scout/labels/use_records.json` (generated)

- [ ] **Step 1: [USER] Pick the products**

Create `scout/labels/focus_products.csv` with about 20 postemergence herbicides labeled for corn, soybean, or alfalfa. Prefer products that cover the weeds in your demo walks, and span several herbicide groups (for example 2, 4, 5, 9, 10, 14, 15, 27). Header and one row per product:
```
reg_no,note
524-549,glyphosate (example row; replace with your picks)
```
Look up registration numbers at https://ordspub.epa.gov/ords/pesticides/f?p=PPLS:1 or on the product labels you know.

- [ ] **Step 2: Write the CLI**

`scout/labels/build_labels.py`:
```python
"""Build use_records.json from focus_products.csv.

Usage: python -m scout.labels.build_labels [--only REG_NO ...]
"""
import argparse
import csv
import json
from pathlib import Path

from scout.agent.llm import Llm
from scout.agronomy.weeds import WeedBook
from scout.config import get_settings
from scout.labels.epa import download_pdf, epa_session, fetch_product
from scout.labels.extract import extract_use_records, page_texts, verify

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    llm, weeds, session = Llm.from_settings(get_settings()), WeedBook.load(), epa_session()
    pdf_dir = HERE / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    out_path = HERE / "use_records.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    with open(HERE / "focus_products.csv", encoding="utf-8") as handle:
        reg_nos = [row["reg_no"].strip() for row in csv.DictReader(handle) if row["reg_no"].strip()]
    if args.only:
        reg_nos = [r for r in reg_nos if r in args.only]
    records = [r for r in existing if r["reg_no"] not in reg_nos]
    for reg_no in reg_nos:
        try:
            facts = fetch_product(reg_no, session)
            if not facts.is_herbicide or not facts.latest_pdf:
                print(f"{reg_no}: skipped (not a herbicide or no label PDF)")
                continue
            pdf = download_pdf(facts.latest_pdf, pdf_dir, session)
            kept, dropped = verify(extract_use_records(pdf, facts, llm, weeds), page_texts(pdf))
            records += [r.model_dump() for r in kept]
            print(f"{reg_no} {facts.product_name}: kept {len(kept)}, dropped {len(dropped)}")
        except Exception as error:  # one bad product must not stop the batch
            print(f"{reg_no}: failed ({error})")
    out_path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Try two products first**

Run: `python -m scout.labels.build_labels --only <first reg_no> <second reg_no>`
Expected: two lines with kept/dropped counts. If almost everything is dropped, compare a dropped quote against `page_texts`. If quotes are paraphrased, add "copy characters exactly; do not paraphrase" to `EXTRACT_PROMPT` and rerun.

- [ ] **Step 4: Run all products**

Run: `python -m scout.labels.build_labels`
Expected: one line per product and `Wrote N records`, with N > 0 for each demo species/crop pair.

- [ ] **Step 5: [USER] Hand-check 10 records**

Pick 10 random entries from `use_records.json`, open each PDF at its `page`, and confirm crop, weed, height limit, and rate. Target ≥ 9/10 correct. Record the result in `docs/spike-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add scout/labels/focus_products.csv scout/labels/build_labels.py scout/labels/use_records.json docs/spike-notes.md
git commit -m "data: build verified label use records"
```

### Task 24: Label lookup

**Files:**
- Create: `scout/labels/lookup.py`
- Test: `tests/test_lookup.py`

- [ ] **Step 1: Write the failing test**

`tests/test_lookup.py`:
```python
from scout.labels.lookup import find_label_options, load_records, stage_index
from scout.models import UseRecord


def _r(reg, groups, limit=None, stage_max=None, species="waterhemp", crop="soybean", timing="postemergence"):
    return UseRecord(reg_no=reg, product_name=f"P{reg}", active_ingredients=[], herbicide_groups=groups, crop=crop,
                     species_id=species, weed_label_text=species, timing=timing, max_weed_height_in=limit,
                     crop_stage_min="VE" if stage_max else None, crop_stage_max=stage_max, page=1, quote="q")


def test_stage_index_orders_stages():
    assert stage_index("V3") < stage_index("R1") and stage_index("v3") == stage_index("V3")
    assert stage_index("bloom") is None and stage_index(None) is None


def test_filters_crop_species_timing_and_stage():
    records = [_r("1", ["14"], 4), _r("2", ["14"], crop="corn"), _r("3", ["15"], timing="preemergence"),
               _r("4", ["10"], stage_max="V2"), _r("5", ["9"], species="velvetleaf")]
    options = find_label_options(records, "soybean", "waterhemp", 2.0, "V3")
    assert [o.record.reg_no for o in options] == ["1"]


def test_over_size_kept_last_and_groups_diversified():
    records = [_r("a", ["14"], 2), _r("b", ["14"], 6), _r("c", ["10"], 5), _r("b", ["14"], 3)]
    options = find_label_options(records, "soybean", "waterhemp", 4.0, None)
    assert [o.record.reg_no for o in options] == ["b", "c", "a"]
    assert options[-1].over_size_limit and not options[0].over_size_limit
    assert options[0].record.max_weed_height_in == 6


def test_load_records_missing_file(tmp_path):
    assert load_records(tmp_path / "none.json") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lookup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.labels.lookup'`

- [ ] **Step 3: Write the implementation**

`scout/labels/lookup.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lookup.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/labels/lookup.py tests/test_lookup.py
git commit -m "feat: add label option lookup"
```

---

# Day 4: Mon Nov 2. Field step, report, and app

### Task 25: Field step

**Files:**
- Create: `scout/agent/field_agent.py`
- Test: `tests/test_field_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_field_agent.py`:
```python
from scout.agent.field_agent import build_report, numbers_ok
from scout.agent.llm import LlmTurn
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import StopResult, UseRecord
from tests.helpers import FakeLlm

BOOK = WeedBook({"waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="A. tuberculatus",
                                       type="broadleaf", zero_tolerance=True, sources=["src"])})
RECORD = UseRecord(reg_no="100-1", product_name="Product A", active_ingredients=["fomesafen"], herbicide_groups=["14"],
                   crop="soybean", species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                   max_weed_height_in=4.0, page=3, quote="up to 4 inches")


def _stops():
    return [StopResult(stop_index=0, image_hash="a", species={"waterhemp": 2}, species_cover_pct={"waterhemp": 1.2},
                       species_max_height_in={"waterhemp": 3.0}, weed_cover_pct=1.2, weed_count=2, crop_stage="V3"),
            StopResult(stop_index=1, image_hash="b", crop_stage="V3")]


def test_numbers_ok():
    assert numbers_ok("Waterhemp at 1 of 2 stops, 3 in tall.", '{"a": 1, "b": 2.0, "h": 3.0}')
    assert not numbers_ok("Spray 12 oz.", '{"a": 1}')


def test_report_uses_llm_text_when_numbers_check_out():
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found and has zero tolerance. Read the label before use.")])
    report = build_report("w1", "soybean", _stops(), BOOK, [RECORD], llm)
    assert report.decision == "treat" and report.treat_mode == "broadcast"
    assert [o.record.reg_no for o in report.options] == ["100-1"]
    assert report.urgency.startswith("Treat within ~3 days")
    assert report.explanation.startswith("Waterhemp was found")


def test_report_falls_back_to_template_on_invented_numbers():
    llm = FakeLlm([LlmTurn("r1", [], "Spray 99 acres."), LlmTurn("r2", [], "Spray 98 acres.")])
    report = build_report("w1", "soybean", _stops(), BOOK, [RECORD], llm)
    assert report.explanation.startswith("Treat") and report.explanation.endswith("Read the label before use.")


def test_no_label_match_is_stated():
    report = build_report("w1", "soybean", _stops(), BOOK, [], FakeLlm([]))
    assert report.options == []
    assert any("No product in the current label set covers waterhemp on soybean" in line for line in report.rule_trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_field_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.field_agent'`

- [ ] **Step 3: Write the implementation**

`scout/agent/field_agent.py`:
```python
"""Field step: rules decide, the label table supplies options, Muse Spark explains using only known numbers."""
import json
import re
from collections import Counter

from scout.agent.llm import user_message
from scout.agent.prompts import FIELD_INSTRUCTIONS
from scout.agronomy.rules import decide, urgency
from scout.agronomy.weeds import WeedBook
from scout.labels.lookup import find_label_options
from scout.models import FieldReport, LabelOption, StopResult, UseRecord

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
OUTCOME_WORDS = {"no_action": "No action needed", "scout_again": "Scout again in 5-7 days", "treat": "Treat"}


def _canon(number: str) -> str:
    return f"{float(number):g}"


def numbers_ok(text: str, allowed: str) -> bool:
    """True when every number in `text` also appears in `allowed`."""
    known = {_canon(n) for n in _NUMBER.findall(allowed)}
    return all(_canon(n) in known for n in _NUMBER.findall(text))


def _urgency_rank(text: str | None) -> int:
    if text is None:
        return 9
    return 0 if text.startswith("Beyond") else 1 if "~3 days" in text else 2


def template_explanation(decision: str, treat_mode: str | None, rule_trace: list[str]) -> str:
    head = OUTCOME_WORDS[decision] + (f" ({treat_mode})" if treat_mode else "") + "."
    return " ".join([head, *[line + "." for line in rule_trace[:4]], "Read the label before use."])


def build_report(walk_id: str, crop: str, stops: list[StopResult], weeds: WeedBook, records: list[UseRecord],
                 llm) -> FieldReport:
    d = decide(stops, weeds)
    stages = Counter(s.crop_stage for s in stops if s.status == "ok" and s.crop_stage)
    stage = stages.most_common(1)[0][0] if stages else None
    options: list[LabelOption] = []
    seen: set[str] = set()
    urgency_text = None
    for summary in d.summaries:
        if summary.outcome != "treat":
            continue
        found = find_label_options(records, crop, summary.species_id, summary.max_height_in, stage)
        if not found:
            d.rule_trace.append(f"No product in the current label set covers "
                                f"{weeds.get(summary.species_id).common_name} on {crop}")
        candidate = urgency(summary.max_height_in, found)
        if _urgency_rank(candidate) < _urgency_rank(urgency_text):
            urgency_text = candidate
        for option in found:
            if option.record.reg_no not in seen:
                seen.add(option.record.reg_no)
                options.append(option)
    payload = json.dumps({
        "crop": crop, "decision": d.decision, "treat_mode": d.treat_mode, "urgency": urgency_text,
        "species": [s.model_dump() for s in d.summaries], "rule_trace": d.rule_trace,
        "options": [{"product": o.record.product_name, "groups": o.record.herbicide_groups,
                     "max_weed_height_in": o.record.max_weed_height_in, "over_size_limit": o.over_size_limit}
                    for o in options],
    }, ensure_ascii=False)
    explanation = None
    for _ in range(2):
        try:
            text = llm.turn([user_message(payload)], instructions=FIELD_INSTRUCTIONS).text.strip()
        except Exception:  # the template below is a complete fallback
            break
        if text and numbers_ok(text, payload):
            explanation = text
            break
    return FieldReport(
        walk_id=walk_id, crop=crop, stops=stops, species_summary=d.summaries, decision=d.decision,
        treat_mode=d.treat_mode, urgency=urgency_text, rule_trace=d.rule_trace,
        explanation=explanation or template_explanation(d.decision, d.treat_mode, d.rule_trace), options=options,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_field_agent.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/field_agent.py tests/test_field_agent.py
git commit -m "feat: add field step with checked explanation"
```

### Task 26: Walk pipeline

**Files:**
- Create: `scout/app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from scout.agent.state import StopState
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings
from scout.models import StopResult
from tests.helpers import FakeLlm, make_image, small_weedbook


def fake_runner(image, crop, llm, sam, weeds, settings, on_step=None):
    state = StopState(image, crop, on_step=on_step)
    state.log("segment", "fake step")
    return StopResult(stop_index=image.index, image_hash=image.hash, species={"waterhemp": 1},
                      species_cover_pct={"waterhemp": 1.0}, weed_cover_pct=1.0, weed_count=1), state


def _pipeline(tmp_path, runner):
    return Pipeline(FakeLlm([]), None, small_weedbook(), [], Settings(api_key="t"), tmp_path / "store",
                    demo_dir=tmp_path / "demo", stop_runner=runner)


def test_run_writes_report_and_overlays(tmp_path):
    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    assert walk.status == "done"
    kinds = [e["type"] for e in walk.events]
    assert kinds.count("stop_done") == 2 and kinds.count("step") == 2 and kinds[-1] == "report_ready"
    folder = pipeline.find_dir(walk.id)
    assert (folder / "report.json").exists() and (folder / "stop_1.png").exists()
    assert pipeline.cached(walk.id).walk_id == walk.id


def test_failed_stop_does_not_sink_walk(tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, boom)
    walk = Walk(id=walk_id_for("corn", images), crop="corn")
    pipeline.run(walk, images)
    assert walk.status == "done" and walk.report.stops[0].status == "failed"


def test_walk_id_is_stable_and_hex():
    images = [make_image()]
    assert walk_id_for("corn", images) == walk_id_for("corn", images)
    assert walk_id_for("corn", images) != walk_id_for("soybean", images)
    assert len(walk_id_for("corn", images)) == 16


def test_find_dir_rejects_bad_ids(tmp_path):
    assert _pipeline(tmp_path, fake_runner).find_dir("../etc") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.app.pipeline'`

- [ ] **Step 3: Write the implementation**

`scout/app/pipeline.py`:
```python
"""Run a walk: every stop through the agent (in threads), then the field step. Progress goes to Walk.events."""
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from scout.agent.field_agent import build_report
from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.labels.lookup import load_records
from scout.models import FieldReport, StopResult
from scout.vision.sam_client import SamClient

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
_WALK_ID = re.compile(r"[0-9a-f]{16}")
log = logging.getLogger(__name__)


@dataclass
class Walk:
    id: str
    crop: str
    status: str = "running"  # running | done | failed
    events: list[dict] = field(default_factory=list)
    report: FieldReport | None = None
    error: str | None = None

    def emit(self, event: dict) -> None:
        self.events.append(event)


def walk_id_for(crop: str, images: list[StopImage]) -> str:
    key = crop + "|" + ",".join(image.hash for image in images)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, llm, sam, weeds: WeedBook, records, settings: Settings, store_dir: Path,
                 demo_dir: Path = DEMO_DIR, stop_runner=run_stop):
        self.llm = llm
        self.sam = sam
        self.weeds = weeds
        self.records = records
        self.settings = settings
        self.store_dir = Path(store_dir)
        self.demo_dir = Path(demo_dir)
        self._stop_runner = stop_runner

    @classmethod
    def from_settings(cls, settings: Settings) -> "Pipeline":
        return cls(Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load(), load_records(),
                   settings, Path(os.environ.get("SCOUT_STORE", "store")))

    def find_dir(self, walk_id: str) -> Path | None:
        if not _WALK_ID.fullmatch(walk_id):
            return None
        for base in (self.store_dir, self.demo_dir):
            folder = base / walk_id
            if (folder / "report.json").exists():
                return folder
        return None

    def cached(self, walk_id: str) -> FieldReport | None:
        folder = self.find_dir(walk_id)
        if folder is None:
            return None
        return FieldReport.model_validate_json((folder / "report.json").read_text(encoding="utf-8"))

    def samples(self) -> list[dict]:
        index = self.demo_dir / "samples.json"
        return json.loads(index.read_text(encoding="utf-8")) if index.exists() else []

    def run(self, walk: Walk, images: list[StopImage]) -> None:
        """Events are emitted before status changes, so a reader that sees a final status has every event."""
        try:
            report = self.cached(walk.id)
            if report is None:
                out_dir = self.store_dir / walk.id
                out_dir.mkdir(parents=True, exist_ok=True)
                with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_stops) as pool:
                    stops = list(pool.map(lambda image: self._stop(walk, image, out_dir), images))
                report = build_report(walk.id, walk.crop, stops, self.weeds, self.records, self.llm)
                (out_dir / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
            walk.report = report
            walk.emit({"type": "report_ready"})
            walk.status = "done"
        except Exception as error:
            log.exception("walk %s failed", walk.id)
            walk.error = str(error)
            walk.emit({"type": "error", "message": "The walk couldn't be analyzed. Try again."})
            walk.status = "failed"

    def _stop(self, walk: Walk, image: StopImage, out_dir: Path) -> StopResult:
        def on_step(step):
            walk.emit({"type": "step", "stop": image.index, "tool": step.tool, "summary": step.summary, "ok": step.ok})

        try:
            result, state = self._stop_runner(image, walk.crop, self.llm, self.sam, self.weeds, self.settings,
                                              on_step=on_step)
            state.overlay().save(out_dir / f"stop_{image.index}.png")
        except Exception:
            log.exception("stop %s of walk %s failed", image.index, walk.id)
            result = StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, status="failed",
                                flags=["This stop could not be analyzed"])
            image.work.save(out_dir / f"stop_{image.index}.png")
        walk.emit({"type": "stop_done", "stop": image.index, "status": result.status})
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scout/app/pipeline.py tests/test_pipeline.py
git commit -m "feat: add walk pipeline with cache and failure isolation"
```

### Task 27: Report HTML and CSV

**Files:**
- Create: `scout/report/render.py`, `scout/report/templates/report.html`
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing test**

`tests/test_render.py`:
```python
from scout.models import FieldReport, LabelOption, SpeciesSummary, StopResult, UseRecord
from scout.report.render import CSV_COLUMNS, report_csv, report_html
from tests.helpers import small_weedbook


def _report(gps=None):
    record = UseRecord(reg_no="100-1", product_name="Product A", active_ingredients=[], herbicide_groups=["14"],
                       crop="soybean", species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                       max_weed_height_in=4.0, rate_min=1.0, rate_max=1.5, rate_unit="pt/A", page=3,
                       quote="up to 4 inches", pdf_file="000100-00001-20240101.pdf")
    stop = StopResult(stop_index=0, image_hash="a", gps=gps, species={"waterhemp": 2},
                      species_cover_pct={"waterhemp": 1.2}, weed_cover_pct=1.2, weed_count=2,
                      flags=["Check s0-i6: low confidence"])
    summary = SpeciesSummary(species_id="waterhemp", stops_present=1, frequency=1.0, mean_cover_pct=1.2,
                             mean_density_per_m2=None, max_height_in=3.0, outcome="treat")
    return FieldReport(walk_id="w1", crop="soybean", stops=[stop], species_summary=[summary], decision="treat",
                       treat_mode="broadcast", urgency="Treat within ~3 days (3 in, label limit 4 in)",
                       rule_trace=["waterhemp: zero tolerance"], explanation="Waterhemp found.",
                       options=[LabelOption(record=record)])


def test_html_has_decision_options_and_overlays():
    html = report_html(_report(), small_weedbook(), lambda i: f"/walks/w1/overlay/{i}.png")
    assert "Treat (broadcast)" in html and "Product A (100-1)" in html
    assert "/walks/w1/overlay/0.png" in html and "Read the label before use" in html
    assert "#page=3" in html and "Check s0-i6: low confidence" in html
    assert "leaflet" not in html


def test_html_shows_map_when_gps_present():
    html = report_html(_report(gps=(41.5, -93.6)), small_weedbook(), lambda i: f"/o/{i}.png")
    assert "leaflet" in html and "41.5" in html


def test_csv_has_one_row_per_stop():
    lines = report_csv(_report()).strip().splitlines()
    assert lines[0].split(",") == CSV_COLUMNS and len(lines) == 2
    assert "waterhemp:2" in lines[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.report.render'`

- [ ] **Step 3: Write the template**

`scout/report/templates/report.html`:
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scouting report: {{ report.crop }}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #fafaf7; color: #1f1f1c; }
  main { max-width: 960px; margin: 0 auto; padding: 16px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .card { background: #fff; border: 1px solid #e3e1d9; border-radius: 10px; padding: 12px; }
  .card b { display: block; font-size: 22px; font-weight: 600; }
  .treat { background: #faeeda; } .scout_again { background: #e6f1fb; } .no_action { background: #eaf3de; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  td, th { border-bottom: 1px solid #e3e1d9; padding: 6px; text-align: left; vertical-align: top; }
  .stops { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
  .stops img { width: 100%; border-radius: 8px; }
  .flag { color: #a32d2d; font-size: 13px; }
  .muted { color: #6b6a63; font-size: 13px; }
  .table-wrap { overflow-x: auto; }
  @media print { .noprint { display: none; } }
</style>
</head>
<body>
<main>
<p class="noprint"><a href="/">New walk</a> · <a href="stops.csv">Download CSV</a> · <a href="#" onclick="window.print(); return false;">Save as PDF</a></p>
<h1>Scouting report: {{ report.crop }}, {{ report.stops|length }} stops</h1>
<div class="cards">
  <div class="card">Mean weed cover<b>{{ "%.1f"|format(mean_cover) }}%</b></div>
  <div class="card">Density<b>{{ density_text }}</b></div>
  <div class="card">Stops with weeds<b>{{ stops_with_weeds }} of {{ ok_count }}</b></div>
  <div class="card {{ report.decision }}">Decision<b>{{ decision_text }}</b>{% if report.urgency %}<span class="muted">{{ report.urgency }}</span>{% endif %}</div>
</div>

<h2>Why</h2>
<p>{{ report.explanation }}</p>
<details><summary>Rule trace</summary><ul>{% for line in report.rule_trace %}<li>{{ line }}</li>{% endfor %}</ul></details>

<h2>Weeds found</h2>
{% if report.species_summary %}
<div class="table-wrap"><table>
  <tr><th>Species</th><th>Stops</th><th>Mean cover</th><th>Density</th><th>Max height</th><th>Outcome</th></tr>
  {% for s in report.species_summary %}
  <tr>
    <td>{{ names[s.species_id] }}</td><td>{{ s.stops_present }}</td><td>{{ s.mean_cover_pct }}%</td>
    <td>{{ s.mean_density_per_m2 if s.mean_density_per_m2 is not none else "—" }}</td>
    <td>{{ (s.max_height_in|string + " in") if s.max_height_in is not none else "—" }}</td>
    <td>{{ outcome_text[s.outcome] }}</td>
  </tr>
  {% endfor %}
</table></div>
{% else %}<p>No weeds found.</p>{% endif %}

<h2>Label-backed options</h2>
{% if report.options %}
<div class="table-wrap"><table>
  <tr><th>Product (EPA reg. no.)</th><th>Group</th><th>Limits</th><th>Rate</th><th>Pre-harvest interval</th><th>Source</th></tr>
  {% for o in report.options %}{% set r = o.record %}
  <tr>
    <td>{{ r.product_name }} ({{ r.reg_no }})</td>
    <td>{{ r.herbicide_groups|join(", ") }}</td>
    <td>{% if r.max_weed_height_in is not none %}Weeds up to {{ r.max_weed_height_in }} in{% endif %}{% if r.crop_stage_max %} · crop up to {{ r.crop_stage_max }}{% endif %}{% if o.over_size_limit %}<div class="flag">Weeds are over this size limit</div>{% endif %}</td>
    <td>{% if r.rate_min is not none %}{{ r.rate_min }}{% if r.rate_max is not none and r.rate_max != r.rate_min %}–{{ r.rate_max }}{% endif %} {{ r.rate_unit or "" }}{% else %}—{% endif %}</td>
    <td>{{ (r.phi_days|string + " days") if r.phi_days is not none else "—" }}</td>
    <td>{% if r.pdf_file %}<a href="https://www3.epa.gov/pesticides/chem_search/ppls/{{ r.pdf_file }}#page={{ r.page }}">p. {{ r.page }}</a>{% else %}p. {{ r.page }}{% endif %}<div class="muted">“{{ r.quote }}”</div></td>
  </tr>
  {% endfor %}
</table></div>
{% else %}<p>No product in the current label set matches.</p>{% endif %}
<p class="muted">Read the label before use. The label is the law. This report is decision support, not a prescription.</p>

{% set pins = report.stops | selectattr("gps") | list %}
{% if pins %}
<h2>Stops on the map</h2>
<div id="map" style="height: 280px; border-radius: 10px;"></div>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const pins = {{ pins | map(attribute="gps") | list | tojson }};
  const labels = {{ pins | map(attribute="stop_index") | list | tojson }};
  const map = L.map("map");
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(map);
  pins.forEach((p, i) => L.marker(p).addTo(map).bindPopup(`Stop ${labels[i] + 1}`));
  map.fitBounds(pins, { padding: [30, 30], maxZoom: 18 });
</script>
{% endif %}

<h2>Stops</h2>
<div class="stops">
{% for s in report.stops %}
  <div class="card">
    <img src="{{ overlay_url(s.stop_index) }}" alt="Stop {{ s.stop_index + 1 }}: crop in green, weeds in red">
    <b>Stop {{ s.stop_index + 1 }}</b>
    <span class="muted">{{ s.view_type.replace("_", " ") }}{% if s.crop_stage %} · crop {{ s.crop_stage }}{% endif %} · {{ s.status.replace("_", " ") }}</span>
    <p>Weed cover {{ s.weed_cover_pct }}% · {{ s.weed_count }} weeds{% if s.count_approximate %} (approx.){% endif %}{% if s.density_per_m2 is not none %} · {{ s.density_per_m2 }}/m²{% endif %}</p>
    {% for f in s.flags %}<div class="flag">{{ f }}</div>{% endfor %}
  </div>
{% endfor %}
</div>
</main>
</body>
</html>
```

- [ ] **Step 4: Write the renderer**

`scout/report/render.py`:
```python
"""Report page (HTML) and per-stop CSV."""
import csv
import io
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from scout.agronomy.weeds import WeedBook
from scout.models import FieldReport

_ENV = Environment(loader=FileSystemLoader(Path(__file__).with_name("templates")),
                   autoescape=select_autoescape(["html"]))
OUTCOME_TEXT = {"no_action": "No action", "scout_again": "Scout again in 5–7 days", "treat": "Treat"}
CSV_COLUMNS = ["stop", "status", "view_type", "weed_cover_pct", "crop_cover_pct", "weed_count", "count_approximate",
               "density_per_m2", "in_row_weed_share", "species", "max_height_in", "flags"]


def _blank(value) -> object:
    return "" if value is None else value


def report_html(report: FieldReport, weeds: WeedBook, overlay_url: Callable[[int], str]) -> str:
    ok = [s for s in report.stops if s.status == "ok"]
    densities = [s.density_per_m2 for s in ok if s.density_per_m2 is not None]
    names = {s.species_id: weeds.get(s.species_id).common_name if s.species_id in weeds else s.species_id
             for s in report.species_summary}
    decision = OUTCOME_TEXT[report.decision] + (f" ({report.treat_mode})" if report.treat_mode else "")
    return _ENV.get_template("report.html").render(
        report=report,
        mean_cover=sum(s.weed_cover_pct for s in ok) / len(ok) if ok else 0.0,
        density_text=f"{sum(densities) / len(densities):.1f} /m²" if densities else "—",
        stops_with_weeds=sum(1 for s in ok if s.weed_count),
        ok_count=len(ok),
        decision_text=decision,
        names=names,
        outcome_text=OUTCOME_TEXT,
        overlay_url=overlay_url,
    )


def report_csv(report: FieldReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for s in report.stops:
        writer.writerow([s.stop_index + 1, s.status, s.view_type, s.weed_cover_pct, s.crop_cover_pct, s.weed_count,
                         s.count_approximate, _blank(s.density_per_m2), _blank(s.in_row_weed_share),
                         ";".join(f"{k}:{v}" for k, v in s.species.items()), _blank(s.max_height_in),
                         " | ".join(s.flags)])
    return buf.getvalue()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_render.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add scout/report tests/test_render.py
git commit -m "feat: render report page and CSV"
```

### Task 28: Web page

**Files:**
- Create: `scout/web/index.html`, `scout/web/app.js`, `scout/web/style.css`

- [ ] **Step 1: Write the page**

`scout/web/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weed scout</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <h1>Weed scout</h1>
  <p class="lede">Upload the photos from one walk through one field. An agent finds the weeds, measures them, and checks herbicide labels.</p>
  <form id="walk-form">
    <label>Crop
      <select name="crop" required>
        <option value="soybean">Soybean</option>
        <option value="corn">Corn</option>
        <option value="alfalfa">Alfalfa</option>
      </select>
    </label>
    <label>Photos (1 to 10)
      <input type="file" name="files" accept="image/*,.heic,.heif" multiple required>
    </label>
    <button type="submit">Scout this field</button>
    <p id="form-error" class="error" hidden></p>
  </form>
  <section id="samples" hidden>
    <h2>Or open a sample walk</h2>
    <ul id="sample-list"></ul>
  </section>
  <section id="progress" hidden>
    <h2>Agent steps</h2>
    <div id="stops"></div>
    <p id="status" class="muted"></p>
  </section>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

`scout/web/style.css`:
```css
body { font-family: system-ui, sans-serif; margin: 0; background: #fafaf7; color: #1f1f1c; }
main { max-width: 720px; margin: 0 auto; padding: 16px; }
.lede, .muted { color: #6b6a63; }
form { display: grid; gap: 12px; background: #fff; border: 1px solid #e3e1d9; border-radius: 10px; padding: 16px; }
label { display: grid; gap: 4px; font-weight: 500; }
select, input, button { font: inherit; }
button { padding: 10px 14px; border-radius: 8px; border: 1px solid #3b6d11; background: #3b6d11; color: #fff; cursor: pointer; }
.error { color: #a32d2d; margin: 0; }
#stops section { background: #fff; border: 1px solid #e3e1d9; border-radius: 10px; padding: 8px 16px; margin-bottom: 8px; }
#stops li.failed { color: #a32d2d; }
#stops li.done { color: #3b6d11; font-weight: 500; }
```

`scout/web/app.js`:
```js
const form = document.getElementById("walk-form");
const errorBox = document.getElementById("form-error");
const stopsBox = document.getElementById("stops");
const statusBox = document.getElementById("status");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function stopList(index) {
  let list = document.getElementById(`stop-${index}`);
  if (!list) {
    const section = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = `Stop ${index + 1}`;
    list = document.createElement("ol");
    list.id = `stop-${index}`;
    section.append(heading, list);
    stopsBox.appendChild(section);
  }
  return list;
}

function addLine(index, text, className) {
  const item = document.createElement("li");
  item.textContent = text;
  if (className) item.className = className;
  stopList(index).appendChild(item);
}

function follow(walkId) {
  document.getElementById("progress").hidden = false;
  statusBox.textContent = "Working…";
  const events = new EventSource(`/walks/${walkId}/events`);
  events.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type === "step") {
      addLine(event.stop, event.summary, event.ok ? "" : "failed");
    } else if (event.type === "stop_done") {
      addLine(event.stop, event.status === "ok" ? "Stop done" : `Stop ${event.status.replaceAll("_", " ")}`, "done");
    } else if (event.type === "report_ready") {
      events.close();
      window.location.href = `/walks/${walkId}/report`;
    } else if (event.type === "error") {
      events.close();
      statusBox.textContent = event.message;
    }
  };
  events.onerror = () => events.close();
}

form.addEventListener("submit", async (submitEvent) => {
  submitEvent.preventDefault();
  errorBox.hidden = true;
  const count = form.elements.files.files.length;
  if (count < 1 || count > 10) {
    showError("Pick 1 to 10 photos.");
    return;
  }
  const button = form.querySelector("button");
  button.textContent = "Uploading…";
  try {
    const response = await fetch("/walks", { method: "POST", body: new FormData(form) });
    const body = await response.json();
    if (!response.ok) {
      showError(body.detail || "Upload failed. Try again.");
      return;
    }
    follow(body.walk_id);
  } catch {
    showError("Couldn't reach the server. Try again.");
  } finally {
    button.textContent = "Scout this field";
  }
});

fetch("/samples")
  .then((response) => response.json())
  .then((samples) => {
    if (!samples.length) return;
    const list = document.getElementById("sample-list");
    for (const sample of samples) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/walks/${sample.walk_id}/report`;
      link.textContent = sample.name;
      item.appendChild(link);
      list.appendChild(item);
    }
    document.getElementById("samples").hidden = false;
  })
  .catch(() => {});
```

- [ ] **Step 2: Commit**

```bash
git add scout/web
git commit -m "feat: add upload page with live agent steps"
```

### Task 29: FastAPI app

**Files:**
- Create: `scout/app/main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

`tests/test_main.py`:
```python
from fastapi.testclient import TestClient

from scout.app.main import create_app
from scout.config import Settings
from scout.models import FieldReport
from tests.helpers import png_bytes, small_weedbook


class FakePipeline:
    def __init__(self, root):
        self.weeds = small_weedbook()
        self.root = root

    def find_dir(self, walk_id):
        folder = self.root / walk_id
        return folder if (folder / "report.json").exists() else None

    def cached(self, walk_id):
        folder = self.find_dir(walk_id)
        return FieldReport.model_validate_json((folder / "report.json").read_text()) if folder else None

    def samples(self):
        return [{"name": "Sample", "walk_id": "0123456789abcdef"}]

    def run(self, walk, images):
        report = FieldReport(walk_id=walk.id, crop=walk.crop, stops=[], species_summary=[], decision="no_action",
                             rule_trace=["No weeds found at any stop -> no action"],
                             explanation="Nothing to do. Read the label before use.", options=[])
        folder = self.root / walk.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(report.model_dump_json())
        walk.emit({"type": "step", "stop": 0, "tool": "segment", "summary": "fake", "ok": True})
        walk.report = report
        walk.emit({"type": "report_ready"})
        walk.status = "done"


def _client(tmp_path, **settings):
    return TestClient(create_app(FakePipeline(tmp_path), Settings(api_key="t", **settings)))


def _upload(client, n=1, crop="soybean", shade=0):
    files = [("files", (f"p{i}.png", png_bytes(40, 40, (i, 100, shade)), "image/png")) for i in range(n)]
    return client.post("/walks", data={"crop": crop}, files=files)


def test_walk_flow(tmp_path):
    client = _client(tmp_path)
    walk_id = _upload(client).json()["walk_id"]
    body = client.get(f"/walks/{walk_id}/events").text
    assert '"type": "step"' in body and "report_ready" in body
    assert "Read the label before use" in client.get(f"/walks/{walk_id}/report").text
    assert client.get(f"/walks/{walk_id}/stops.csv").headers["content-type"].startswith("text/csv")
    assert client.get(f"/walks/{walk_id}/report.json").json()["decision"] == "no_action"


def test_rejects_bad_crop_and_too_many_photos(tmp_path):
    client = _client(tmp_path)
    assert _upload(client, crop="rice").status_code == 400
    assert _upload(client, n=11).status_code == 400


def test_rate_limit_per_ip(tmp_path):
    client = _client(tmp_path, walks_per_ip_per_hour=1)
    assert _upload(client, shade=1).status_code == 200
    assert _upload(client, shade=2).status_code == 429


def test_unknown_walk_is_404_and_samples_listed(tmp_path):
    client = _client(tmp_path)
    assert client.get("/walks/ffffffffffffffff/report").status_code == 404
    assert client.get("/walks/ffffffffffffffff/overlay/0.png").status_code == 404
    assert client.get("/samples").json()[0]["name"] == "Sample"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.app.main'`

- [ ] **Step 3: Write the implementation**

`scout/app/main.py`:
```python
"""HTTP app. Run locally: uvicorn scout.app.main:create_app --factory --port 8000"""
import asyncio
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from scout.app.ingest import load_image
from scout.app.pipeline import Pipeline, Walk, walk_id_for
from scout.config import Settings, get_settings
from scout.models import FieldReport
from scout.report.render import report_csv, report_html

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
CROPS = {"corn", "soybean", "alfalfa"}


def create_app(pipeline=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    pipeline = pipeline or Pipeline.from_settings(settings)
    app = FastAPI(title="Weed scout")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    walks: dict[str, Walk] = {}
    recent: dict[str, list[float]] = defaultdict(list)

    def report_for(walk_id: str) -> FieldReport:
        walk = walks.get(walk_id)
        report = walk.report if walk and walk.report else pipeline.cached(walk_id)
        if report is None:
            raise HTTPException(404, "No report for that walk.")
        return report

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/samples")
    def samples():
        return pipeline.samples()

    @app.post("/walks")
    async def create_walk(request: Request, crop: str = Form(...), files: list[UploadFile] = File(...)):
        if crop not in CROPS:
            raise HTTPException(400, "Pick corn, soybean, or alfalfa.")
        if not 1 <= len(files) <= settings.max_photos:
            raise HTTPException(400, f"Upload 1 to {settings.max_photos} photos.")
        images = []
        for index, upload in enumerate(files):
            data = await upload.read()
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(400, f"{upload.filename} is over {settings.max_upload_bytes // 1_000_000} MB.")
            try:
                images.append(load_image(index, data, settings.full_max_px, settings.work_max_px))
            except Exception:
                raise HTTPException(400, f"{upload.filename} isn't a photo we can read.") from None
        walk_id = walk_id_for(crop, images)
        if walk_id in walks and walks[walk_id].status != "failed":
            return {"walk_id": walk_id}
        if pipeline.cached(walk_id) is None:
            ip = request.client.host if request.client else "unknown"
            now = time.time()
            recent[ip] = [t for t in recent[ip] if now - t < 3600]
            if len(recent[ip]) >= settings.walks_per_ip_per_hour:
                raise HTTPException(429, "You've reached the hourly limit. Open a sample walk instead.")
            recent[ip].append(now)
        walk = Walk(id=walk_id, crop=crop)
        walks[walk_id] = walk
        threading.Thread(target=pipeline.run, args=(walk, images), daemon=True).start()
        return {"walk_id": walk_id}

    @app.get("/walks/{walk_id}/events")
    async def events(walk_id: str):
        walk = walks.get(walk_id)
        if walk is None:
            if pipeline.cached(walk_id) is None:
                raise HTTPException(404, "Unknown walk.")
            walk = Walk(id=walk_id, crop="", status="done", events=[{"type": "report_ready"}])

        async def stream():
            sent = 0
            while True:
                done = walk.status != "running"  # read first: events are appended before status changes
                while sent < len(walk.events):
                    yield f"data: {json.dumps(walk.events[sent])}\n\n"
                    sent += 1
                if done:
                    break
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/walks/{walk_id}/report", response_class=HTMLResponse)
    def report_page(walk_id: str):
        return report_html(report_for(walk_id), pipeline.weeds, lambda i: f"/walks/{walk_id}/overlay/{i}.png")

    @app.get("/walks/{walk_id}/report.json")
    def report_json(walk_id: str):
        return report_for(walk_id).model_dump(mode="json")

    @app.get("/walks/{walk_id}/stops.csv")
    def stops_csv(walk_id: str):
        return Response(report_csv(report_for(walk_id)), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="walk-{walk_id}.csv"'})

    @app.get("/walks/{walk_id}/overlay/{index}.png")
    def overlay(walk_id: str, index: int):
        folder = pipeline.find_dir(walk_id)
        path = folder / f"stop_{index}.png" if folder else None
        if path is None or not path.exists():
            raise HTTPException(404, "No such overlay.")
        return FileResponse(path, media_type="image/png")

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_main.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite and commit**

Run: `python -m pytest -q`
Expected: all tests pass

```bash
git add scout/app/main.py tests/test_main.py
git commit -m "feat: add FastAPI routes with SSE progress"
```

### Task 30: End-to-end run on a real walk

**Files:**
- Create: `.claude/launch.json`

- [ ] **Step 1: Add a launch config for the preview browser**

`.claude/launch.json`:
```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "scout",
      "runtimeExecutable": ".venv/Scripts/python",
      "runtimeArgs": ["-m", "uvicorn", "scout.app.main:create_app", "--factory", "--port", "8000"],
      "port": 8000
    }
  ]
}
```

- [ ] **Step 2: Start the app and run a walk**

Run: `python -m uvicorn scout.app.main:create_app --factory --port 8000`
Then:
1. Open http://localhost:8000 and pick the crop.
2. Upload the photos in `field-images/walk1/`.

Expected:
- Steps stream per stop, and the browser then opens the report.
- The report shows overlays, metric cards, a decision with a rule trace, and options with working `p. N` links.
- If GPS was present, map pins appear.

- [ ] **Step 3: Fix what breaks**

For any failure, use superpowers:systematic-debugging. Add a regression test in the matching `tests/test_*.py` before fixing. Commit each fix separately.

- [ ] **Step 4: Commit**

```bash
git add .claude/launch.json
git commit -m "chore: add local launch config"
```

---

# Day 5: Tue Nov 3. Samples, deploy, check, video

### Task 31: Sample walks served from cache

**Files:**
- Create: `scripts/make_sample.py`, `demo/` (generated)

- [ ] **Step 1: Write the script**

`scripts/make_sample.py`:
```python
"""Run a walk locally and save it as a sample the deployed app serves without API calls.

GPS is shifted by one random offset per sample, so pins keep their layout but not the farm's location.
Usage: python scripts/make_sample.py --name "Soybean with waterhemp" --crop soybean field-images/walk1/*
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from scout.app.ingest import load_image
from scout.app.pipeline import DEMO_DIR, Pipeline, Walk, walk_id_for
from scout.config import get_settings
from scout.models import FieldReport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--name", required=True)
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    args = parser.parse_args()
    settings = get_settings()
    pipeline = Pipeline.from_settings(settings)
    images = [load_image(i, Path(p).read_bytes(), settings.full_max_px, settings.work_max_px)
              for i, p in enumerate(args.photos)]
    walk = Walk(id=walk_id_for(args.crop, images), crop=args.crop)
    pipeline.run(walk, images)
    if walk.status != "done":
        raise SystemExit(f"Walk failed: {walk.error}")
    target = DEMO_DIR / walk.id
    shutil.copytree(pipeline.find_dir(walk.id), target, dirs_exist_ok=True)
    report = FieldReport.model_validate_json((target / "report.json").read_text(encoding="utf-8"))
    d_lat, d_lon = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
    for stop in report.stops:
        if stop.gps:
            stop.gps = (round(stop.gps[0] + d_lat, 6), round(stop.gps[1] + d_lon, 6))
    (target / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    index = DEMO_DIR / "samples.json"
    samples = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    samples = [s for s in samples if s["walk_id"] != walk.id] + [{"name": args.name, "walk_id": walk.id}]
    index.write_text(json.dumps(samples, indent=1), encoding="utf-8")
    print(f"Saved sample '{args.name}' -> /walks/{walk.id}/report")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make two samples**

Run:
```bash
python scripts/make_sample.py --name "Soybean walk" --crop soybean field-images/walk1/*
python scripts/make_sample.py --name "Corn walk" --crop corn field-images/walk2/*
```
Expected: two "Saved sample" lines, and `demo/` holding two folders plus `samples.json`.

- [ ] **Step 3: [USER] Review and commit**

Open each sample report locally and confirm nothing private shows: no house or person visible in the overlays, and the pins are offset. Then:
```bash
git add scripts/make_sample.py demo
git commit -m "feat: add cached sample walks"
```

### Task 32: [USER] Public repo and deployment

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `README.md`

- [ ] **Step 1: Write the Dockerfile**

`Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scout ./scout
COPY demo ./demo
ENV SCOUT_STORE=/tmp/store
EXPOSE 7860
CMD ["uvicorn", "scout.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
```

`.dockerignore`:
```
.venv
.git
field-images
store
spike_out
run_out
scout/labels/pdfs
tests
evaluation
docs
```

- [ ] **Step 2: Write the README**

`README.md`. The YAML header is required by Hugging Face Spaces.
```markdown
---
title: Weed scout
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Weed scout

Upload the photos from one walk through one field. A Muse Spark agent drives SAM 3.1 to separate weeds from crop at every stop, code measures weed intensity and applies threshold rules, and the report lists herbicide options whose EPA label covers this crop, this weed, and this weed size, each with a page citation.

**Built during the Meta Global AI Developer Hackathon (Oct 30 – Nov 4, 2026).** Everything in this repository was written during the event. Earlier projects by the author ([Kronia](https://github.com/AjjayK/Kronia-Pesticide-LLM) and [sf-epa-data-ingestion](https://github.com/AjjayK/sf-epa-data-ingestion)) inspired the label idea; no code was reused.

## How it works
1. **Segment.** Muse Spark asks SAM 3.1 for the crop ("soybean plant") and for all plants ("plant"). Plant minus crop gives weed candidates.
2. **Check.** The agent zooms into doubtful candidates, re-prompts SAM with points, and names each weed and its size.
3. **Measure.** Code computes weed cover, counts, density (straight-down shots), and in-row vs between-row share. The model never writes a number.
4. **Decide.** Rules compare species against sourced thresholds (zero tolerance for resistant amaranths), and choose no action, scout again, or treat (spot or broadcast).
5. **Ground in labels.** A table built from EPA label PDFs (each record's quote verified on its page) lists products labeled for this crop, weed, and size.

## Meta models used
- `muse-spark-1.3`: agent tool loop, zoom inspection, label extraction to structured JSON, field explanation.
- `sam-3.1`: concept prompts, point and box prompts, confidence scores.

## Run locally
    python -m venv .venv && source .venv/Scripts/activate
    pip install -r requirements.txt
    cp .env.example .env   # add MODEL_API_KEY
    python -m pytest -q
    python -m uvicorn scout.app.main:create_app --factory --port 8000

## Limits
Decision support, not a prescription. Read the label before use; the label is the law. Density is only reported for straight-down photos. Thresholds marked "tool default" are not from an agronomic source.
```

- [ ] **Step 3: Check the Docker image**

Run: `docker build -t weed-scout . && docker run --rm -p 7860:7860 -e MODEL_API_KEY=$(grep MODEL_API_KEY .env | cut -d= -f2) weed-scout`
Expected: http://localhost:7860 serves the page, and the sample walks open. Skip this step if Docker isn't installed; Spaces builds the same file.

- [ ] **Step 4: [USER] Create the public GitHub repo and push**

Confirm with the user before running. This publishes the code.
```bash
git add Dockerfile .dockerignore README.md
git commit -m "chore: add Dockerfile and README"
gh repo create weed-or-wheat --public --source . --push
```

- [ ] **Step 5: [USER] Create the Hugging Face Space**

1. On huggingface.co, create a Space with SDK **Docker** (blank template).
2. In the Space settings, add the secret `MODEL_API_KEY`.
3. Push the code. Confirm with the user before pushing.
```bash
git remote add space https://huggingface.co/spaces/<hf-username>/weed-scout
git push space main
```
Expected: the Space builds and shows the page; the samples open without spending credits; one fresh walk completes.

### Task 33: Agreement check

**Files:**
- Create: `evaluation/__init__.py`, `evaluation/run_eval.py`, `evaluation/labels.csv`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
from evaluation.run_eval import bucket, score


def test_bucket_edges():
    assert bucket(1.99) == "<2" and bucket(2) == "2-10" and bucket(10) == "2-10" and bucket(10.5) == ">10"


def test_score_skips_blank_species():
    rows = [
        {"weeds_present": "y", "pred_weeds_present": "y", "main_species": "waterhemp",
         "pred_main_species": "waterhemp", "cover_bucket": "<2", "pred_cover_bucket": "2-10"},
        {"weeds_present": "n", "pred_weeds_present": "y", "main_species": "", "pred_main_species": "",
         "cover_bucket": "<2", "pred_cover_bucket": "<2"},
    ]
    assert score(rows) == {"weeds_present": 0.5, "main_species": 1.0, "cover_bucket": 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Write the implementation**

Create an empty `evaluation/__init__.py`.

`evaluation/run_eval.py`:
```python
"""Compare the stop agent with your own quick labels.

Usage: python -m evaluation.run_eval   (reads evaluation/labels.csv, writes evaluation/results.csv)
"""
import csv
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def bucket(cover_pct: float) -> str:
    return "<2" if cover_pct < 2 else ("2-10" if cover_pct <= 10 else ">10")


def score(rows: list[dict]) -> dict[str, float | None]:
    fields: dict[str, list[bool]] = {"weeds_present": [], "main_species": [], "cover_bucket": []}
    for row in rows:
        fields["weeds_present"].append(row["weeds_present"] == row["pred_weeds_present"])
        if row["main_species"]:
            fields["main_species"].append(row["main_species"] == row["pred_main_species"])
        fields["cover_bucket"].append(row["cover_bucket"] == row["pred_cover_bucket"])
    return {key: round(sum(values) / len(values), 2) if values else None for key, values in fields.items()}


def main() -> None:
    from scout.agent.llm import Llm
    from scout.agent.stop_agent import run_stop
    from scout.agronomy.weeds import WeedBook
    from scout.app.ingest import load_image
    from scout.config import get_settings
    from scout.vision.sam_client import SamClient

    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()
    with open(HERE / "labels.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image = load_image(0, (ROOT / row["photo"]).read_bytes(), settings.full_max_px, settings.work_max_px)
        result, _ = run_stop(image, row["crop"], llm, sam, weeds, settings)
        covers = result.species_cover_pct
        row["pred_weeds_present"] = "y" if result.weed_count else "n"
        row["pred_main_species"] = max(covers, key=covers.get) if covers else ""
        row["pred_cover_bucket"] = bucket(result.weed_cover_pct)
        print(f"{row['photo']}: {row['pred_weeds_present']} {row['pred_main_species']} {row['pred_cover_bucket']}")
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(score(rows))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval.py -v`
Expected: 2 passed

- [ ] **Step 5: [USER] Label ~15 photos and run**

Create `evaluation/labels.csv`. Use `species_id` values from `weeds.yaml` for `main_species`, or leave it blank if unsure.
```
photo,crop,weeds_present,main_species,cover_bucket
field-images/IMG_4215.JPG,corn,y,giant_foxtail,2-10
```
Add ~15 rows, then run: `python -m evaluation.run_eval`
Expected: one line per photo, then a dict of agreement rates. Use misses to tune `prompts.py`, rerun, and note the final numbers in `docs/spike-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add evaluation tests/test_eval.py docs/spike-notes.md scout/agent/prompts.py
git commit -m "feat: add agreement check against hand labels"
```

### Task 34: [USER] Demo video and submission

**Files:**
- Create: `docs/demo-script.md`

- [ ] **Step 1: Write the 3-minute script**

`docs/demo-script.md`:
```markdown
# Demo video (3:00)

0:00–0:20  Problem: scouting means walking a field, counting weeds per stop, and matching products to labels by hand.
0:20–1:20  Live: upload a 6-photo soybean walk on the deployed URL. Agent steps stream: segment crop, segment
           plants, zoom on a doubtful plant, identify waterhemp ~3 in, measure.
1:20–2:10  Report: overlays (crop green, weeds red), cover/density/frequency cards, decision "Treat (spot)",
           rule trace, label options; click "p. N" to open the EPA label at the cited page.
2:10–2:45  Under the hood: Muse Spark <-> SAM 3.1 loop (concept prompts, point prompts, confidence); code computes
           every number; label table built from EPA PDFs with verified quotes. Show the agreement rate.
2:45–3:00  Impact and next: insects and disease severity, video row walks. "Built during the hackathon."
```

- [ ] **Step 2: [USER] Record and upload**

Record the screen and voice following the script, upload the video (YouTube unlisted, or as the platform requires), and check that the link plays while logged out.

- [ ] **Step 3: [USER] Submit by noon ET Wed Nov 4**

Check:
- [ ] The public GitHub repo opens while logged out.
- [ ] The live Space URL works, and the samples open.
- [ ] The video link works.
- [ ] The submission names the track (Indie) and the theme (Build for Your World).
- [ ] Optional: post the demo on X and submit the link for the Most Viral bonus.

- [ ] **Step 4: Commit**

```bash
git add docs/demo-script.md
git commit -m "docs: add demo video script"
git push
```

---

## Cut order if behind schedule (from the spec)

Cut in this order: CSV export (Task 27 CSV route), then GPS map (template block), then in-row/between-row split (`row_band`), then straight-down density.

Never cut: overlays, agent step log, decision with rule trace, cited options, live URL, video.

## Stretch goals (only after Task 30 works end to end)

- **Video row walk:** SAM 3.1 tracking for weeds per meter (sparse weeds only; 16 objects per frame).
- **Density from angled shots:** use 30-inch row spacing as a scale.
- **Disease severity:** lesion area ÷ leaf area on close-ups.
- **Spray conditions:** a check from a weather API.
