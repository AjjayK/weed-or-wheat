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
- **Scripts:** run `scripts/*.py` from the repo root with the repo on the import path: `PYTHONPATH=. python scripts/x.py` in Git Bash, or `$env:PYTHONPATH="."` first in PowerShell.
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
  - The payload is base85 over printable ASCII minus `" \ , ; < > |`, so `;` always separates fields and `|` ends the token. Fields are read by name (order doesn't matter) and whitespace around keys and values is trimmed.
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
2. `Instance.source` gains `"candidate"`. It was for pieces derived as plant minus crop; since the boxes redesign (item 9) it is unused but stays in the Literal.
3. Tools changed:
   - `segment` takes a `kind` (`crop` or `weed`).
   - `mark(kind, boxes)` replaces `segment_at`.
   - `measure(view_type)` sets the view type, and `finish(crop_stage, notes)` no longer takes one.
   - `classify` accepts a batch of items.
4. `StopResult` gains `species_cover_pct`, `species_density`, and `species_max_height_in`. The rules need per-species values.
5. `UseRecord` gains `pdf_file`, so the report can link straight to the label page. `FieldReport.options` is `list[LabelOption]`, where `LabelOption` wraps a record with an `over_size_limit` flag.
6. Zoom crops always go in a user message, since the docs say images are only processed there.
7. Offline tests build realistic SAM text with the parser's own encoder instead of recorded fixtures. Spike outputs go to the git-ignored `spike_out/`.
8. Sample walks are run once, then committed as cached reports in `demo/`. The live site can then show them without spending API credits. GPS in samples is shifted by a random offset.
9. **Boxes redesign** (after the live tests; `docs/superpowers/specs/2026-10-30-boxes-redesign.md`):
   - Weeds are found by Muse Spark boxing each plant in 0–1000 coordinates. SAM segments all of a call's boxes in one request (`SamClient.segment_boxes`). There are no plant-minus-crop candidates.
   - Weeds win over crop in the overlay and the measurements.
   - The in-row zone is the row canopy when a crop mask spans a third of the image, else fitted rows, and none for alfalfa.
   - `stop_timeout_s` is 240.

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
scout/vision/sam_client.py   SAM requests (concept, point/box, multi-box), retry, cache
scout/vision/overlay.py      Colored overlay image
scout/vision/rows.py         Crop-row fitting and in-row band
scout/vision/measure.py      Metrics from instances + masks
scout/agent/llm.py           Responses API wrapper: LlmTurn/ToolCall, user_message
scout/agent/state.py         StopState: 0-1000 box conversion, weed check, in-row zone, step log, overlay
scout/agent/tools.py         Tool schemas + ToolRunner
scout/agent/prompts.py       Agent instructions
scout/agent/stop_agent.py    Tool loop for one stop + finalize
scout/agent/field_agent.py   Field decision, label options, checked explanation
scout/agronomy/weeds.yaml    36 entries (34 species + 2 catch-alls) with names, synonyms, thresholds, sources
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

**Amended for the boxes redesign:** `stop_timeout_s` is 240 (it was 90), because Muse Spark takes 13–16 s per turn. Change the committed `scout/config.py` to match.

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
meta-sam-parser==0.1.0
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

`tests/conftest.py`:
```python
"""Test-wide fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    """Tests never call the network, so a real key from .env must not be visible to them."""
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
```

`tests/test_config.py`:
```python
from scout.config import Settings, get_settings


def test_get_settings_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "k-123")
    monkeypatch.setenv("AGENT_MODEL", "muse-spark-1.2")
    monkeypatch.delenv("SAM_MODEL", raising=False)
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
    stop_timeout_s: float = 240.0  # Muse Spark takes 13-16 s per turn
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
import httpx
import openai
import pytest

from scout.retry import BadRequest, with_retry

_REQUEST = httpx.Request("POST", "https://api.meta.ai/v1/responses")


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _raiser(error):
    def fn():
        raise error
    return fn


def test_retries_transient_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise StatusError(503)
        return "ok"

    assert with_retry(fn, sleep=sleeps.append) == "ok"
    assert sleeps == [1, 2]


def test_rate_limit_waits_four_times_longer():
    sleeps = []

    def fn():
        raise StatusError(429)

    with pytest.raises(StatusError):
        with_retry(fn, retries=2, sleep=sleeps.append)
    assert sleeps == [4, 8]


@pytest.mark.parametrize("status", [400, 413, 422])
def test_rejected_request_is_not_retried(status):
    calls = []

    def fn():
        calls.append(1)
        raise StatusError(status)

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


def test_sdk_connection_error_is_retried():
    sleeps = []
    with pytest.raises(openai.APIConnectionError):
        with_retry(_raiser(openai.APIConnectionError(request=_REQUEST)), retries=1, sleep=sleeps.append)
    assert sleeps == [1]


def test_sdk_timeout_is_not_retried():
    sleeps = []
    with pytest.raises(openai.APITimeoutError):
        with_retry(_raiser(openai.APITimeoutError(request=_REQUEST)), sleep=sleeps.append)
    assert sleeps == []


def test_sdk_bad_request_becomes_bad_request():
    error = openai.BadRequestError("rejected", response=httpx.Response(400, request=_REQUEST), body=None)
    with pytest.raises(BadRequest):
        with_retry(_raiser(error), sleep=lambda s: None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.retry'`

- [ ] **Step 3: Write the implementation**

`scout/retry.py`:
```python
"""Retry policy shared by the SAM and Muse Spark clients.

The OpenAI clients are built with max_retries=0, so this is the only retry layer.
"""
import time
from typing import Callable, TypeVar

T = TypeVar("T")

RETRY_STATUS = {429, 500, 502, 503, 504}
REJECTED_STATUS = {400, 413, 422}
# Timeouts are not retried: a call that already used its whole timeout would not fit in the stop's budget twice.
TRANSIENT_ERRORS = {"APIConnectionError"}


class BadRequest(Exception):
    """The API rejected the request (HTTP 400, 413 or 422). Retrying the same request will not help."""


def with_retry(fn: Callable[[], T], retries: int = 3, sleep: Callable[[float], None] = time.sleep) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status in REJECTED_STATUS:
                raise BadRequest(str(error)) from error
            transient = status in RETRY_STATUS or type(error).__name__ in TRANSIENT_ERRORS
            if not transient or attempt == retries:
                raise
            sleep(2 ** attempt * (4 if status == 429 else 1))  # rate limits clear slowly when 4 stops run at once
    raise AssertionError("unreachable")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retry.py -v`
Expected: 10 passed

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


def test_pieces_drops_slivers_along_excluded_edges():
    crop = np.zeros((60, 60), bool)
    crop[10:40, 10:40] = True
    plant = np.zeros((60, 60), bool)
    plant[8:42, 8:42] = True
    assert mk.pieces(plant, min_area=5, exclude=crop) == []


def test_pieces_keeps_thin_blades():
    blade = np.zeros((60, 60), bool)
    for y in range(5, 55):
        blade[y, y - 2:y + 1] = True
    assert [int(p.sum()) for p in mk.pieces(blade, min_area=20)] == [int(blade.sum())]


def test_pad_box_pads_thin_boxes_on_both_axes():
    assert mk.pad_box((50, 10, 53, 110), 0.15, (200, 200)) == (35, 0, 68, 125)


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

_EIGHT = np.ones((3, 3), dtype=bool)


def paste(local: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    """Place a box-local mask onto a full canvas, clipping anything outside it."""
    width, height = size
    canvas = np.zeros((height, width), dtype=bool)
    left, top, right, bottom = box
    if local.shape != (bottom - top, right - left):
        raise ValueError(f"mask shape {local.shape} does not match box {box}")
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


def pieces(mask: np.ndarray, min_area: float, exclude: np.ndarray | None = None, margin: int = 2) -> list[np.ndarray]:
    """Split a mask into 8-connected pieces of at least min_area pixels.

    Pixels within `margin` px of `exclude` are dropped first. That removes the slivers left where the plant and
    crop masks disagree slightly, without the erosion that would also erase thin grass blades elsewhere.
    """
    if exclude is not None:
        mask = mask & ~ndimage.binary_dilation(exclude, structure=_EIGHT, iterations=margin)
    labels, count = ndimage.label(mask, structure=_EIGHT)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    return [labels == index for index in range(1, count + 1) if areas[index] >= min_area]


def bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def pad_box(box: Box, frac: float, size: tuple[int, int]) -> Box:
    left, top, right, bottom = box
    pad = int(max(right - left, bottom - top) * frac)  # same pad on both axes, so a thin blade's box still widens
    width, height = size
    return (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_masks.py -v`
Expected: 8 passed

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
from tests.helpers import sam_record_text, sam_text


def _corrupt(record):
    """Change the payload's last digit so the strict one_bit decode rejects it."""
    return record[:-3] + ("A" if record[-3] != "A" else "B") + record[-2:]


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
    assert record.mask.shape == (4, 4) and record.mask.all() and record.resized


def test_resize_keeps_orientation():
    pattern = np.array([[1, 0, 0], [0, 0, 1]], bool)
    record = parse_sam_output(sam_text([("0", (0, 0, 6, 4), pattern)], (10, 10)))[0]
    assert record.mask.shape == (4, 6) and record.mask[0, 0] and record.mask[3, 5] and not record.mask[0, 5]


def test_empty_output_gives_no_records():
    assert parse_sam_output("") == []


def test_bad_record_is_skipped_and_logged(caplog):
    good = sam_record_text("1", (0, 0, 3, 3), np.ones((3, 3), bool), (10, 10))
    bad = _corrupt(sam_record_text("0", (5, 5, 9, 9), np.eye(4, dtype=bool), (10, 10)))
    assert [r.object_id for r in parse_sam_output("<0f>" + bad + "," + good)] == ["1"]
    assert "kept 1 records" in caplog.text


def test_truncated_last_record_is_counted(caplog):
    text = sam_text([("0", (0, 0, 3, 3), np.ones((3, 3), bool)), ("1", (4, 4, 7, 7), np.ones((3, 3), bool))], (10, 10))
    assert [r.object_id for r in parse_sam_output(text[:-2])] == ["0"]
    assert "1 record(s) not in" in caplog.text


def test_fields_in_any_order_with_spaces():
    payload = sam_record_text("0", (2, 2, 5, 5), np.ones((3, 3), bool), (10, 10)).split("data=3,3,")[1][:-2]
    text = f"<0f>0<|box; y1 = 2 ;x1=2;x2=4;y2=4;w=10;h=10|><|mask;data=3,3,{payload}; c = 0.7 ;x=0;y=0|>"
    record = parse_sam_output(text)[0]
    assert record.box == (2, 2, 5, 5) and record.confidence == 0.7 and record.mask.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.sam_parse'`

- [ ] **Step 3: Write the implementation**

`scout/vision/sam_parse.py`:
```python
"""Parse SAM 3.1 output text into records with box-local boolean masks.

Record grammar: ID<|box;x1=;y1=;x2=;y2=;w=;h=[;c=]|><|mask;x=0;y=0[;c=];data=H,W,<payload>|>
x2/y2 are inclusive. The mask raster covers the box. Fields are read by name, and the base85 payload never
contains , ; < > or |, so ';' always separates fields.
"""
import logging
import re
from dataclasses import dataclass

import numpy as np
from meta_sam_parser import InvalidSegmentationMaskError, SegmentationMask, decode_mask_to_raster
from PIL import Image

from scout.models import Box
from scout.vision.masks import paste

log = logging.getLogger(__name__)

_RECORD = re.compile(r"([A-Za-z0-9_-]+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_DATA = re.compile(r"(\d+),(\d+),([!~][^|]+)")


@dataclass
class SamRecord:
    object_id: str
    box: Box
    confidence: float | None
    mask: np.ndarray  # bool, shaped (box height, box width)
    resized: bool = False  # True when SAM's raster size differed from the box and was stretched to fit

    def full_mask(self, size: tuple[int, int]) -> np.ndarray:
        return paste(self.mask, self.box, size)


def _fields(body: str) -> dict[str, str]:
    out = {}
    for part in body.split(";"):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _record(object_id: str, box_body: str, mask_body: str) -> SamRecord:
    box, mask = _fields(box_body), _fields(mask_body)
    parsed = _DATA.fullmatch(mask.get("data", ""))
    if parsed is None:
        raise ValueError("mask has no H,W,payload data")
    height, width, payload = int(parsed[1]), int(parsed[2]), parsed[3]
    encoding = "lossless" if payload.startswith("~") else "one_bit"
    raster = decode_mask_to_raster(SegmentationMask(encoding=encoding, payload=payload, width=width, height=height))
    local = np.frombuffer(raster, dtype=np.uint8).reshape(height, width).astype(bool)
    left, top = int(box["x1"]), int(box["y1"])
    right, bottom = int(box["x2"]) + 1, int(box["y2"]) + 1
    resized = local.shape != (bottom - top, right - left)
    if resized:
        local = np.array(Image.fromarray(local).resize((right - left, bottom - top), Image.NEAREST)).astype(bool)
    confidence = box.get("c") or mask.get("c")
    return SamRecord(object_id, (left, top, right, bottom), float(confidence) if confidence else None, local, resized)


def parse_sam_output(text: str) -> list[SamRecord]:
    """Records that fail to parse or decode are skipped and logged, so one bad record cannot sink the call."""
    records, errors = [], []
    for match in _RECORD.finditer(text):
        try:
            records.append(_record(*match.groups()))
        except (InvalidSegmentationMaskError, KeyError, ValueError) as error:
            errors.append(f"record {match[1]}: {type(error).__name__}: {error}")
    unmatched = text.count("<|box") - len(records) - len(errors)
    if unmatched > 0:
        errors.append(f"{unmatched} record(s) not in ID<|box;...|><|mask;...|> form")
    if errors:
        log.warning("SAM output: kept %d records, skipped: %s", len(records), errors[:3])
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam_parse.py -v`
Expected: 7 passed

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

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

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


def test_load_image_reads_gps_from_exif():
    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: "S", 2: (IFDRational(41, 1), IFDRational(30, 1), IFDRational(0, 1)),
                                 3: "W", 4: (IFDRational(93, 1), IFDRational(36, 1), IFDRational(0, 1))})
    buf = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buf, format="JPEG", exif=exif)
    assert load_image(0, buf.getvalue()).gps == (-41.5, -93.6)


def test_gps_from_ifd_rejects_unknown_and_impossible_values():
    unknown = IFDRational(0, 0)  # Pillow reads 0/0 as NaN
    assert gps_from_ifd({1: "N", 2: (unknown,) * 3, 3: "W", 4: (unknown,) * 3}) is None
    assert gps_from_ifd({1: "N", 2: (0.0, 0.0, 0.0), 3: "E", 4: (0.0, 0.0, 0.0)}) is None
    assert gps_from_ifd({1: "N", 2: (95.0, 0.0, 0.0), 3: "W", 4: (93.0, 0.0, 0.0)}) is None


def test_heic_is_rotated_exactly_once():
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buf, format="HEIF", exif=exif.tobytes())
    assert load_image(0, buf.getvalue()).work.size == (20, 40)


def test_small_rgba_image_is_not_upscaled_and_becomes_rgb():
    buf = io.BytesIO()
    Image.new("RGBA", (50, 30)).save(buf, format="PNG")
    image = load_image(0, buf.getvalue())
    assert image.full.size == image.work.size == (50, 30) and image.work.mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image", png_bytes(500, 500)[:100]])
def test_unreadable_bytes_raise(data):
    with pytest.raises((OSError, ValueError)):
        load_image(0, data)


def test_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setattr("scout.app.ingest.MAX_PIXELS", 99)
    with pytest.raises(ValueError):
        load_image(0, png_bytes(10, 10))


def test_rejects_formats_outside_jpeg_png_heic():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="TIFF")
    with pytest.raises(OSError):
        load_image(0, buf.getvalue())
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
import math
from dataclasses import dataclass
from functools import cached_property

import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

GPS_IFD = 0x8825
FORMATS = ("JPEG", "PNG", "HEIF")  # JPEG also opens iPhone/DJI MPO files
MAX_PIXELS = 64_000_000  # 48-64 MP phone photos; refuses decompression bombs long before Pillow's 179 MP error


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
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180 or lat == lon == 0:
        return None  # 0/0 rationals read as NaN; (0, 0) is a phone with no fix
    return (round(lat, 6), round(lon, 6))


def load_image(index: int, data: bytes, full_max_px: int = 2048, work_max_px: int = 1024) -> StopImage:
    """Raises (OSError, ValueError, or Pillow's DecompressionBombError) for bytes that aren't a usable photo."""
    image = Image.open(io.BytesIO(data), formats=FORMATS)
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"image is {image.width}x{image.height}; too many pixels")
    gps = gps_from_ifd(dict(image.getexif().get_ifd(GPS_IFD)))
    image.draft("RGB", (full_max_px, full_max_px))  # JPEG: decode big photos at 1/2 or 1/4 scale; no-op otherwise
    ImageOps.exif_transpose(image, in_place=True)  # pillow-heif already rotated HEIC and reset its Orientation tag
    if image.mode != "RGB":
        image = image.convert("RGB")
    full = _shrink(image, full_max_px)
    return StopImage(
        index=index,
        hash=hashlib.sha256(data).hexdigest()[:16],
        full=full,
        work=_shrink(full, work_max_px),
        gps=gps,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add scout/app/ingest.py tests/helpers.py tests/test_ingest.py
git commit -m "feat: ingest photos with rotation, GPS, and two sizes"
```

### Task 7: SAM client

**Files:**
- Create: `scout/vision/sam_client.py`
- Test: `tests/test_sam_client.py`

**Amended for the boxes redesign:** `segment_boxes` sends up to 16 boxes as one multi-object request. It and its two tests were added after this task was first committed; apply them to the committed files before Task 15.

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


def test_cache_is_per_image():
    calls = []
    sam = SamClient(lambda p: calls.append(p) or "", sleep=lambda s: None)
    sam.segment_concept(make_image(32, 32, color=(10, 200, 10)), "plant")
    sam.segment_concept(make_image(32, 32, color=(200, 10, 10)), "plant")
    assert len(calls) == 2


def test_cache_is_bounded():
    image, calls = make_image(32, 32), []
    sam = SamClient(lambda p: calls.append(p) or "", sleep=lambda s: None, max_entries=2)
    for phrase in ("a", "b", "c", "a"):
        sam.segment_concept(image, phrase)
    assert len(calls) == 4  # "a" was evicted by "c"
    sam.segment_concept(image, "a")
    assert len(calls) == 4


def test_segment_boxes_sends_all_boxes_in_one_request():
    image = make_image(100, 80)
    sent = []

    def send(payload):
        sent.append(payload)
        return sam_text([("1", (5, 6, 15, 16), np.ones((10, 10), bool), 0.9),
                         ("3", (60, 10, 70, 20), np.ones((10, 10), bool), 0.7)], image.size)

    sam = SamClient(send, sleep=lambda s: None)
    records = sam.segment_boxes(image, [(5, 6, 50, 60), (10.4, 10, 20, 30), (60, 10, 70, 20)])
    assert sent[0]["input"][0]["content"][0]["text"] == (
        "<0f>1<|box;x1=5;y1=6;x2=49;y2=59;w=100;h=80|>,2<|box;x1=10;y1=10;x2=19;y2=29;w=100;h=80|>,"
        "3<|box;x1=60;y1=10;x2=69;y2=19;w=100;h=80|>")
    assert [r.object_id for r in records] == ["1", "3"]  # SAM found nothing in box 2
    sam.segment_boxes(image, [(5, 6, 50, 60), (10, 10, 20, 30), (60, 10, 70, 20)])
    assert len(sent) == 1  # same boxes after int coercion: served from the cache


def test_segment_boxes_takes_1_to_16_boxes():
    sam = SamClient(lambda p: "", sleep=lambda s: None)
    with pytest.raises(ValueError):
        sam.segment_boxes(make_image(32, 32), [])
    with pytest.raises(ValueError):
        sam.segment_boxes(make_image(32, 32), [(0, 0, 4, 4)] * 17)
    assert sam.segment_boxes(make_image(32, 32), [(0, 0, 4, 4)] * 16) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.sam_client'`

- [ ] **Step 3: Write the implementation**

`scout/vision/sam_client.py`:
```python
"""SAM 3.1 requests over the Meta Model API: concept prompts and box/point prompts, cached per image."""
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Sequence

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt

from scout.app.ingest import StopImage
from scout.models import Box
from scout.retry import with_retry
from scout.vision.sam_parse import SamRecord, parse_sam_output

log = logging.getLogger(__name__)
MAX_OBJECTS = 16  # SAM 3.1 takes at most 16 objects in one box/point request


class SamClient:
    def __init__(self, send: Callable[[dict], str], model: str = "sam-3.1", sleep: Callable[[float], None] = time.sleep,
                 max_entries: int = 128):
        self._send = send
        self._model = model
        self._sleep = sleep
        # One client serves every walk for the life of the process, so both caches are bounded LRUs.
        self._max_entries = max_entries
        self._lock = threading.Lock()  # guards dict ops only; requests and decoding run unlocked
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._parsed: OrderedDict[tuple[str, str], list[SamRecord]] = OrderedDict()

    @classmethod
    def from_settings(cls, settings) -> "SamClient":
        import httpx
        from openai import OpenAI

        # max_retries=0 leaves with_retry as the only retry layer; connect=5 fails fast on an unreachable host.
        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                        timeout=httpx.Timeout(120.0, connect=5.0), max_retries=0)

        def send(payload: dict) -> str:
            response = client.responses.create(**payload)
            parts = [part for item in response.output if getattr(item, "type", None) == "message"
                     for part in getattr(item, "content", None) or []]
            if not any(getattr(part, "type", None) == "output_text" for part in parts):
                # An empty output_text part is SAM's normal "no match"; a missing one means an unexpected reply shape.
                log.warning("SAM reply had no output_text part (status=%s, items=%s)", response.status,
                            [getattr(item, "type", "?") for item in response.output])
            return response.output_text

        return cls(send, model=settings.sam_model)

    def _get(self, store: OrderedDict, key):
        with self._lock:
            if key in store:
                store.move_to_end(key)
                return store[key]
        return None

    def _put(self, store: OrderedDict, key, value):
        with self._lock:
            store[key] = value
            store.move_to_end(key)
            while len(store) > self._max_entries:
                store.popitem(last=False)
        return value

    def fetch_raw(self, image: StopImage, text: str) -> str:
        key = (image.hash, text)
        raw = self._get(self._cache, key)
        if raw is None:
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
            raw = self._put(self._cache, key, with_retry(lambda: self._send(payload), sleep=self._sleep))
        return raw

    def _records(self, image: StopImage, text: str) -> list[SamRecord]:
        """Parsed records are cached too, since decoding masks costs seconds of CPU."""
        key = (image.hash, text)
        records = self._get(self._parsed, key)
        if records is None:
            records = self._put(self._parsed, key, parse_sam_output(self.fetch_raw(image, text)))
        return records

    def segment_concept(self, image: StopImage, phrase: str) -> list[SamRecord]:
        return self._records(image, phrase.strip())

    def segment_at(self, image: StopImage, box: Box | None = None,
                   points: Sequence[tuple[int, int, bool]] = ()) -> list[SamRecord]:
        prompt = FramePrompt(
            frame=0,
            box=PromptBox(*(int(v) for v in box)) if box else None,
            points=[PromptPoint(int(x), int(y), label="positive" if positive else "negative") for x, y, positive in points],
        )
        text = build_object_prompt(size=image.size, objects=[PromptObject(id=1, prompts=[prompt])])
        return self._records(image, text)

    def segment_boxes(self, image: StopImage, boxes: Sequence[Box]) -> list[SamRecord]:
        """One multi-object request: object i + 1 is boxes[i]. A record's object_id names its box ("1" is boxes[0]);
        boxes SAM found nothing for are absent."""
        if not 1 <= len(boxes) <= MAX_OBJECTS:
            raise ValueError(f"segment_boxes takes 1 to {MAX_OBJECTS} boxes, got {len(boxes)}")
        objects = [PromptObject(id=i + 1, prompts=[FramePrompt(frame=0, box=PromptBox(*(int(v) for v in box)))])
                   for i, box in enumerate(boxes)]
        return self._records(image, build_object_prompt(size=image.size, objects=objects))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam_client.py -v`
Expected: 8 passed

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

from scout.vision.overlay import FLAG_RGB, render_overlay


def test_overlay_tints_crop_green_weed_magenta_and_outlines_flags():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    crop = np.zeros((10, 10), bool)
    crop[0, 0] = True
    weed = np.zeros((10, 10), bool)
    weed[9, 9] = True
    out = render_overlay(image, crop, weed, flagged=[(2, 2, 9, 9)])
    r, g, b = out.getpixel((0, 0))
    assert g > r and g > b
    r, g, b = out.getpixel((9, 9))
    assert r > g and b > g
    assert out.getpixel((4, 4)) == FLAG_RGB
    assert out.getpixel((5, 5)) == (255, 255, 255)


def test_overlay_shades_band_under_plants_and_skips_empty_boxes():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    band = np.zeros((10, 10), bool)
    band[:, 0] = True
    weed = np.zeros((10, 10), bool)
    weed[0, 0] = True
    out = render_overlay(image, np.zeros((10, 10), bool), weed, band, flagged=[(0, 0, 0, 0)])
    assert out.getpixel((0, 5)) == (225, 225, 223)
    r, g, b = out.getpixel((0, 0))
    assert r > g and b > g
    assert out.getpixel((5, 5)) == (255, 255, 255)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.overlay'`

- [ ] **Step 3: Write the implementation**

`scout/vision/overlay.py`:
```python
"""Draw crop (green), weeds (magenta), the in-row band (gray), and boxes around plants that need a check.

Weeds are magenta, not red: red over green leaves turns olive for red-green color-blind viewers, the same as the crop.
"""
import numpy as np
from PIL import Image, ImageDraw

from scout.models import Box

CROP_RGB = (99, 153, 34)
WEED_RGB = (230, 0, 170)
BAND_RGB = (136, 135, 128)
FLAG_RGB = (255, 214, 0)


def render_overlay(image: Image.Image, crop: np.ndarray, weed: np.ndarray,
                   band: np.ndarray | None = None, flagged: list[Box] = ()) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, rgb, alpha in ((band, BAND_RGB, 0.25), (crop, CROP_RGB, 0.55), (weed, WEED_RGB, 0.65)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    width = max(3, round(max(result.size) / 200))  # 5 px on a 1024 px photo: still visible when shown phone-width
    for left, top, right, bottom in flagged:
        if right > left and bottom > top:  # an empty mask's bbox is (0, 0, 0, 0)
            draw.rectangle((left, top, right - 1, bottom - 1), outline=FLAG_RGB, width=width)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_overlay.py -v`
Expected: 2 passed

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
            mismatched = sum(r.resized for r in records)
            confs = [round(r.confidence, 2) for r in records if r.confidence is not None]
            print(f"{stem} | {prompt!r}: {len(records)} found, api {fetched - started:.1f}s, "
                  f"decode {decoded - fetched:.1f}s, conf {confs[:8]}, mask/box size mismatches {mismatched}")
            render_overlay(image.work, unions[prompt], unions[prompt] & False).save(OUT / f"{stem}__{slug(prompt)}.png")
        crop = unions[args.crop_prompt]
        candidates = mk.pieces(unions["plant"], 0.0005 * image.size[0] * image.size[1], exclude=crop)
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

Open the `*__candidates.png` files. **Go** if, on at least 4 of the 6 photos, the magenta pieces mostly sit on weeds rather than on crop, and most visible weeds (grasses included) carry some magenta. **No-go:** try `--extra "weed" "seedling" "green plant"` on the failures.

If the "plant" concept is still poor, adopt the fallback (spec §8.2.3). The agent marks points on weeds in the full photo and calls `segment_at`. The tools in Task 16 already support this; strengthen `prompts.py` step 3 wording in Task 17. (Outcome: no-go. The live tests chose boxes over points; Tasks 15–17 follow the boxes redesign note.)

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

import pytest

from scout.agent.llm import Llm, LlmError, user_message


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


@pytest.mark.parametrize("raw", ["{not json", "[]", '"x"', "null", "5"])
def test_non_object_arguments_are_flagged(raw):
    create = lambda **k: _response([NS(type="function_call", call_id="c1", name="zoom", arguments=raw)])
    assert Llm(create, "m").turn("x").calls[0].arguments == {"__invalid__": raw}


def test_unfinished_response_raises():
    create = lambda **k: NS(id="r", output=[], output_text="", status="incomplete",
                            incomplete_details=NS(reason="max_output_tokens"), error=None)
    with pytest.raises(LlmError, match="incomplete"):
        Llm(create, "m").turn("x")


def test_real_sdk_response_is_parsed():
    from openai._models import construct_type
    from openai.types.responses import Response
    response = construct_type(type_=Response, value={
        "id": "resp_1", "object": "response", "created_at": 0, "model": "m", "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []},
                   {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "zoom",
                    "arguments": '{"instance_id": "s0-i1"}', "status": "completed"},
                   {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}]}]})
    turn = Llm(lambda **k: response, "m").turn("x")
    assert turn.response_id == "resp_1" and turn.text == "done"
    assert [(c.call_id, c.name, c.arguments) for c in turn.calls] == [("c1", "zoom", {"instance_id": "s0-i1"})]
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


class LlmError(RuntimeError):
    """Muse Spark answered, but the response did not complete (failed, incomplete, or cancelled)."""


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
        import httpx
        from openai import OpenAI

        # max_retries=0 leaves with_retry as the only retry layer; connect=5 fails fast on an unreachable host.
        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                        timeout=httpx.Timeout(180.0, connect=5.0), max_retries=0)
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
        status = getattr(response, "status", None)
        if status in ("failed", "incomplete", "cancelled"):
            detail = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
            raise LlmError(f"Muse Spark response {response.id} was {status}: {detail}")
        calls = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):  # tools are called with **arguments, so only an object is usable
                arguments = {"__invalid__": item.arguments}
            calls.append(ToolCall(item.call_id, item.name, arguments))
        return LlmTurn(response.id, calls, getattr(response, "output_text", "") or "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 11 passed

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
"""Day-1 test: Muse Spark tool calling, an image in a follow-up user message, structured output, and pointing.

Usage: python scripts/spike_agent.py field-images/IMG_4215.JPG [photos to test pointing on ...]
Pointing writes spike_out/<photo>__pointing_<pixel|norm1000>.jpg: SAM masks (magenta) from Muse Spark's points (yellow).
"""
import json
import sys
import time
from pathlib import Path

from PIL import ImageDraw

from scout.agent.llm import Llm, user_message
from scout.app.ingest import load_image, to_data_url
from scout.config import get_settings
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.sam_client import SamClient

TOOL = {
    "type": "function",
    "name": "report_rows",
    "description": "Report how many crop rows are visible.",
    "parameters": {"type": "object", "properties": {"count": {"type": "integer"}, "notes": {"type": "string"}},
                   "required": ["count"], "additionalProperties": False},
}
SCHEMA = {
    "type": "json_schema", "name": "plants", "strict": True,
    "schema": {"type": "object", "properties": {"crop": {"type": "string"}, "weeds_visible": {"type": "boolean"}},
               "required": ["crop", "weeds_visible"], "additionalProperties": False},
}
POINTS = {
    "type": "json_schema", "name": "weed_points", "strict": True,
    "schema": {"type": "object", "properties": {"weeds": {"type": "array", "items": {
        "type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "what": {"type": "string"}},
        "required": ["x", "y", "what"], "additionalProperties": False}}},
        "required": ["weeds"], "additionalProperties": False},
}


def pointing(llm: Llm, sam: SamClient, photo: str) -> None:
    """Ask for one point per weed in two coordinate conventions, segment each point with SAM, and save overlays."""
    image = load_image(0, Path(photo).read_bytes())
    width, height = image.size
    asks = {
        "pixel": (f"This field photo is {width}x{height} pixels. Give one point (x, y in pixels from the top-left "
                  "corner) on each weed plant you can see, not on the crop. Put each point on the plant's leaves.",
                  1.0, 1.0),
        "norm1000": ("Give one point on each weed plant you can see, not on the crop, as x and y from 0 to 1000 "
                     "(0,0 is the top-left corner, 1000,1000 the bottom-right). Put each point on the plant's leaves.",
                     width / 1000, height / 1000),
    }
    for name, (prompt, sx, sy) in asks.items():
        started = time.time()
        weeds = json.loads(llm.turn([user_message(prompt, [image.work_data_url])], text_format=POINTS).text)["weeds"]
        points = [(min(width - 1, max(0, round(w["x"] * sx))), min(height - 1, max(0, round(w["y"] * sy))))
                  for w in weeds]
        masks = [r.full_mask(image.size) for x, y in points for r in sam.segment_at(image, points=[(x, y, True)])]
        weed = mk.union(masks, image.size)
        out = render_overlay(image.work, weed & False, weed)
        draw = ImageDraw.Draw(out)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 214, 0), width=3)
        Path("spike_out").mkdir(exist_ok=True)
        path = f"spike_out/{Path(photo).stem}__pointing_{name}.jpg"
        out.save(path, quality=85)
        print(f"4-{name}) {time.time() - started:.1f}s {len(points)} points {points[:12]} "
              f"what={[w['what'] for w in weeds][:6]} -> {path}")


def main(photo: str, pointing_photos: list[str]) -> None:
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
    sam = SamClient.from_settings(get_settings())
    for other in pointing_photos:
        pointing(llm, sam, other)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
```

- [ ] **Step 2: Run it**

Run (from the repo root, with the repo on the import path):
`PYTHONPATH=. python scripts/spike_agent.py field-images/IMG_4215.JPG field-images/IMG_4219.JPG field-images/IMG_4215.JPG field-images/IMG_3926.JPG`
Expected:
- Line 1 shows a `report_rows` call with an integer count. The tool schema leaves `notes` optional, so a 400 here means Meta applies strict mode to tools by default. In that case add `"strict": False` to `_fn` in Task 16.
- Line 2 describes the close-up.
- Line 3 prints a dict with `crop` and `weeds_visible`.
- `4-pixel` and `4-norm1000` lines for each pointing photo. Open the `__pointing_*.jpg` files. Each yellow circle should sit on a weed and have a magenta SAM mask on that weed. Pick the convention whose points land on weeds, and record per photo how many points hit a weed, hit crop, or hit soil. Task 17's prompt uses the winning convention.

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
import numpy as np

from scout.vision.rows import band_mask, canopy_band, fit_rows


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


def _strip(x0, x1, shape=(100, 100)):
    mask = np.zeros(shape, bool)
    mask[:, x0:x1] = True
    return mask


def test_canopy_band_is_canopy_plus_margin():
    band = canopy_band(_strip(10, 30) | _strip(60, 80))
    assert band[50, 10] and band[50, 8] and band[50, 32]
    assert not band[50, 35] and not band[50, 45]


def test_canopy_band_fills_weed_holes_not_shut_in_furrows():
    crop = _strip(10, 40)
    crop[50:55, 20:25] = False
    shut = _strip(60, 70) | _strip(90, 100)
    shut[0:5, 60:100] = shut[95:100, 60:100] = True
    band = canopy_band(crop | shut)
    assert band[52, 22] and not band[50, 80]


def test_canopy_band_none_without_crop_or_between_row_ground():
    assert canopy_band(np.zeros((50, 50), bool)) is None
    assert canopy_band(np.ones((50, 50), bool)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rows.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.vision.rows'`

- [ ] **Step 3: Write the implementation**

`scout/vision/rows.py`:
```python
"""In-row zone: lines fitted through separate crop plants (RANSAC), or the row canopies SAM returns."""
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


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


def canopy_band(crop: np.ndarray, margin: int = 3, max_cover: float = 0.9) -> np.ndarray | None:
    """In-row zone when crop masks are whole row canopies (a SAM prompt on a crop row returns the row).

    The canopy plus weed-sized holes in it, grown by `margin` px so slightly disagreeing mask edges still meet.
    Bigger enclosed gaps (a furrow shut in where two rows touch) stay between-row. None when no between-row
    ground is left to compare against.
    """
    if not crop.any():
        return None
    holes, count = ndimage.label(ndimage.binary_fill_holes(crop) & ~crop)
    half_width = 2.0 * float(ndimage.distance_transform_edt(crop)[crop].mean())  # exact for a straight strip
    small = np.bincount(holes.ravel(), minlength=count + 1) <= half_width ** 2
    small[0] = False
    zone = ndimage.binary_dilation(crop | small[holes], structure=_EIGHT, iterations=margin)
    return zone if zone.mean() <= max_cover else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rows.py -v`
Expected: 6 passed

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
from scout.vision.measure import compute_metrics, footprint_m2, species_key

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


def _boxed(iid, role, species=None, source="prompt", prompt="box", height=None):
    return Instance(id=iid, source=source, prompt=prompt, box=(0, 0, 1, 1), area_px=0, role=role,
                    species_id=species, height_in=height)


def test_footprint_is_the_same_portrait_or_landscape():
    assert footprint_m2((4032, 3024), "top_down") == pytest.approx(2.04, abs=0.01)
    assert footprint_m2((3024, 4032), "top_down") == pytest.approx(footprint_m2((4032, 3024), "top_down"))
    assert footprint_m2((0, 0), "top_down") is None


def test_boxed_weed_inside_row_canopy_wins_over_crop():
    row, weed = _mask((40, 0, 60, 100)), _mask((45, 40, 55, 50))
    m = compute_metrics([_boxed("c", "crop"), _boxed("w", "weed", "waterhemp", height=4.0)], {"c": row, "w": weed},
                        SIZE, "row", band=_mask((35, 0, 65, 100)))
    assert (m.weed_cover_pct, m.crop_cover_pct, m.weed_count) == (1.0, 19.0, 1)
    assert m.in_row_weed_share == 1.0 and m.max_height_in == 4.0


def test_weed_over_crop_edge_counts_once_as_weed():
    m = compute_metrics([_boxed("c", "crop"), _boxed("w", "weed", "waterhemp")],
                        {"c": _mask((0, 0, 10, 10)), "w": _mask((5, 0, 15, 10))}, SIZE, "row")
    assert (m.weed_cover_pct, m.crop_cover_pct) == (1.0, 0.5)


def test_overlapping_weeds_count_each_pixel_once():
    m = compute_metrics([_boxed("a", "weed", "waterhemp"), _boxed("b", "weed", "giant_ragweed")],
                        {"a": _mask((0, 0, 10, 10)), "b": _mask((5, 0, 15, 10))}, SIZE, "row")
    assert m.weed_cover_pct == 1.5 and m.species_cover_pct == {"waterhemp": 1.0, "giant_ragweed": 0.5}
    dup = compute_metrics([_boxed("a", "weed", "waterhemp"), _boxed("b", "weed", "waterhemp")],
                          {"a": _mask((0, 0, 10, 10)), "b": _mask((2, 2, 12, 12))}, SIZE, "row")
    assert dup.weed_count == 1  # a second box mostly on the same plant


def test_sky_rules():
    weed = {"w": _mask((0, 60, 10, 70))}
    half_sky = _mask((0, 0, 100, 50))
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "top_down", sky=half_sky).weed_cover_pct == 1.0
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "row", sky=half_sky).weed_cover_pct == 2.0
    all_sky = np.ones((100, 100), bool)
    assert compute_metrics([_boxed("w", "weed")], weed, SIZE, "wide", sky=all_sky).weed_cover_pct == 1.0


def test_roles_missing_masks_and_density_semantics():
    instances = [_boxed("u", "unknown"), _boxed("i", "ignore"), _boxed("gone", "weed")]
    masks = {"u": _mask((0, 0, 10, 10)), "i": _mask((20, 0, 30, 10))}
    m = compute_metrics(instances, masks, SIZE, "top_down")
    assert m.weed_count == 0 and m.density_per_m2 == 0.0 and m.species_density == {}  # a top-down stop with no weeds
    assert m.in_row_weed_share is None
    assert compute_metrics(instances, masks, SIZE, "close_up").species_density is None


def test_approximate_only_for_big_pieces_outside_close_ups():
    small = compute_metrics([_boxed("w", "weed")], {"w": _mask((0, 0, 10, 10))}, SIZE, "row")
    assert small.count_approximate is False
    big = {"w": _mask((0, 0, 30, 30))}
    assert compute_metrics([_boxed("w", "weed")], big, SIZE, "row").count_approximate is True
    assert compute_metrics([_boxed("w", "weed")], big, SIZE, "close_up").count_approximate is False


def test_grass_words_in_labels_mean_unidentified_grass():
    assert species_key(_boxed("a", "weed", prompt="giant foxtail")) == "unidentified_grass"
    assert species_key(_boxed("b", "weed", prompt="pigweed seedling")) == "unidentified_broadleaf"
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
GRASS_WORDS = ("grass", "foxtail", "sedge", "barnyard", "panicum", "shattercane")
PHONE_HFOV_DEG = 69.0  # spans the sensor's long side, whichever way the phone was held
MIN_GROUND_FRAC = 0.2  # a sky mask leaving less ground than this is a bad SAM answer; ignore it
MIN_OWN_FRAC = 0.5  # a weed must own at least half of its on-ground pixels to count as a plant


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
    """Ground area of a straight-down photo; None for any other view. The FOV spans the photo's long side."""
    long_px, short_px = max(size), min(size)
    if view_type != "top_down" or short_px <= 0:
        return None
    ground_long = 2 * capture_height_m * math.tan(math.radians(hfov_deg / 2))
    return ground_long * ground_long * short_px / long_px


def species_key(instance: Instance) -> str:
    if instance.species_id:
        return instance.species_id
    label = instance.prompt.lower()
    return "unidentified_grass" if any(word in label for word in GRASS_WORDS) else "unidentified_broadleaf"


def compute_metrics(instances: list[Instance], masks: dict[str, np.ndarray], size: tuple[int, int],
                    view_type: str, sky: np.ndarray | None = None, band: np.ndarray | None = None,
                    capture_height_m: float = CAPTURE_HEIGHT_M) -> Metrics:
    """Each pixel counts once. Weeds win over crop (the tools already dropped weed masks that were mostly crop),
    and where weeds overlap, the one registered first owns the pixels."""
    width, height = size
    ground = np.ones((height, width), dtype=bool)
    if sky is not None and view_type in ("row", "wide") and (~sky).mean() >= MIN_GROUND_FRAC:
        ground = ~np.asarray(sky, dtype=bool)
    ground_px = max(int(ground.sum()), 1)
    crop = np.zeros((height, width), dtype=bool)
    for inst in instances:
        if inst.role == "crop" and inst.id in masks:
            crop |= np.asarray(masks[inst.id], dtype=bool)
    crop &= ground
    weed_mask = np.zeros((height, width), dtype=bool)
    species_masks: dict[str, np.ndarray] = {}
    species_counts: dict[str, int] = {}
    species_heights: dict[str, float] = {}
    heights: list[float] = []
    approximate = False
    for inst in instances:
        if inst.role != "weed" or inst.id not in masks:
            continue
        on_ground = np.asarray(masks[inst.id], dtype=bool) & ground
        own = on_ground & ~weed_mask
        own_px = int(own.sum())
        if own_px == 0 or own_px < MIN_OWN_FRAC * int(on_ground.sum()):
            continue  # a second box on a plant already counted
        key = species_key(inst)
        weed_mask |= own
        if key in species_masks:
            species_masks[key] |= own
        else:
            species_masks[key] = own
        species_counts[key] = species_counts.get(key, 0) + 1
        if inst.height_in is not None:
            heights.append(inst.height_in)
            species_heights[key] = max(species_heights.get(key, 0.0), inst.height_in)
        if view_type != "close_up" and own_px > APPROX_PIECE_FRAC * ground_px:
            approximate = True
    crop &= ~weed_mask
    weed_px = int(weed_mask.sum())
    count = sum(species_counts.values())
    area = footprint_m2(size, view_type, capture_height_m)
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
Expected: 12 passed

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
import pytest

from scout.agronomy.weeds import WeedBook, WeedInfo


def test_default_book_loads_all_species():
    book = WeedBook.load()
    assert len(book.ids()) == 36
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


def _info(sid, **kw):
    return WeedInfo(id=sid, common_name=kw.pop("common_name", sid), scientific_name="", type="broadleaf", **kw)


@pytest.mark.parametrize("phrase, expected", [
    ("waterhemp pigweed", "waterhemp"),
    ("Waterhemp (Amaranthus tuberculatus)", "waterhemp"),
    ("dandelion flower", "dandelion"),
    ("Dandelions", "dandelion"),
    ("green foxtail seedlings", "green_foxtail"),
    ("giant_foxtail", "giant_foxtail"),
    ("Lamb's-quarters", "common_lambsquarters"),
    ("lambs quarters", "common_lambsquarters"),
    ("velvet leaf", "velvetleaf"),
    ("morning glories", "morningglory"),
    ("Palmer's amaranth", "palmer_amaranth"),
    ("horsetail weed", "field_horsetail"),
    ("Marestail", "horseweed"),
])
def test_resolve_tolerates_model_phrasing(phrase, expected):
    assert WeedBook.load().resolve(phrase) == expected


@pytest.mark.parametrize("phrase", ["pigweed", "ragweed", "giant ragweed or common ragweed", "not a weed", "", None])
def test_resolve_refuses_unknown_and_ambiguous(phrase):
    assert WeedBook.load().resolve(phrase) is None


def test_every_name_belongs_to_one_species():
    assert [p for p in WeedBook.load().validate() if "several species" in p] == []


def test_validate_flags_bad_entries():
    book = WeedBook({
        "a": _info("a", zero_tolerance=True),
        "b": WeedInfo(id="b", common_name="b", scientific_name="", type="weedy", action_density=1.5, sources=["s"]),
        "c": _info("c", action_cover_pct="1-2", sources=["s"]),
        "d": _info("d", synonyms=["shared"]), "e": _info("e", synonyms=["shared"]),
    })
    problems = "\n".join(book.validate())
    for fragment in ("a: threshold without a source", "b: type 'weedy'", "c: action_cover_pct", "'shared'"):
        assert fragment in problems


def test_load_errors_name_the_species(tmp_path):
    path = tmp_path / "weeds.yaml"
    path.write_text("waterhemp: {common_name: x, scientific_name: y, type: broadleaf, zero_tolerence: true}\n")
    with pytest.raises(ValueError, match="waterhemp"):
        WeedBook.load(path)
    path.write_text("waterhemp: {common_name: x, scientific_name: y, type: broadleaf, synonyms: common waterhemp}\n")
    with pytest.raises(ValueError, match="waterhemp"):
        WeedBook.load(path)
    path.write_text("")
    with pytest.raises(ValueError):
        WeedBook.load(path)


def test_contains_ignores_non_strings():
    book = WeedBook.load()
    assert "waterhemp" in book and ["waterhemp"] not in book and None not in book
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_weeds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agronomy.weeds'`

- [ ] **Step 3: Write the data file**

`scout/agronomy/weeds.yaml`. Thresholds start empty; Task 19 fills them with sources.
```yaml
waterhemp: {common_name: waterhemp, scientific_name: Amaranthus tuberculatus, type: broadleaf, synonyms: [common waterhemp, tall waterhemp, waterhemp pigweed, Amaranthus rudis]}
palmer_amaranth: {common_name: Palmer amaranth, scientific_name: Amaranthus palmeri, type: broadleaf, synonyms: [palmer pigweed, careless weed, palmers amaranth]}
redroot_pigweed: {common_name: redroot pigweed, scientific_name: Amaranthus retroflexus, type: broadleaf, synonyms: [redroot, rough pigweed]}
common_lambsquarters: {common_name: common lambsquarters, scientific_name: Chenopodium album, type: broadleaf, synonyms: [lambsquarters]}
giant_ragweed: {common_name: giant ragweed, scientific_name: Ambrosia trifida, type: broadleaf, synonyms: [buffaloweed, great ragweed]}
common_ragweed: {common_name: common ragweed, scientific_name: Ambrosia artemisiifolia, type: broadleaf, synonyms: [annual ragweed, short ragweed]}
velvetleaf: {common_name: velvetleaf, scientific_name: Abutilon theophrasti, type: broadleaf, synonyms: [buttonweed, butterprint]}
horseweed: {common_name: horseweed, scientific_name: Conyza canadensis, type: broadleaf, synonyms: [marestail, Erigeron canadensis, Canada fleabane, Canadian horseweed]}
kochia: {common_name: kochia, scientific_name: Bassia scoparia, type: broadleaf, synonyms: [burningbush, summer cypress, Mexican fireweed, Kochia scoparia]}
common_cocklebur: {common_name: common cocklebur, scientific_name: Xanthium strumarium, type: broadleaf, synonyms: [cocklebur]}
pennsylvania_smartweed: {common_name: Pennsylvania smartweed, scientific_name: Persicaria pensylvanica, type: broadleaf, synonyms: [smartweed, Polygonum pensylvanicum]}
morningglory: {common_name: morningglory, scientific_name: Ipomoea spp., type: broadleaf, synonyms: [ivyleaf morningglory, tall morningglory, entireleaf morningglory, pitted morningglory, Ipomoea hederacea, Ipomoea purpurea, Ipomoea lacunosa]}
jimsonweed: {common_name: jimsonweed, scientific_name: Datura stramonium, type: broadleaf, synonyms: [thornapple]}
eastern_black_nightshade: {common_name: eastern black nightshade, scientific_name: Solanum ptycanthum, type: broadleaf, synonyms: [black nightshade, nightshade, Solanum ptychanthum, Solanum nigrum]}
giant_foxtail: {common_name: giant foxtail, scientific_name: Setaria faberi, type: grass, synonyms: [foxtail]}
green_foxtail: {common_name: green foxtail, scientific_name: Setaria viridis, type: grass, synonyms: []}
yellow_foxtail: {common_name: yellow foxtail, scientific_name: Setaria pumila, type: grass, synonyms: [Setaria glauca]}
barnyardgrass: {common_name: barnyardgrass, scientific_name: Echinochloa crus-galli, type: grass, synonyms: [cockspur grass]}
large_crabgrass: {common_name: large crabgrass, scientific_name: Digitaria sanguinalis, type: grass, synonyms: [crabgrass, hairy crabgrass]}
fall_panicum: {common_name: fall panicum, scientific_name: Panicum dichotomiflorum, type: grass, synonyms: []}
shattercane: {common_name: shattercane, scientific_name: Sorghum bicolor, type: grass, synonyms: [wild cane]}
woolly_cupgrass: {common_name: woolly cupgrass, scientific_name: Eriochloa villosa, type: grass, synonyms: []}
volunteer_corn: {common_name: volunteer corn, scientific_name: Zea mays, type: grass, synonyms: []}
canada_thistle: {common_name: Canada thistle, scientific_name: Cirsium arvense, type: broadleaf, synonyms: [creeping thistle]}
dandelion: {common_name: dandelion, scientific_name: Taraxacum officinale, type: broadleaf, synonyms: [common dandelion]}
common_milkweed: {common_name: common milkweed, scientific_name: Asclepias syriaca, type: broadleaf, synonyms: [milkweed]}
wild_mustard: {common_name: wild mustard, scientific_name: Sinapis arvensis, type: broadleaf, synonyms: [charlock, mustard, Brassica kaber]}
field_pennycress: {common_name: field pennycress, scientific_name: Thlaspi arvense, type: broadleaf, synonyms: [pennycress, fanweed, stinkweed]}
henbit: {common_name: henbit, scientific_name: Lamium amplexicaule, type: broadleaf, synonyms: [henbit deadnettle]}
purple_deadnettle: {common_name: purple deadnettle, scientific_name: Lamium purpureum, type: broadleaf, synonyms: [red deadnettle]}
common_chickweed: {common_name: common chickweed, scientific_name: Stellaria media, type: broadleaf, synonyms: [chickweed]}
yellow_nutsedge: {common_name: yellow nutsedge, scientific_name: Cyperus esculentus, type: sedge, synonyms: [nutsedge, nutgrass]}
quackgrass: {common_name: quackgrass, scientific_name: Elymus repens, type: grass, synonyms: [couchgrass, Elytrigia repens, Agropyron repens]}
field_horsetail: {common_name: field horsetail, scientific_name: Equisetum arvense, type: other, synonyms: [horsetail]}
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_weeds.py -v`
Expected: 26 passed

- [ ] **Step 6: Commit**

```bash
git add scout/agronomy/weeds.yaml scout/agronomy/weeds.py tests/test_weeds.py
git commit -m "feat: add weed book with names and synonyms"
```

### Task 15: Stop state

Tasks 15–17 follow the boxes redesign (`docs/superpowers/specs/2026-10-30-boxes-redesign.md`). SAM text prompts don't find crop plants or weeds inside a crop canopy, so there are no "plant minus crop" candidates. Instead, Muse Spark draws a box around each crop row or plant and each weed, in 0–1000 coordinates, and SAM segments all of a call's boxes in one request (`SamClient.segment_boxes`, Task 7). This task holds the state for one stop:
- Box conversion between the agent's 0–1000 coordinates and work-image pixels.
- The weed check:
  - A weed mask that lies mostly (≥ 60%) on separate crop plants is dropped.
  - Row canopies never veto a weed, since a weed inside the row sits wholly in the canopy mask.
  - Weeds that overlap any crop by more than 30%, and patches over 12% of the frame, are kept with `check_this`.
- A patch makes the count approximate, except in a close-up, where a big weed is just a big plant.
- Sky for row and wide views (`ensure_sky`): asked once and best effort. Without it, cover is measured over the whole photo.
- The in-row zone, and the overlay, where weeds win over crop.

**Files:**
- Create: `scout/agent/state.py`
- Modify: `tests/helpers.py` (append)
- Test: `tests/test_state.py`

- [ ] **Step 1: Append helpers and write the failing test**

Append to `tests/helpers.py`:
```python
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.sam_parse import SamRecord


def rect(left, top, right, bottom, size=(100, 100)):
    mask = np.zeros((size[1], size[0]), bool)
    mask[top:bottom, left:right] = True
    return mask


def mask_record(object_id, mask, confidence=0.9):
    """A SAM record for a full-frame mask: its bounding box plus the box-local raster."""
    left, top, right, bottom = mk.bbox(mask)
    return SamRecord(object_id, (left, top, right, bottom), confidence, mask[top:bottom, left:right].copy())


def field_scene():
    """What a 100x100 soybean photo really holds. Test boxes are 0-1000, so 10x these pixel numbers."""
    return {
        "row": rect(10, 0, 30, 100),  # a crop row seen as one canopy, full height
        "plant": rect(40, 30, 52, 42),  # one separate crop plant
        "weed_a": rect(60, 60, 70, 70),
        "weed_b": rect(80, 10, 90, 20),
        "edge_weed": rect(26, 40, 36, 50),  # 40% of it inside the row canopy
        "patch": rect(40, 75, 100, 100),  # 15% of the frame: a weed patch
    }


class FakeSam:
    """Concept prompts answer from a dict. A box prompt returns the scene object that fills most of the box, cut to
    the box as SAM does; a box over bare ground returns nothing."""

    def __init__(self, scene=None, concepts=None, fail=None, confidence=0.9):
        self.scene = field_scene() if scene is None else scene
        self.concepts = concepts or {}
        self.fail = fail or set()
        self.confidence = confidence
        self.calls = []

    def segment_concept(self, image, phrase):
        self.calls.append(("concept", phrase))
        if phrase in self.fail:
            raise BadRequest(f"rejected {phrase}")
        return list(self.concepts.get(phrase, []))

    def segment_boxes(self, image, boxes):
        self.calls.append(("boxes", tuple(boxes)))
        if "boxes" in self.fail:
            raise BadRequest("rejected boxes")
        records = []
        for n, (left, top, right, bottom) in enumerate(boxes, start=1):
            inside = {name: int(mask[top:bottom, left:right].sum()) for name, mask in self.scene.items()}
            best = max(inside, key=inside.get, default=None)
            if best is not None and inside[best]:
                local = self.scene[best][top:bottom, left:right].copy()
                records.append(SamRecord(str(n), (left, top, right, bottom), self.confidence, local))
        return records


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
import numpy as np
import pytest

from scout.agent.state import StopState, to_pixels, to_thousandths
from tests.helpers import field_scene, make_image, mask_record, rect


def _state(crop="soybean"):
    return StopState(make_image(100, 100), crop)


def test_boxes_convert_between_thousandths_and_pixels():
    assert to_pixels([100, 200, 300, 400], (100, 50)) == (10, 10, 30, 20)
    assert to_thousandths((10, 10, 30, 20), (100, 50)) == [100, 200, 300, 400]
    assert to_pixels([300, 400, 100, 200], (100, 50)) == (10, 10, 30, 20)  # corners in either order
    assert to_pixels([-50, 990, 1200, 1000], (100, 100)) == (0, 98, 100, 100)  # clamped, still 2 px tall
    assert to_pixels([500.4, 500, 500, 500], (100, 100)) == (50, 50, 52, 52)  # grown to 2 px a side
    with pytest.raises(ValueError):
        to_pixels([1, 2, 3], (100, 100))


def test_weed_on_a_crop_plant_is_dropped_but_weeds_in_a_row_canopy_are_flagged():
    state, scene = _state(), field_scene()
    state.add_mask(scene["row"], "prompt", "soybean row", role="crop")
    state.add_mask(scene["plant"], "prompt", "soybean plant", role="crop")
    assert state.add_weed(scene["plant"] & rect(42, 32, 50, 40), "prompt", "pigweed") is None
    in_row = state.add_weed(scene["row"] & rect(12, 50, 22, 60), "prompt", "leaf in the row")
    edge = state.add_weed(scene["edge_weed"], "prompt", "pigweed")
    clear = state.add_weed(scene["weed_a"], "prompt", "waterhemp")
    assert in_row.role == edge.role == clear.role == "weed" and edge.id_confidence is None
    assert in_row.check_this and edge.check_this and not clear.check_this
    assert edge.area_px == 100  # the whole mask is kept, including the part inside the row
    assert state.unclassified() == [in_row.id, edge.id, clear.id] == ["s0-i3", "s0-i4", "s0-i5"]


def test_large_weed_mask_is_kept_as_an_approximate_patch():
    state = _state()
    assert not state.count_approximate()
    patch = state.add_weed(field_scene()["patch"], "prompt", "foxtail patch")
    assert patch.role == "weed" and patch.check_this and state.is_patch(patch)
    assert state.has_patch() and state.count_approximate()


def test_row_band_from_a_row_canopy_leaves_out_sky():
    state = _state()
    assert state.row_band() is None  # no crop yet
    state.add_mask(field_scene()["row"], "prompt", "soybean row", role="crop")
    band = state.row_band()
    assert band[50, 8] and band[50, 31] and not band[50, 40]
    state.sky = rect(0, 0, 100, 20)
    band = state.row_band()
    assert not band[10, 20] and band[50, 20]


def _plants_in_two_rows(state):
    for x in (17, 67):
        for y in (10, 40, 70):
            state.add_mask(rect(x, y, x + 6, y + 6), "prompt", "soybean plant", role="crop")


def test_row_band_fits_lines_through_separate_plants():
    state = _state()
    _plants_in_two_rows(state)
    band = state.row_band()
    assert band[50, 20] and band[50, 70] and not band[50, 45]


def test_row_band_is_none_for_alfalfa():
    state = _state("alfalfa")
    _plants_in_two_rows(state)
    state.add_mask(field_scene()["row"], "prompt", "alfalfa", role="crop")
    assert state.row_band() is None


def test_overlay_paints_weeds_over_crop():
    state, scene = _state(), field_scene()
    state.add_mask(scene["row"], "prompt", "soybean row", role="crop")
    state.add_mask(scene["edge_weed"], "prompt", "pigweed", role="weed")
    pixels = np.asarray(state.overlay())
    # (45, 28) is in the row and the weed, (45, 31) only in the weed, (45, 20) only in the row; all are in the band
    assert tuple(pixels[45, 28]) == tuple(pixels[45, 31]) != tuple(pixels[45, 20])


def test_log_reports_steps():
    seen = []
    state = StopState(make_image(), "corn", on_step=seen.append, clock=lambda: 0.0)
    state.log("mark", "Mark crop: 2 of 2 boxes kept", ["s0-i1"])
    assert seen[0].summary.startswith("Mark") and seen[0].t_ms == 0


def test_overlay_renders_without_instances():
    assert StopState(make_image(40, 30), "corn").overlay().size == (40, 30)


def test_patch_is_approximate_except_in_a_close_up():
    state = _state()
    state.add_weed(field_scene()["patch"], "prompt", "foxtail patch")
    assert state.count_approximate() and state.count_approximate("row") and state.count_approximate("top_down")
    assert not state.count_approximate("close_up")  # a few big plants fill a close-up
    state.box_cap_hit = True
    assert state.count_approximate("close_up")  # a floor, whatever the view


class _SkySam:
    def __init__(self, records=None, error=None):
        self.records, self.error, self.calls = records or [], error, 0

    def segment_concept(self, image, phrase):
        self.calls += 1
        if self.error:
            raise self.error
        return self.records


def test_ensure_sky_asks_once():
    state, sam = _state(), _SkySam([mask_record("0", rect(0, 0, 100, 20))])
    state.ensure_sky(sam)
    state.ensure_sky(sam)
    assert sam.calls == 1 and state.sky[10, 50] and not state.sky[50, 50] and state.steps == []


def test_ensure_sky_is_best_effort_and_not_retried():
    state, sam = _state(), _SkySam(error=TimeoutError("timed out"))
    state.ensure_sky(sam)
    state.ensure_sky(sam)
    assert sam.calls == 1 and state.sky is None
    assert [(s.tool, s.ok) for s in state.steps] == [("sky", False)]
    assert "TimeoutError" in state.steps[0].summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.state'`

- [ ] **Step 3: Write the implementation**

`scout/agent/state.py`:
```python
"""Everything known about one stop while the agent works on it.

The agent sees and gives boxes as [x1, y1, x2, y2] from 0 to 1000 on both axes (x from the left edge, y from the
top). Code works in work-image pixels; to_pixels and to_thousandths convert between the two.
"""
import time
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from scout.app.ingest import StopImage
from scout.models import AgentStep, Box, Instance
from scout.vision import masks as mk
from scout.vision.overlay import render_overlay
from scout.vision.rows import band_mask, canopy_band, fit_rows

SCALE = 1000
MIN_BOX_PX = 2
LOW_SAM_CONFIDENCE = 0.5
CANOPY_SPAN_FRAC = 1 / 3  # a crop mask this long (share of the image's long side) is a row canopy, not one plant
MOSTLY_CROP = 0.6  # a weed mask this much on crop plants is a crop hit and is dropped
CHECK_OVERLAP = 0.3  # a weed mask this much on any crop is kept but flagged for a closer look
PATCH_FRAC = 0.12  # a weed mask over 12% of the frame is a patch of plants: kept as one, flagged, count approximate
MAX_BOXES = 16  # SAM takes at most 16 objects per request


class StopFailed(Exception):
    """The stop cannot continue (e.g. SAM rejected the same tool twice)."""


def _span(a: float, b: float, extent: int) -> tuple[int, int]:
    lo, hi = sorted((a * extent / SCALE, b * extent / SCALE))
    lo = round(min(max(lo, 0.0), extent - MIN_BOX_PX))
    hi = round(min(max(hi, lo + MIN_BOX_PX), extent))
    return lo, hi


def to_pixels(box: Sequence[float], size: tuple[int, int]) -> Box:
    """A 0-1000 box from the agent -> work-image pixels: rounded, clamped to the image, at least 2 px a side."""
    if len(box) != 4:
        raise ValueError(f"a box is [x1, y1, x2, y2], got {list(box)}")
    x1, y1, x2, y2 = (float(v) for v in box)
    (left, right), (top, bottom) = _span(x1, x2, size[0]), _span(y1, y2, size[1])
    return (left, top, right, bottom)


def to_thousandths(box: Box, size: tuple[int, int]) -> list[int]:
    """A work-image pixel box -> [x1, y1, x2, y2] from 0 to 1000, for the agent."""
    width, height = size
    left, top, right, bottom = box
    return [round(left * SCALE / width), round(top * SCALE / height),
            round(right * SCALE / width), round(bottom * SCALE / height)]


class StopState:
    def __init__(self, image: StopImage, crop: str, on_step: Callable[[AgentStep], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.image = image
        self.crop = crop
        self.instances: dict[str, Instance] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.sky: np.ndarray | None = None
        self._sky_tried = False
        self.view_type: str | None = None
        self.crop_stage: str | None = None
        self.notes = ""
        self.steps: list[AgentStep] = []
        self.calls = 0
        self.bad_requests: dict[str, int] = {}
        self.box_cap_hit = False  # a weed mark used all 16 boxes, so the weed count is a floor
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
                 confidence: float | None = None) -> Instance:
        self._next += 1
        inst = Instance(id=f"s{self.image.index}-i{self._next}", source=source, prompt=prompt, box=mk.bbox(mask),
                        sam_confidence=confidence, area_px=int(mask.sum()), role=role,
                        check_this=confidence is not None and confidence < LOW_SAM_CONFIDENCE)
        self.instances[inst.id] = inst
        self.masks[inst.id] = mask
        return inst

    def union_of(self, predicate: Callable[[Instance], bool]) -> np.ndarray:
        return mk.union([self.masks[i.id] for i in self.instances.values() if predicate(i)], self.size)

    def is_canopy(self, inst: Instance) -> bool:
        """A crop mask spanning a third of the image's long side is a row canopy, not one plant."""
        left, top, right, bottom = inst.box
        return max(right - left, bottom - top) >= CANOPY_SPAN_FRAC * max(self.size)

    def is_patch(self, inst: Instance) -> bool:
        width, height = self.size
        return inst.area_px > PATCH_FRAC * width * height

    def add_weed(self, mask: np.ndarray, source: str, prompt: str, confidence: float | None = None) -> Instance | None:
        """Register a weed mask, or return None when it mostly lies on crop plants (a box drawn on crop).

        Only separate crop plants can veto a weed: a weed growing inside a row sits almost wholly in the row-canopy
        mask, so canopy overlap only flags it. Weeds overlapping any crop, and large patches, get check_this.
        """
        area = max(int(mask.sum()), 1)
        plants = self.union_of(lambda i: i.role == "crop" and not self.is_canopy(i))
        if float((mask & plants).sum()) / area >= MOSTLY_CROP:
            return None
        crop = self.union_of(lambda i: i.role == "crop")
        inst = self.add_mask(mask, source, prompt, role="weed", confidence=confidence)
        on_crop = float((mask & crop).sum()) / area > CHECK_OVERLAP
        inst.check_this = bool(inst.check_this or on_crop or self.is_patch(inst))
        return inst

    def unclassified(self) -> list[str]:
        """Weeds that were marked but never classified."""
        return [i.id for i in self.instances.values() if i.role == "weed" and i.id_confidence is None]

    def has_patch(self) -> bool:
        return any(i.role == "weed" and self.is_patch(i) for i in self.instances.values())

    def count_approximate(self, view_type: str | None = None) -> bool:
        """A patch counts as one weed (but a big plant in a close-up is just a plant), and a capped mark may have
        left weeds unmarked."""
        return (self.has_patch() and view_type != "close_up") or self.box_cap_hit

    def ensure_sky(self, sam) -> None:
        """Sky for row and wide views, asked once. Best effort: without it, cover is measured over the whole photo."""
        if self.sky is not None or self._sky_tried:
            return
        self._sky_tried = True
        try:
            records = sam.segment_concept(self.image, "sky")
        except Exception as error:
            self.log("sky", f"Sky not found ({type(error).__name__}); cover uses the whole photo", ok=False)
            return
        self.sky = mk.union([r.full_mask(self.size) for r in records], self.size)

    def row_band(self) -> np.ndarray | None:
        """In-row zone: none for alfalfa (solid-seeded) or without crop; the row canopies when any crop mask is
        canopy-sized, else lines fitted through crop-plant centers. Sky is never in the band."""
        crops = [i for i in self.instances.values() if i.role == "crop"]
        if self.crop == "alfalfa" or not crops:
            return None
        if any(self.is_canopy(i) for i in crops):
            band = canopy_band(self.union_of(lambda i: i.role == "crop"))
        else:
            centers = [((i.box[0] + i.box[2]) / 2, (i.box[1] + i.box[3]) / 2) for i in crops]
            widths = [((i.box[2] - i.box[0]) + (i.box[3] - i.box[1])) / 2 for i in crops]
            fit = fit_rows(centers, widths)
            band = band_mask(fit, self.size) if fit else None
        if band is not None and self.sky is not None:
            band = band & ~self.sky
        return band

    def overlay(self) -> Image.Image:
        weed = self.union_of(lambda i: i.role == "weed")
        crop = self.union_of(lambda i: i.role == "crop") & ~weed  # weeds win: a weed in the row keeps its pixels
        flagged = [i.box for i in self.instances.values() if i.role == "weed" and i.check_this]
        return render_overlay(self.image.work, crop, weed, self.row_band(), flagged)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/state.py tests/helpers.py tests/test_state.py
git commit -m "feat: add stop state with 0-1000 boxes, weed check, and in-row zone"
```

### Task 16: Agent tools

The six tools:
- **`segment(phrase, kind)`:** one SAM concept prompt. It works for "crop row" (crop) and for distinct weeds such as "dandelion" or "yellow flower" (weed). A bare "flower" would match alfalfa blossoms.
- **`mark(kind, boxes)`:** 1–16 `{box, label}` items, sent as one SAM request. The label becomes the instance's `prompt`.
- **`zoom(instance_id | box)`**, **`classify(items)`**, **`measure(view_type)`** and **`finish(crop_stage, notes)`**.

Every box the agent gives or gets is 0–1000. Weed masks from `segment` and `mark` both go through `StopState.add_weed`. The agent hears back, per box:
- "box N: mostly crop" (dropped),
- "box N: large patch; counted as one, approximate" (kept), or
- "box N: nothing found".

A weed `mark` that uses all 16 boxes makes the count approximate, since it is then a floor. `finish` refuses while marked weeds are unclassified and calls remain.

`classify` checks every item before it changes any: numbers may arrive as `3` or `"3"`, and a species counts only for a weed (a crop item labelled "soybean" is fine). A confident verdict clears the canopy and low-SAM `check_this` flag; patches stay flagged.

**Files:**
- Create: `scout/agent/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tools.py`:
```python
import base64
import io
import json

import pytest
from PIL import Image

from scout.agent.state import StopFailed, StopState
from scout.agent.tools import TOOL_SCHEMAS, ToolRunner
from scout.vision.sam_parse import SamRecord
from tests.helpers import FakeSam, field_scene, make_image, mask_record, rect, small_weedbook

# Boxes as the agent gives them, 0-1000; the test photo is 100x100, so pixels are a tenth of these.
ROW, PLANT = [100, 0, 300, 1000], [400, 300, 520, 420]
WEED_A, WEED_B, EDGE = [600, 600, 700, 700], [800, 100, 900, 200], [260, 400, 360, 500]
ON_PLANT, IN_ROW = [420, 320, 500, 400], [120, 500, 220, 600]
PATCH, BARE = [400, 750, 1000, 1000], [400, 500, 500, 600]


def _runner(sam=None, max_calls=12):
    state = StopState(make_image(100, 100), "soybean")
    return ToolRunner(state, sam or FakeSam(), small_weedbook(), max_calls=max_calls), state


def _mark(runner, kind, *boxes):
    items = [{"box": box, "label": f"{kind} {n}"} for n, box in enumerate(boxes, start=1)]
    return json.loads(runner.run("mark", {"kind": kind, "boxes": items}).output)


def test_schemas_name_six_tools():
    assert [t["name"] for t in TOOL_SCHEMAS] == ["segment", "mark", "zoom", "classify", "measure", "finish"]


def test_mark_sends_pixels_in_one_request_and_reports_thousandths():
    sam = FakeSam()
    runner, state = _runner(sam)
    crop = _mark(runner, "crop", ROW, PLANT)
    assert [f["box"] for f in crop["found"]] == [ROW, PLANT] and crop["notes"] == []
    out = _mark(runner, "weed", WEED_A, WEED_B, EDGE, ON_PLANT, IN_ROW, PATCH, BARE)
    assert sam.calls[-1] == ("boxes", ((60, 60, 70, 70), (80, 10, 90, 20), (26, 40, 36, 50), (42, 32, 50, 40),
                                       (12, 50, 22, 60), (40, 75, 100, 100), (40, 50, 50, 60)))
    assert [(f["box_no"], f["id"]) for f in out["found"]] == [(1, "s0-i3"), (2, "s0-i4"), (3, "s0-i5"), (5, "s0-i6"),
                                                              (6, "s0-i7")]
    assert out["found"][2]["box"] == EDGE and out["found"][2]["check_this"] is True
    assert "check_this" not in out["found"][0] and out["found"][3]["check_this"] is True  # in the row: zoom on it
    assert out["notes"] == ["box 4: mostly crop", "box 6: large patch; counted as one, approximate",
                            "box 7: nothing found"]
    assert state.instances["s0-i3"].prompt == "weed 1" and state.instances["s0-i3"].role == "weed"
    assert state.steps[-1].summary == "Mark weed: 5 of 7 boxes kept"


def test_segment_phrase_adds_crop_or_checked_weeds():
    scene = field_scene()
    sam = FakeSam(concepts={"crop row": [mask_record("0", scene["row"], 0.7)],
                            "yellow flower": [mask_record("0", scene["weed_b"]),
                                       mask_record("1", scene["plant"] & rect(40, 30, 50, 40))]})
    runner, state = _runner(sam)
    _mark(runner, "crop", PLANT)
    crop = json.loads(runner.run("segment", {"phrase": " crop row ", "kind": "crop"}).output)
    assert crop["total_found"] == 1 and state.instances["s0-i2"].role == "crop"
    weeds = json.loads(runner.run("segment", {"phrase": "yellow flower", "kind": "weed"}).output)
    assert weeds["total_found"] == 1 and weeds["notes"] == ["match 2: mostly crop"]
    assert state.instances["s0-i3"].prompt == "yellow flower"
    assert state.steps[-1].summary == 'Segment "yellow flower": 1 found, 1 dropped as crop'
    assert runner.run("segment", {"phrase": "plant", "kind": "all_plants"}).output.startswith("error:")


def test_mark_rejects_bad_boxes_before_calling_sam():
    sam = FakeSam()
    runner, _ = _runner(sam)
    one = [{"box": WEED_A, "label": "w"}]
    assert runner.run("mark", {"kind": "weed", "boxes": []}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": one * 17}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [{"box": [1, 2, 3], "label": "w"}]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "weed", "boxes": [[600, 600, 700, 700]]}).output.startswith("error:")
    assert runner.run("mark", {"kind": "plant", "boxes": one}).output.startswith("error:")
    assert sam.calls == []


def test_sixteen_weed_boxes_make_the_count_approximate():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, *[BARE] * 15)
    assert state.box_cap_hit and len(state.unclassified()) == 1
    metrics = json.loads(runner.run("measure", {"view_type": "close_up"}).output)
    assert metrics["weed_count"] == 1 and metrics["count_approximate"] is True


def test_classify_resolves_synonyms_and_rejects_unknown():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, WEED_B)
    ok = runner.run("classify", {"items": [{"instance_id": "s0-i2", "role": "weed", "species_id": "Marestail",
                                            "height_in": 2.5, "leaf_count": 6, "id_confidence": "low"}]})
    assert json.loads(ok.output) == {"ok": ["s0-i2"]}
    inst = state.instances["s0-i2"]
    assert inst.species_id == "horseweed" and inst.check_this is True
    bad = runner.run("classify", {"items": [{"instance_id": "s0-i1", "role": "weed", "species_id": "dragonweed",
                                             "id_confidence": "high"}]})
    assert bad.output.startswith("error:") and state.instances["s0-i1"].id_confidence is None


def _verdict(instance_id, role="weed", species=None, confidence="medium", **extra):
    return {"instance_id": instance_id, "role": role, "species_id": species, "height_in": None, "leaf_count": None,
            "id_confidence": confidence, **extra}


def test_classify_coerces_numbers_and_ignores_crop_names():
    runner, state = _runner()
    _mark(runner, "crop", PLANT)
    _mark(runner, "weed", WEED_A, WEED_B)  # s0-i1 is the crop plant; s0-i2 and s0-i3 are weeds
    ok = runner.run("classify", {"items": [
        _verdict("s0-i2", species="", confidence="high", height_in="3", leaf_count="4.0"),
        _verdict("s0-i1", role="crop", species="soybean", confidence="high"),  # a crop's name is not a weed species
        _verdict("s0-i3", species="Marestail", height_in=2.5, leaf_count=6)]})
    assert json.loads(ok.output) == {"ok": ["s0-i2", "s0-i1", "s0-i3"]}
    weed, crop, other = (state.instances[i] for i in ("s0-i2", "s0-i1", "s0-i3"))
    assert (weed.height_in, weed.leaf_count, weed.species_id) == (3.0, 4, None)
    assert (crop.role, crop.species_id) == ("crop", None)
    assert (other.species_id, other.height_in, other.leaf_count) == ("horseweed", 2.5, 6)


def test_classify_rejects_bad_values_before_changing_anything():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, WEED_B)
    good = _verdict("s0-i1", species="waterhemp", confidence="high", height_in=3)
    for bad in ({"height_in": "tall"}, {"height_in": -2}, {"height_in": "nan"}, {"height_in": True},
                {"leaf_count": "many"}, {"leaf_count": [3]}, {"id_confidence": "sure"}, {"role": "tree"}):
        out = runner.run("classify", {"items": [good, _verdict("s0-i2", **bad)]})
        assert out.output.startswith("error:"), bad
        assert all(i.id_confidence is None and i.height_in is None for i in state.instances.values()), bad


def test_a_confident_verdict_clears_the_check_flag_but_patches_stay_flagged():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    _mark(runner, "weed", EDGE, WEED_A, PATCH)  # s0-i2 sits partly in the row canopy, s0-i4 is a patch
    assert [state.instances[i].check_this for i in ("s0-i2", "s0-i3", "s0-i4")] == [True, False, True]
    runner.run("classify", {"items": [_verdict("s0-i2", confidence="high"), _verdict("s0-i3", confidence="low"),
                                      _verdict("s0-i4", confidence="high")]})
    assert [state.instances[i].check_this for i in ("s0-i2", "s0-i3", "s0-i4")] == [False, True, True]


def test_finish_tidies_stage_and_notes():
    runner, state = _runner()
    runner.run("measure", {"view_type": "close_up"})
    assert runner.run("finish", {"crop_stage": " V3 ", "notes": None}).finished
    assert (state.crop_stage, state.notes) == ("V3", "")
    assert runner.run("finish", {"crop_stage": "  ", "notes": 7}).finished
    assert (state.crop_stage, state.notes) == (None, "7")


class _ZeroBasedSam(FakeSam):
    """Numbers its objects from 0, though the first box's object should be 1."""

    def segment_boxes(self, image, boxes):
        return [SamRecord(str(int(r.object_id) - 1), r.box, r.confidence, r.mask)
                for r in super().segment_boxes(image, boxes)]


def test_mark_logs_object_ids_sam_was_not_asked_for():
    runner, state = _runner(_ZeroBasedSam())
    _mark(runner, "weed", WEED_A, WEED_B)
    odd = [s for s in state.steps if s.tool == "mark" and not s.ok]
    assert [s.summary for s in odd] == ["SAM returned unexpected object ids ['0']"]
    runner, state = _runner()  # ids 1..N, as asked: nothing to log
    _mark(runner, "weed", WEED_A, WEED_B)
    assert all(s.ok for s in state.steps)


def test_segment_lists_up_to_a_hundred_matches():
    cells = [(n % 10 * 10, n // 10 * 10) for n in range(100)]
    matches = [mask_record(str(n), rect(x, y, x + 4, y + 4)) for n, (x, y) in enumerate(cells)]
    runner, _ = _runner(FakeSam(concepts={"dandelion": matches}))
    out = json.loads(runner.run("segment", {"phrase": "dandelion", "kind": "weed"}).output)
    assert out["total_found"] == 100 and len(out["found"]) == 100


def test_zoom_on_an_instance_or_a_thousandths_box():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    result = runner.run("zoom", {"instance_id": "s0-i1"})
    assert result.images and result.images[0][1].startswith("data:image/jpeg;base64,")
    boxed = runner.run("zoom", {"box": [500, 500, 600, 600]})
    close_up = Image.open(io.BytesIO(base64.b64decode(boxed.images[0][1].split(",", 1)[1])))
    assert close_up.size == (12, 12)  # pixels (50, 50, 60, 60), padded by 15%
    assert state.steps[-1].summary == "Zoom on box [500, 500, 600, 600]"


def test_unknown_tool_and_bad_arguments():
    runner, _ = _runner()
    assert runner.run("explode", {}).output.startswith("error:")
    assert runner.run("segment_at", {"kind": "weed", "box": WEED_A}).output.startswith("error:")  # not a tool
    assert runner.run("zoom", {"__invalid__": "{"}).output.startswith("error:")
    assert runner.run("measure", {"view": "row"}).output.startswith("error:")


def test_finish_needs_classified_weeds_and_measure():
    runner, state = _runner()
    _mark(runner, "crop", ROW)
    _mark(runner, "weed", WEED_A, WEED_B, EDGE)
    refused = runner.run("finish", {"crop_stage": None, "notes": ""})
    assert refused.output.startswith("error:") and "s0-i2" in refused.output and not refused.finished
    runner.run("classify", {"items": [{"instance_id": i, "role": "weed", "species_id": None, "height_in": None,
                                       "leaf_count": None, "id_confidence": "medium"}
                                      for i in ("s0-i2", "s0-i3", "s0-i4")]})
    assert runner.run("finish", {"crop_stage": None, "notes": ""}).output == "error: call measure first"
    metrics = json.loads(runner.run("measure", {"view_type": "row"}).output)
    assert metrics["weed_count"] == 3 and metrics["in_row_weed_share"] > 0
    assert runner.run("finish", {"crop_stage": "V3", "notes": "ok"}).finished
    assert state.crop_stage == "V3" and state.finished


def test_two_bad_requests_fail_the_stop():
    runner, _ = _runner(FakeSam(fail={"boxes"}))
    args = {"kind": "weed", "boxes": [{"box": WEED_A, "label": "w"}]}
    assert runner.run("mark", args).output.startswith("error:")
    with pytest.raises(StopFailed):
        runner.run("mark", args)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.tools'`

- [ ] **Step 3: Write the implementation**

`scout/agent/tools.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tools.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add scout/agent/tools.py tests/test_tools.py
git commit -m "feat: add the six stop-agent tools with box marking"
```

### Task 17: Prompts and the stop agent loop

Muse Spark takes 13–16 s per turn, so the prompt aims for 6 turns or fewer within the 12-call budget:
1. Mark the crop and the weeds in the first turn.
2. Zoom on at most 2 doubtful instances, and mark missed weeds once more if needed.
3. Classify every weed in one call.
4. Measure, then finish. Classify, measure and finish can share one turn.

Within a turn, crop `mark`/`segment` calls run first, so the weed check sees the crop. Then come the other tools, then `measure`, then `finish`. Outputs still go back in the model's order. `run_stop` turns any exception (`StopFailed`, `BadRequest`, `LlmError`, timeouts, dropped connections) into a failed stop that keeps its step log, so one bad stop never takes down the walk. A stop where the agent never produced an instance or ran `measure` is also `failed`, not `no_crop_found`. `finalize` keeps marked-but-unclassified weeds as weeds with `check_this`. An alfalfa stop needs no crop boxes to count, because alfalfa is solid-seeded.

**Files:**
- Create: `scout/agent/prompts.py`, `scout/agent/stop_agent.py`
- Modify: `tests/helpers.py` (append)
- Test: `tests/test_stop_agent.py`

- [ ] **Step 1: Append the fake LLM and write the failing test**

Append to `tests/helpers.py`:
```python
from scout.agent.llm import LlmTurn


class FakeLlm:
    """Plays back turns in order; an exception in the list is raised in its place, like a failed API call."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def turn(self, input, **kwargs):
        self.requests.append({"input": input, **kwargs})
        turn = self.turns.pop(0) if self.turns else LlmTurn(f"r-end{len(self.requests)}", [], "")
        if isinstance(turn, Exception):
            raise turn
        return turn
```

`tests/test_stop_agent.py`:
```python
import itertools
from dataclasses import replace

from scout.agent.llm import LlmError, LlmTurn, ToolCall
from scout.agent.prompts import stop_instructions
from scout.agent.stop_agent import run_stop
from scout.config import Settings
from scout.retry import BadRequest
from tests.helpers import FakeLlm, FakeSam, make_image, small_weedbook

SETTINGS = Settings(api_key="test")
MARK_CROP = ToolCall("c1", "mark", {"kind": "crop", "boxes": [{"box": [100, 0, 300, 1000], "label": "soybean row"},
                                                              {"box": [400, 300, 520, 420], "label": "soybean"}]})
MARK_WEEDS = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "waterhemp"},
                                                               {"box": [800, 100, 900, 200], "label": "marestail"},
                                                               {"box": [420, 320, 500, 400], "label": "pigweed"}]})


def _classify(*ids, species=None, confidence="medium"):
    return ToolCall("c-cls", "classify", {"items": [
        {"instance_id": i, "role": "weed", "species_id": species, "height_in": None, "leaf_count": None,
         "id_confidence": confidence} for i in ids]})


def _close(view_type="row"):
    return [ToolCall("c-m", "measure", {"view_type": view_type}),
            ToolCall("c-f", "finish", {"crop_stage": None, "notes": "done"})]


class TimeoutSam(FakeSam):
    """A SAM whose calls time out: the sky prompt, or every box request after the first."""

    def __init__(self, sky=False, later_boxes=False):
        super().__init__()
        self.sky, self.later_boxes = sky, later_boxes

    def segment_concept(self, image, phrase):
        if self.sky and phrase == "sky":
            self.calls.append(("concept", phrase))
            raise TimeoutError("sky timed out")
        return super().segment_concept(image, phrase)

    def segment_boxes(self, image, boxes):
        if self.later_boxes and any(kind == "boxes" for kind, _ in self.calls):
            raise TimeoutError("boxes timed out")
        return super().segment_boxes(image, boxes)


def _run(llm, crop="soybean", settings=SETTINGS, sam=None, **kwargs):
    return run_stop(make_image(100, 100), crop, llm, sam or FakeSam(), small_weedbook(), settings, **kwargs)


def test_full_stop():
    llm = FakeLlm([
        LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
        LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i4"})], ""),
        LlmTurn("r3", [ToolCall("c4", "classify", {"items": [
            {"instance_id": "s0-i3", "role": "weed", "species_id": "waterhemp", "height_in": 3, "leaf_count": 5,
             "id_confidence": "high"},
            {"instance_id": "s0-i4", "role": "weed", "species_id": "marestail", "height_in": None, "leaf_count": None,
             "id_confidence": "low"}]})], ""),
        LlmTurn("r4", [ToolCall("c5", "measure", {"view_type": "top_down"}),
                       ToolCall("c6", "finish", {"crop_stage": "V3", "notes": "Two weeds between rows."})], ""),
    ])
    steps = []
    result, _ = _run(llm, on_step=steps.append)
    assert result.status == "ok" and result.weed_count == 2  # the pigweed box sat on a soybean plant
    assert result.species == {"waterhemp": 1, "horseweed": 1}
    assert result.crop_stage == "V3" and result.view_type == "top_down" and result.density_per_m2 is not None
    assert result.flags == ["Check s0-i4: low confidence"]
    assert len(steps) == len(result.steps) == 6
    first, third = llm.requests[0], llm.requests[2]
    assert [t["name"] for t in first["tools"]][:2] == ["segment", "mark"] and "0 to 1000" in first["instructions"]
    assert third["previous_response_id"] == "r2"
    assert [part["type"] for part in third["input"]] == ["function_call_output", "message"]  # the zoom image rides
    # in a user message after the tool output


def test_crop_marks_run_first_within_a_turn():
    llm = FakeLlm([LlmTurn("r1", [MARK_WEEDS, MARK_CROP], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, state = _run(llm)
    assert [s.summary for s in result.steps[:2]] == ["Mark crop: 2 of 2 boxes kept", "Mark weed: 2 of 3 boxes kept"]
    assert [p["call_id"] for p in llm.requests[1]["input"]] == ["c2", "c1"]  # outputs keep the model's order
    assert result.status == "ok" and result.weed_count == 2 and state.instances["s0-i1"].role == "crop"


def test_call_limit_closes_stop_and_flags_unclassified_weeds():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i3"})], "")])
    result, _ = _run(llm, settings=replace(SETTINGS, max_tool_calls=3))
    assert result.status == "ok" and result.weed_count == 2
    assert any(step.tool == "limit" for step in result.steps)
    assert result.flags == ["Check s0-i3: marked but never classified", "Check s0-i4: marked but never classified"]


PATCH = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [400, 750, 1000, 1000], "label": "foxtail"}]})


def test_patch_makes_the_count_approximate():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, PATCH], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("top_down")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is True
    assert "Weed count is approximate: some weeds merged into patches" in result.flags


def test_a_big_plant_in_a_close_up_is_not_a_patch():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, PATCH], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("close_up")], "")])
    result, _ = _run(llm)
    assert result.weed_count == 1 and result.count_approximate is False
    assert result.flags == ["Check s0-i3: may be crop or several plants"]


def test_repeated_bad_request_fails_stop_and_keeps_steps():
    call = [ToolCall("c1", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "w"}]})]
    llm = FakeLlm([LlmTurn("r1", call, ""), LlmTurn("r2", call, "")])
    result, _ = _run(llm, sam=FakeSam(fail={"boxes"}))
    assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "mark", "error"]


def test_model_errors_fail_the_stop_and_keep_steps():
    for error in (LlmError("response r2 was incomplete"), BadRequest("image too large"),
                  TimeoutError("request timed out"), ConnectionError("connection dropped")):
        result, _ = _run(FakeLlm([LlmTurn("r1", [MARK_CROP], ""), error]))
        assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "error"]


def test_prose_reply_is_nudged_then_fails_the_stop():
    llm = FakeLlm([LlmTurn("r1", [], "I think there are weeds."), LlmTurn("r2", [], "Still prose.")])
    result, _ = _run(llm)
    assert result.status == "failed" and len(llm.requests) == 2  # the agent produced nothing: not "no crop found"
    assert result.flags == ["This stop could not be analyzed"]


def test_no_crop_found_needs_a_stop_the_agent_worked_on():
    llm = FakeLlm([LlmTurn("r1", _close("top_down"), "")])
    result, _ = _run(llm)
    assert result.status == "no_crop_found" and result.weed_count == 0


def test_sky_timeout_does_not_fail_the_stop():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close("row")], "")])
    sam = TimeoutSam(sky=True)
    result, state = _run(llm, sam=sam)
    assert result.status == "ok" and result.weed_count == 2 and state.sky is None
    assert [s.tool for s in result.steps].count("sky") == 1  # logged once
    assert sam.calls.count(("concept", "sky")) == 1  # asked once, by measure; finalize does not ask again


def test_a_sam_timeout_fails_only_this_stop_and_keeps_steps():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP], ""), LlmTurn("r2", [MARK_WEEDS], "")])
    result, _ = _run(llm, sam=TimeoutSam(later_boxes=True))
    assert result.status == "failed" and [s.tool for s in result.steps] == ["mark", "error"]
    assert result.steps[-1].summary.startswith("TimeoutError")


def test_measure_and_finish_run_last_whatever_order_the_model_lists_them():
    finish, measure = _close()[1], _close()[0]
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""), LlmTurn("r2", [_classify("s0-i3", "s0-i4")], ""),
                   LlmTurn("r3", [finish, measure], "")])
    result, state = _run(llm)
    assert len(llm.requests) == 3 and state.finished and result.notes == "done"
    assert [s.tool for s in result.steps][-2:] == ["measure", "finish"]
    # classify, measure and finish together in one turn, listed backwards
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [finish, measure, _classify("s0-i3", "s0-i4")], "")])
    result, state = _run(llm)
    assert len(llm.requests) == 2 and state.finished and result.weed_count == 2 and result.flags == []
    assert [s.tool for s in result.steps] == ["mark", "mark", "classify", "measure", "finish"]


def test_time_limit_closes_the_stop_after_the_turn_in_progress():
    ticks = itertools.count(0, 100)  # every clock reading is 100 s later: the 240 s budget is gone after one turn
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""), LlmTurn("r2", [_classify("s0-i3", "s0-i4")], "")])
    result, _ = _run(llm, clock=lambda: float(next(ticks)))
    assert len(llm.requests) == 1 and result.status == "ok"
    assert result.steps[-1].tool == "limit" and result.steps[-1].ok is False


def test_weed_free_top_down_stop_has_zero_density():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP], ""), LlmTurn("r2", _close("top_down"), "")])
    result, _ = _run(llm)
    assert result.status == "ok" and result.weed_count == 0 and result.flags == []
    assert result.density_per_m2 == 0.0 and result.species_density == {}


def test_alfalfa_needs_no_crop_boxes():
    weeds = ToolCall("c1", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "dandelion"},
                                                              {"box": [800, 100, 900, 200], "label": "dandelion"}]})
    llm = FakeLlm([LlmTurn("r1", [weeds], ""), LlmTurn("r2", [_classify("s0-i1", "s0-i2"), *_close("close_up")], "")])
    result, _ = _run(llm, crop="alfalfa")
    assert result.status == "ok" and result.weed_count == 2 and result.in_row_weed_share is None


def test_instructions_use_thousandths_and_crop_hints():
    soybean = stop_instructions("soybean", ["waterhemp", "horseweed"], 12)
    alfalfa = stop_instructions("alfalfa", ["dandelion"], 12)
    assert "0 to 1000" in soybean and "waterhemp, horseweed" in soybean and "12 tool calls" in soybean
    assert '"crop row"' in soybean and "crop boxes are optional" in alfalfa
    assert "crop boxes are optional" not in soybean
    assert "no weeds" in soybean and '"yellow flower"' in soybean and '"flower"' not in soybean  # alfalfa blooms
    assert "estimate" in soybean and "null" in soybean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stop_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scout.agent.prompts'`

- [ ] **Step 3: Write the prompts**

`scout/agent/prompts.py`:
```python
"""Instructions for the stop agent and the field explanation."""


def stop_instructions(crop: str, species_ids: list[str], max_calls: int) -> str:
    if crop == "alfalfa":
        crop_boxes = "Alfalfa is solid-seeded, so crop boxes are optional."
    else:
        crop_boxes = (f"For the crop, box each visible {crop} row (one tall box per row is fine) or each {crop} "
                      'plant. In a view along the rows, segment "crop row" with kind "crop" also works.')
    return f"""You are an agronomist scouting a {crop} field. You see one photo. Find every weed, identify it, then \
have code measure it.

Coordinates: every box you give or get is [x1, y1, x2, y2] from 0 to 1000 on both axes. x runs from the left edge \
(0) to the right edge (1000), and y from the top (0) to the bottom (1000).

Procedure (aim for 6 turns or fewer):
1. In your first turn, call mark with kind "crop" and mark with kind "weed"; both calls can go in the same turn. \
{crop_boxes} For the weeds, give a tight box around every weed, one box per plant, nearest or largest first, at \
most 16. If you see no weeds, skip the weed mark: never box crop, soil or residue as a weed. A distinctive \
flowering weed can also be found by phrase: segment "dandelion" or "yellow flower" with kind "weed".
2. Read the results: "mostly crop" means that box was on crop and was dropped; "nothing found" means SAM found no \
plant there. Zoom on at most 2 doubtful instances (check_this is true); each close-up arrives at your next turn. If \
you missed weeds, mark them once more.
3. classify every weed instance in one call. Use a species_id from this list only: {", ".join(species_ids)}. Use \
null when you cannot tell. Give leaf_count when you can count the leaves. height_in is an estimate from visible \
size cues (a {crop} plant, a leaf, the row spacing): use null when there is no cue. Use role "crop" for an \
instance that is really crop and "ignore" for soil, residue, or shadow.
4. measure with the view_type that fits: row (looking along rows), top_down (camera pointing straight down), \
close_up (a few plants fill the frame), wide (field to the horizon). Then finish with the crop growth stage if \
you can tell (for example V3) and one sentence of notes. classify, measure and finish can all go in one turn, in \
that order; if nothing needs a zoom, do that in your second turn.

Rules: never invent measurements; cover, counts and density come only from measure. You have at most {max_calls} \
tool calls. Always respond with tool calls, not prose. Every coordinate is 0 to 1000."""


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

from scout.agent.llm import Llm, ToolCall, user_message
from scout.agent.prompts import stop_instructions
from scout.agent.state import StopState
from scout.agent.tools import TOOL_SCHEMAS, ToolRunner
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.models import AgentStep, Instance, StopResult
from scout.vision.measure import compute_metrics

FAILED_FLAG = "This stop could not be analyzed"


def run_order(calls: list[ToolCall]) -> list[ToolCall]:
    """Crop marks and segments run first, so the weed check sees the crop; then the other tools; measure and finish
    last, so they see the turn's verdicts. The order is stable within each rank."""
    def rank(call: ToolCall) -> int:
        if call.name in ("mark", "segment") and call.arguments.get("kind") == "crop":
            return 0
        return {"measure": 2, "finish": 3}.get(call.name, 1)
    return sorted(calls, key=rank)


def _failed(state: StopState) -> StopResult:
    image = state.image
    return StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, steps=state.steps,
                      flags=[FAILED_FLAG], status="failed")


def run_stop(image: StopImage, crop: str, llm: Llm, sam, weeds: WeedBook, settings: Settings,
             on_step: Callable[[AgentStep], None] | None = None,
             clock: Callable[[], float] = time.monotonic) -> tuple[StopResult, StopState]:
    state = StopState(image, crop, on_step=on_step, clock=clock)
    runner = ToolRunner(state, sam, weeds, max_calls=settings.max_tool_calls)
    instructions = stop_instructions(crop, weeds.ids(), settings.max_tool_calls)
    pending: list[dict] = [user_message(f"Stop {image.index + 1}: one photo of a {crop} field. Boxes are 0 to 1000 "
                                        "on both axes.", [image.work_data_url])]
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
            outputs, images = {}, []
            for call in run_order(turn.calls):
                state.calls += 1
                result = runner.run(call.name, call.arguments)
                outputs[call.call_id] = result.output
                images += result.images
            pending = [{"type": "function_call_output", "call_id": call.call_id, "output": outputs[call.call_id]}
                       for call in turn.calls]  # answered in the model's own order
            if images:
                pending.append(user_message("Close-ups you asked for: " + "; ".join(c for c, _ in images),
                                            [url for _, url in images]))
        return finalize(state, sam), state
    except Exception as error:  # StopFailed, BadRequest, LlmError, timeouts, dropped connections: this stop only
        state.log("error", f"{type(error).__name__}: {error}", ok=False)
        return _failed(state), state


def _flag(inst: Instance) -> str:
    if inst.id_confidence is None:
        return f"Check {inst.id}: marked but never classified"
    if inst.id_confidence == "low":
        return f"Check {inst.id}: low confidence"
    return f"Check {inst.id}: may be crop or several plants"


def finalize(state: StopState, sam) -> StopResult:
    """Weeds that were marked but never classified stay weeds, flagged; measurements are recomputed."""
    if not state.instances and state.view_type is None:  # the agent got nowhere: its failure, not a stop without crop
        return _failed(state)
    for inst in state.instances.values():
        if inst.role == "weed" and inst.id_confidence is None:
            inst.check_this = True
    view = state.view_type or "row"
    if view in ("row", "wide"):
        state.ensure_sky(sam)
    m = compute_metrics(list(state.instances.values()), state.masks, state.size, view, state.sky, state.row_band())
    instances = list(state.instances.values())
    flags = [_flag(i) for i in instances if i.role == "weed" and i.check_this]
    if m.count_approximate or (state.has_patch() and view != "close_up"):
        flags.append("Weed count is approximate: some weeds merged into patches")
    if state.box_cap_hit:
        flags.append("Weed count is a floor: a mark call used all 16 boxes")
    has_crop = any(i.role == "crop" for i in instances)
    # Alfalfa is solid-seeded and its crop boxes are optional, so a worked alfalfa stop counts without them.
    worked = state.view_type is not None or any(i.role == "weed" for i in instances)
    return StopResult(
        stop_index=state.image.index, image_hash=state.image.hash, view_type=view, gps=state.image.gps,
        crop_stage=state.crop_stage, weed_cover_pct=m.weed_cover_pct, crop_cover_pct=m.crop_cover_pct,
        weed_count=m.weed_count, count_approximate=m.count_approximate or state.count_approximate(view),
        density_per_m2=m.density_per_m2, in_row_weed_share=m.in_row_weed_share, species=m.species,
        species_cover_pct=m.species_cover_pct, species_density=m.species_density,
        species_max_height_in=m.species_max_height_in, max_height_in=m.max_height_in, flags=flags,
        notes=state.notes, steps=state.steps,
        status="ok" if has_crop or (state.crop == "alfalfa" and worked) else "no_crop_found",
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_stop_agent.py -v`
Expected: 16 passed

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
        try:
            image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
            result, state = run_stop(image, args.crop, llm, sam, weeds, settings,
                                     on_step=lambda s: print(f"  [{s.tool}] {s.summary}"))
            state.overlay().save(OUT / f"{stem}.png")
        except Exception as error:  # an unreadable photo or a dropped connection must not end the batch
            print(f"  !! {stem} failed: {type(error).__name__}: {error}")
            continue
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
Expected: each photo prints its steps and a summary line, and `run_out/` holds overlays and JSON. A photo that fails prints `!!` and the batch moves on.

- [ ] **Step 3: Tune**

Open the overlays and look for three common problems:
- **Weed boxes on crop:** boxes dropped as "mostly crop", and weeds flagged inside a row, are expected. Crop leaves still kept as weeds after zooming are not: tighten the weed wording in step 1 ("a tight box around the weed only").
- **Weeds missed:** if visible weeds carry no box, stress "one box per plant, nearest or largest first" in step 1 and the second `mark` in step 2.
- **Too many turns:** each turn takes 13–16 s. If stops use more than 6 turns, stress putting both marks in the first turn and measure + finish in the last.

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
Expected: 26 passed (`validate()` returns `[]`)

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


def _stop(i, species=None, cover=None, density=None, heights=None, status="ok", approximate=False):
    species, cover = species or {}, cover or {}
    return StopResult(stop_index=i, image_hash=f"h{i}", species=species, species_cover_pct=cover,
                      species_density=density, species_max_height_in=heights or {},
                      weed_cover_pct=sum(cover.values()), weed_count=sum(species.values()), status=status,
                      count_approximate=approximate)


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


def test_approximate_count_also_grades_cover():
    mat = [_stop(0, {"velvetleaf": 1}, {"velvetleaf": 40.0}, density={"velvetleaf": 0.5}, approximate=True)]
    d = decide(mat, BOOK)
    assert d.summaries[0].outcome == "treat" and "% cover" in d.rule_trace[0]
    exact = [_stop(0, {"velvetleaf": 1}, {"velvetleaf": 40.0}, density={"velvetleaf": 0.5})]
    assert decide(exact, BOOK).summaries[0].outcome == "no_action"


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


def _grade(value: float, limit: float) -> Outcome:
    if value >= limit:
        return "treat"
    return "scout_again" if value >= SCOUT_AGAIN_FACTOR * limit else "no_action"


def _species_outcome(info: WeedInfo, present: int, n_ok: int, mean_cover: float,
                     mean_density: float | None, density_approx: bool = False) -> tuple[Outcome, str]:
    """Grade density when it is known, else cover. When the count behind density is approximate (a mat of
    plants segmented as one piece), cover is graded too and the more severe outcome wins."""
    name = info.common_name
    if info.zero_tolerance and present:
        return "treat", f"{name}: zero tolerance; present at {present} of {n_ok} stops -> treat"
    if info.action_cover_pct is not None:
        cover = (mean_cover, info.action_cover_pct, "% cover", "sourced threshold")
    else:
        cover = (mean_cover, DEFAULT_ACTION_COVER_PCT, "% cover", "tool default, not from a source")
    checks = [cover]
    if info.action_density is not None and mean_density is not None:
        density = (mean_density, info.action_density, "plants/m²", "sourced threshold")
        checks = [density, cover] if density_approx else [density]
    value, limit, unit, source = max(checks, key=lambda check: SEVERITY[_grade(check[0], check[1])])
    outcome = _grade(value, limit)
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
        approx = any(s.count_approximate and sid in s.species for s in top_down)
        outcome, line = _species_outcome(weeds.get(sid), present, n, mean_cover, mean_density, approx)
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
Expected: 6 passed

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
<p>Green: crop · Magenta: weeds · Gray: in-row zone · Yellow box: check this plant</p>
<div class="stops">
{% for s in report.stops %}
  <div class="card">
    <img src="{{ overlay_url(s.stop_index) }}" alt="Stop {{ s.stop_index + 1 }}: crop in green, weeds in magenta">
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
                # In a thread: decoding takes 0.5-1.5 s per photo and would otherwise stall every SSE stream.
                images.append(await asyncio.to_thread(load_image, index, data, settings.full_max_px, settings.work_max_px))
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

- [ ] **Step 1: Pin dependencies, then write the Dockerfile**

Hugging Face rebuilds the image from `requirements.txt` on every push, so a new upstream release in deadline week could break the live site. Before deploying, change every line of `requirements.txt` from `>=` to `==` with the version the tests last passed against (read them from `.venv/Scripts/python.exe -m pip freeze`). Pin only the 17 top-level names, not the whole freeze, because a Windows freeze can list packages that don't exist for the Linux image. Rerun `python -m pytest -q`, then commit `chore: pin dependency versions for deploy`.

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
1. **Mark.** Muse Spark boxes the crop rows and every weed, and SAM 3.1 segments all the boxes in one request. Code drops "weeds" that lie mostly on crop plants.
2. **Check.** The agent zooms into doubtful plants, and names each weed and its size.
3. **Measure.** Code computes weed cover, counts, density (straight-down shots), and in-row vs between-row share. The model never writes a number.
4. **Decide.** Rules compare species against sourced thresholds (zero tolerance for resistant amaranths), and choose no action, scout again, or treat (spot or broadcast).
5. **Ground in labels.** A table built from EPA label PDFs (each record's quote verified on its page) lists products labeled for this crop, weed, and size.

## Meta models used
- `muse-spark-1.3`: agent tool loop, zoom inspection, label extraction to structured JSON, field explanation.
- `sam-3.1`: multi-object box prompts, concept prompts ("sky", "crop row", distinct weeds), confidence scores.

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
        row.update(status="failed", pred_weeds_present="", pred_main_species="", pred_cover_bucket="")
        try:
            image = load_image(0, (ROOT / row["photo"]).read_bytes(), settings.full_max_px, settings.work_max_px)
            result, _ = run_stop(image, row["crop"], llm, sam, weeds, settings)
        except Exception as error:  # an unreadable photo or a dropped connection must not end the run
            print(f"{row['photo']}: failed ({type(error).__name__}: {error})")
            continue
        row["status"] = result.status
        if result.status == "failed":
            print(f"{row['photo']}: failed (the stop could not be analyzed)")
            continue
        covers = result.species_cover_pct
        row["pred_weeds_present"] = "y" if result.weed_count else "n"
        row["pred_main_species"] = max(covers, key=covers.get) if covers else ""
        row["pred_cover_bucket"] = bucket(result.weed_cover_pct)
        print(f"{row['photo']}: {row['pred_weeds_present']} {row['pred_main_species']} {row['pred_cover_bucket']}")
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(score([row for row in rows if row["status"] != "failed"]))


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
Expected: one line per photo, then a dict of agreement rates. Failed photos print a reason and are left out of the rates. Use misses to tune `prompts.py`, rerun, and note the final numbers in `docs/spike-notes.md`.

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
0:20–1:20  Live: upload a 6-photo soybean walk on the deployed URL. Agent steps stream: mark crop, mark
           weeds, zoom on a doubtful plant, identify waterhemp ~3 in, measure.
1:20–2:10  Report: overlays (crop green, weeds magenta), cover/density/frequency cards, decision "Treat (spot)",
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
