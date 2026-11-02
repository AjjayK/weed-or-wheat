# Complete Weed Marking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The stop agent marks every weed it can see (boxes for plants it can tell apart, weed-area patches for carpets), flags what it may have missed, and warns when the photo shows another crop than the farmer picked.

**Architecture:** A new `survey` tool records the crop the agent sees and its weed estimate. `mark` loses SAM's 16-box cap (code splits boxes into parallel 16-box SAM requests) and gains a `patch` kind measured by code as green, non-crop pixels inside the box (`scout/vision/patches.py`). Every mark answers with counts, a hint when marking is under the estimate, and a picture of the current overlay. `finalize` adds an unmarked-weeds flag and a crop-mismatch flag; the field report shows a crop-mismatch banner.

**Tech Stack:** Python 3.10, NumPy, SciPy (`ndimage`), Pillow, Pydantic v2, FastAPI, Jinja2, pytest. Spec: `docs/superpowers/specs/2026-11-01-complete-marking-design.md`.

**Conventions for every task:**
- Run tests with the project venv: `.venv/Scripts/python -m pytest ...` (system Python lacks the dependencies).
- Tests never call the network.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never print, echo or log `.env` or `MODEL_API_KEY`.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `scout/config.py` | Modify | Budget defaults: 30 calls, 360 s, 10 stops at once |
| `scout/models.py` | Modify | `Instance.source` gains `"patch"`, `Instance.plant_count`, `StopResult.crop_seen`, `FieldReport.crop_mismatch` |
| `scout/vision/patches.py` | Create | Green mask; patch mask = green ∩ box − dilated crop, without specks |
| `scout/vision/overlay.py` | Modify | Blue outlines for weed areas |
| `scout/vision/measure.py` | Modify | Count `plant_count`; a patch makes the count approximate |
| `scout/agent/state.py` | Modify | Weed areas, refresh after crop changes, survey/finish state, unmarked check, overlay areas |
| `scout/agent/tools.py` | Modify | `survey`, `mark` (120 boxes, chunks, `patch`, feedback), `finish.all_weeds_marked`, refreshes |
| `scout/agent/stop_agent.py` | Modify | Survey runs first, one overlay picture per turn, new flags, `crop_seen` |
| `scout/agent/prompts.py` | Modify | Rewritten procedure: survey, every weed, patches, check the picture |
| `scout/agent/field_agent.py` | Modify | `crop_mismatch()` banner text |
| `scout/report/templates/report.html` | Modify | Banner above the decision; legend adds blue boxes |
| `scout/web/index.html` | Modify | Crop menu has no default |
| `scripts/run_stop.py` | Modify | `--out`, `--jobs`: run photos in parallel, print time per stop |
| `tests/helpers.py` | Modify | `field_photo()`: soil, a green row, two green shoots |
| `tests/test_*.py` | Modify/Create | Tests listed per task |
| `docs/spike-notes.md` | Modify | Live-check results |

---

### Task 1: Bigger budget

**Files:**
- Modify: `scout/config.py:18-20`
- Test: `tests/test_config.py`, `tests/test_stop_agent.py:237-242`

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, replace `test_settings_defaults` with:

```python
def test_settings_defaults():
    settings = Settings(api_key="x")
    assert settings.base_url == "https://api.meta.ai/v1"
    assert settings.work_max_px == 1024 and settings.max_tool_calls == 30
    assert settings.stop_timeout_s == 360.0 and settings.max_concurrent_stops == 10
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py::test_settings_defaults -v`
Expected: FAIL (`assert 12 == 30`).

- [ ] **Step 3: Change the defaults**

In `scout/config.py` replace the three lines:

```python
    max_tool_calls: int = 30  # generous: marking every weed takes more turns than sampling them did
    stop_timeout_s: float = 360.0  # Muse Spark takes 13-16 s per turn; a thorough stop needs 2-4 minutes
    max_concurrent_stops: int = 10  # every photo of a walk at once: the walk takes as long as its slowest stop
```

In `tests/test_stop_agent.py`, `test_time_limit_closes_the_stop_after_the_turn_in_progress` assumed a 240 s budget. Replace its first line with:

```python
    ticks = itertools.count(0, 200)  # every clock reading is 200 s later: the 360 s budget is gone after one turn
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_config.py tests/test_stop_agent.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/config.py tests/test_config.py tests/test_stop_agent.py
git commit -m "feat: bigger agent budget: 30 calls, 360 s, all photos at once" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Model fields

**Files:**
- Modify: `scout/models.py:18`, `:28`, `:57`, `:110`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_weed_areas_and_crop_seen_fields():
    area = Instance(id="s0-i9", source="patch", prompt="horsetail", box=(0, 0, 10, 10), area_px=40, plant_count=6)
    assert area.plant_count == 6
    assert Instance(id="s0-i1", source="prompt", prompt="w", box=(0, 0, 2, 2), area_px=4).plant_count == 1
    assert StopResult(stop_index=0, image_hash="a").crop_seen is None
    assert StopResult(stop_index=0, image_hash="a", crop_seen="corn").crop_seen == "corn"
    assert "crop_mismatch" in FieldReport.model_fields and FieldReport.model_fields["crop_mismatch"].default is None
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_models.py::test_weed_areas_and_crop_seen_fields -v`
Expected: FAIL (validation error: `source` must be one of concept, prompt, candidate).

- [ ] **Step 3: Add the fields**

In `scout/models.py`:

```python
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
```

In `StopResult`, after `notes: str = ""` add:

```python
    crop_seen: str | None = None  # the crop the agent saw: corn, soybean, alfalfa, other or unsure
```

In `FieldReport`, after `options: list[LabelOption]` add:

```python
    crop_mismatch: str | None = None  # a warning when photos show another crop than the walk was scouted as
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/models.py tests/test_models.py
git commit -m "feat: model fields for weed areas, plant counts and the crop seen" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Green-pixel weed areas

**Files:**
- Create: `scout/vision/patches.py`
- Test: `tests/test_patches.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patches.py`:

```python
import numpy as np
from PIL import Image, ImageDraw

from scout.vision.patches import green_mask, patch_mask

SOIL, LEAF, SHADOW, STRAW = (110, 80, 50), (60, 140, 40), (40, 40, 40), (200, 170, 120)


def _photo():
    """100x100 soil with two green shoots (5x20 px each), a shadow, straw, and a 9 px green speck."""
    image = Image.new("RGB", (100, 100), SOIL)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 24, 39), fill=LEAF)
    draw.rectangle((40, 20, 44, 39), fill=LEAF)
    draw.rectangle((60, 60, 69, 69), fill=SHADOW)
    draw.rectangle((70, 20, 79, 29), fill=STRAW)
    draw.rectangle((85, 85, 87, 87), fill=LEAF)
    return image


def test_green_mask_keeps_leaves_and_drops_soil_shadow_and_straw():
    green = green_mask(_photo())
    assert green[30, 22] and green[30, 42] and green[86, 86]
    assert not green[50, 50] and not green[65, 65] and not green[25, 75]


def test_patch_counts_green_pieces_inside_the_box():
    green = green_mask(_photo())
    no_crop = np.zeros((100, 100), bool)
    mask, pieces = patch_mask(green, (10, 10, 95, 95), no_crop)
    assert pieces == 2 and mask.sum() == 200  # the 9 px speck is noise
    mask, pieces = patch_mask(green, (10, 10, 30, 50), no_crop)
    assert pieces == 1 and mask.sum() == 100  # only the first shoot is inside


def test_patch_leaves_out_crop_and_a_margin_around_it():
    green = green_mask(_photo())
    crop = np.zeros((100, 100), bool)
    crop[20:40, 40:45] = True  # the second shoot is really a crop leaf
    mask, pieces = patch_mask(green, (10, 10, 95, 95), crop)
    assert pieces == 1 and mask.sum() == 100
    crop = np.zeros((100, 100), bool)
    crop[20:40, 46:50] = True  # crop 1 px right of the second shoot: the 3 px margin takes its columns 43-44
    mask, pieces = patch_mask(green, (10, 10, 95, 95), crop)
    assert pieces == 2 and mask.sum() == 160
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m pytest tests/test_patches.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'scout.vision.patches'`).

- [ ] **Step 3: Write the module**

Create `scout/vision/patches.py`:

```python
"""Weed areas measured from pixels: green, inside the box the agent drew, and not crop.

For a carpet of shoots too dense or small to box one by one. SAM fills a box around a thin shoot with soil; green
pixels don't: soil, residue and shadow are never green.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from scout.models import Box

GREEN_EXG = 0.08  # excess green 2g - r - b on chromatic coordinates; soil sits near 0, straw near 0.04, leaves 0.3+
CROP_MARGIN_PX = 3  # crop outlines grown this much, so leaf edges the crop mask missed don't count as weed
MIN_PIECE_PX = 20  # green specks smaller than this are noise
_EIGHT = np.ones((3, 3), dtype=bool)


def green_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    total = rgb.sum(axis=2) + 1e-6
    r, g, b = (rgb[..., i] / total for i in range(3))
    return (2 * g - r - b) > GREEN_EXG


def patch_mask(green: np.ndarray, box: Box, crop: np.ndarray) -> tuple[np.ndarray, int]:
    """Green pixels inside the box and away from crop, without specks; and how many separate pieces they form."""
    left, top, right, bottom = box
    inside = np.zeros_like(green)
    inside[top:bottom, left:right] = True
    near_crop = ndimage.binary_dilation(crop, structure=_EIGHT, iterations=CROP_MARGIN_PX)
    labels, count = ndimage.label(green & inside & ~near_crop, structure=_EIGHT)
    keep = np.bincount(labels.ravel(), minlength=count + 1) >= MIN_PIECE_PX
    keep[0] = False
    return keep[labels], int(keep.sum())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_patches.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scout/vision/patches.py tests/test_patches.py
git commit -m "feat: measure weed areas as green non-crop pixels" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Overlay outlines weed areas in blue

**Files:**
- Modify: `scout/vision/overlay.py`
- Test: `tests/test_overlay.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_overlay.py` (and add `AREA_RGB` to its import from `scout.vision.overlay`):

```python
def test_overlay_outlines_weed_areas_in_blue_under_flag_boxes():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    empty = np.zeros((10, 10), bool)
    out = render_overlay(image, empty, empty, areas=[(0, 0, 10, 10)], flagged=[(0, 0, 5, 5)])
    assert out.getpixel((9, 9)) == AREA_RGB
    assert out.getpixel((0, 0)) == FLAG_RGB  # a flag box drawn later stays visible
    assert out.getpixel((5, 5)) == (255, 255, 255)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_overlay.py -v`
Expected: FAIL (`ImportError: cannot import name 'AREA_RGB'`).

- [ ] **Step 3: Draw the areas**

Replace `scout/vision/overlay.py` from the docstring through the end with:

```python
"""Draw crop (green), weeds (magenta), the in-row band (gray), weed areas measured from pixels (blue boxes), and
boxes around plants that need a check (yellow).

Weeds are magenta, not red: red over green leaves turns olive for red-green color-blind viewers, the same as the crop.
"""
import numpy as np
from PIL import Image, ImageDraw

from scout.models import Box

CROP_RGB = (99, 153, 34)
WEED_RGB = (230, 0, 170)
BAND_RGB = (136, 135, 128)
FLAG_RGB = (255, 214, 0)
AREA_RGB = (0, 160, 255)


def render_overlay(image: Image.Image, crop: np.ndarray, weed: np.ndarray,
                   band: np.ndarray | None = None, flagged: list[Box] = (), areas: list[Box] = ()) -> Image.Image:
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, rgb, alpha in ((band, BAND_RGB, 0.25), (crop, CROP_RGB, 0.55), (weed, WEED_RGB, 0.65)):
        if mask is not None and mask.any():
            out[mask] = out[mask] * (1 - alpha) + np.array(rgb, dtype=np.float32) * alpha
    result = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    width = max(3, round(max(result.size) / 200))  # 5 px on a 1024 px photo: still visible when shown phone-width
    for color, boxes in ((AREA_RGB, areas), (FLAG_RGB, flagged)):  # flags last: a plant to check stays visible
        for left, top, right, bottom in boxes:
            if right > left and bottom > top:  # an empty mask's bbox is (0, 0, 0, 0)
                draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=width)
    return result
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_overlay.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/vision/overlay.py tests/test_overlay.py
git commit -m "feat: outline weed areas in blue on the overlay" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Measure counts the plants in a weed area

**Files:**
- Modify: `scout/vision/measure.py:84`, `:88-89`
- Test: `tests/test_measure.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_measure.py`:

```python
def test_a_weed_area_counts_its_pieces_and_makes_the_count_approximate():
    instances, masks = _setup()
    area = Instance(id="p", source="patch", prompt="horsetail", box=(40, 40, 50, 50), area_px=100, role="weed",
                    species_id="waterhemp", plant_count=7)
    masks["p"] = _mask((40, 40, 50, 50))
    m = compute_metrics(instances + [area], masks, SIZE, "close_up")
    assert m.weed_count == 9 and m.species["waterhemp"] == 8 and m.count_approximate is True
    assert m.weed_cover_pct == 3.0  # w1, w2 and the area: 100 px each
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_measure.py::test_a_weed_area_counts_its_pieces_and_makes_the_count_approximate -v`
Expected: FAIL (`assert 3 == 9`).

- [ ] **Step 3: Count `plant_count`**

In `scout/vision/measure.py`, inside the weed loop of `compute_metrics`, replace

```python
        species_counts[key] = species_counts.get(key, 0) + 1
```

with

```python
        species_counts[key] = species_counts.get(key, 0) + inst.plant_count  # a weed area holds several plants
```

and replace

```python
        if view_type != "close_up" and own_px > APPROX_PIECE_FRAC * ground_px:
            approximate = True
```

with

```python
        if inst.source == "patch" or (view_type != "close_up" and own_px > APPROX_PIECE_FRAC * ground_px):
            approximate = True  # a weed area's pieces are not exactly its plants
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_measure.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/vision/measure.py tests/test_measure.py
git commit -m "feat: count the plants in a weed area" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Stop state keeps weed areas, the survey and the unmarked check

**Files:**
- Modify: `scout/agent/state.py`
- Modify: `tests/helpers.py` (add `field_photo`)
- Test: `tests/test_state.py`

- [ ] **Step 1: Add the test photo helper**

Append to `tests/helpers.py`:

```python
from PIL import ImageDraw


def field_photo(index=0):
    """100x100 soil with a green soybean row (x 10-29) and two green weed shoots (x 60-64 and 70-74, y 60-79)."""
    image = Image.new("RGB", (100, 100), (110, 80, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 0, 29, 99), fill=(60, 140, 40))
    draw.rectangle((60, 60, 64, 79), fill=(60, 140, 40))
    draw.rectangle((70, 60, 74, 79), fill=(60, 140, 40))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return load_image(index, buf.getvalue())
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_state.py` (add `field_photo` and `rect` to its `tests.helpers` import, and `from scout.vision.overlay import AREA_RGB`):

```python
def test_weed_area_counts_green_pixels_away_from_crop():
    state = StopState(field_photo(), "soybean")
    state.add_mask(rect(10, 0, 30, 100), "prompt", "soybean row", role="crop")
    inst, note = state.add_patch((50, 50, 90, 90), "horsetail")
    assert note is None and inst.source == "patch" and inst.role == "weed" and inst.box == (50, 50, 90, 90)
    assert inst.area_px == 200 and inst.plant_count == 2 and state.patch_boxes == {inst.id: (50, 50, 90, 90)}
    assert state.has_weed_area() and state.count_approximate("close_up")
    assert state.weeds_marked() == (0, 1) and state.plants_marked() == 2


def test_a_crop_leaf_marked_later_leaves_the_weed_area():
    state = StopState(field_photo(), "soybean")
    inst, _ = state.add_patch((50, 50, 90, 90), "horsetail")
    state.add_mask(rect(70, 60, 75, 80), "prompt", "soybean leaf", role="crop")
    state.refresh_patches()
    assert inst.area_px == 100 and inst.plant_count == 1 and state.masks[inst.id].sum() == 100


def test_weed_area_on_the_row_or_on_bare_soil_is_refused():
    state = StopState(field_photo(), "soybean")
    state.add_mask(rect(10, 0, 30, 100), "prompt", "soybean row", role="crop")
    assert state.add_patch((5, 10, 35, 90), "row") == (None, "mostly crop: box the weeds between the rows, not the row")
    assert state.add_patch((35, 10, 55, 40), "soil") == (None, "no green weed pixels in this area")
    assert state.patch_boxes == {}


def test_unmarked_weeds_are_suspected_until_confirmed_and_the_estimate_is_met():
    state = StopState(field_photo(), "soybean")
    assert state.maybe_unmarked()  # finish never said every weed is marked
    state.all_weeds_marked, state.weeds_estimate = True, 10
    state.add_mask(rect(60, 60, 65, 80), "prompt", "waterhemp", role="weed")
    assert state.maybe_unmarked()  # 1 plant against an estimate of 10
    state.weeds_estimate = 1
    assert not state.maybe_unmarked()
    state.weeds_estimate = 10
    state.add_patch((66, 50, 90, 90), "horsetail")
    assert not state.maybe_unmarked()  # a weed area covers what boxes can't count
    state.all_weeds_marked = False
    assert state.maybe_unmarked()


def test_overlay_outlines_weed_areas_in_blue():
    state = StopState(field_photo(), "soybean")
    state.add_patch((50, 50, 90, 90), "horsetail")
    assert state.overlay().getpixel((50, 70)) == AREA_RGB
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `.venv/Scripts/python -m pytest tests/test_state.py -v`
Expected: the 5 new tests FAIL (`AttributeError: 'StopState' object has no attribute 'add_patch'`).

- [ ] **Step 4: Implement**

In `scout/agent/state.py`:

Add to the imports:

```python
from scout.vision.patches import green_mask, patch_mask
```

After `MAX_BOXES = 16  # SAM takes at most 16 objects per request` add:

```python
MAX_MARK_BOXES = 120  # one mark call; code splits it into SAM requests of MAX_BOXES
PATCH_BOX_MOSTLY_CROP = 0.5  # a weed-area box this much on crop is a box around the row: refused
UNMARKED_SHARE = 0.8  # marking fewer plants than this share of the agent's own estimate suggests weeds were missed
```

In `StopState.__init__`, after `self.box_cap_hit = False ...` add:

```python
        self.patch_boxes: dict[str, Box] = {}  # weed areas the agent drew; their masks follow the crop
        self.crop_seen: str | None = None  # from survey
        self.weeds_estimate: int | None = None  # from survey: weed plants the agent sees
        self.all_weeds_marked: bool | None = None  # from finish; None when the stop never finished
        self._green: np.ndarray | None = None
```

Add these methods after `is_patch`:

```python
    @property
    def green(self) -> np.ndarray:
        if self._green is None:
            self._green = green_mask(self.image.work)
        return self._green

    def _crop_for_areas(self) -> np.ndarray:
        """Crop that weed areas leave out. A weed area later classified as crop is not crop for the others."""
        return self.union_of(lambda i: i.role == "crop" and i.source != "patch")

    def add_patch(self, box: Box, label: str) -> tuple[Instance | None, str | None]:
        """Register a weed area measured from green pixels, or say why not."""
        left, top, right, bottom = box
        crop = self._crop_for_areas()
        if crop[top:bottom, left:right].mean() > PATCH_BOX_MOSTLY_CROP:
            return None, "mostly crop: box the weeds between the rows, not the row"
        mask, pieces = patch_mask(self.green, box, crop)
        if not mask.any():
            return None, "no green weed pixels in this area"
        self._next += 1
        inst = Instance(id=f"s{self.image.index}-i{self._next}", source="patch", prompt=label, box=box,
                        area_px=int(mask.sum()), role="weed", plant_count=max(pieces, 1))
        self.instances[inst.id], self.masks[inst.id], self.patch_boxes[inst.id] = inst, mask, box
        return inst, None

    def refresh_patches(self) -> None:
        """Recompute every weed area against the current crop, so a crop leaf marked later leaves the area."""
        crop = self._crop_for_areas()
        for iid, box in self.patch_boxes.items():
            mask, pieces = patch_mask(self.green, box, crop)
            self.masks[iid] = mask
            self.instances[iid].area_px, self.instances[iid].plant_count = int(mask.sum()), max(pieces, 1)

    def has_weed_area(self) -> bool:
        return any(i.role == "weed" and i.source == "patch" for i in self.instances.values())

    def weeds_marked(self) -> tuple[int, int]:
        """(weeds boxed one by one, weed areas)."""
        weeds = [i for i in self.instances.values() if i.role == "weed"]
        areas = sum(1 for i in weeds if i.source == "patch")
        return len(weeds) - areas, areas

    def plants_marked(self) -> int:
        return sum(i.plant_count for i in self.instances.values() if i.role == "weed")

    def maybe_unmarked(self) -> bool:
        """The agent didn't confirm every weed is marked, or, without weed areas, marked well under its estimate."""
        if self.all_weeds_marked is not True:
            return True
        if self.weeds_estimate and not self.has_weed_area():
            return self.plants_marked() < UNMARKED_SHARE * self.weeds_estimate
        return False
```

Replace `count_approximate` with:

```python
    def count_approximate(self, view_type: str | None = None) -> bool:
        """A patch counts as one weed (but a big plant in a close-up is just a plant), a weed area's pieces are not
        exactly its plants, and a capped mark may have left weeds unmarked."""
        return (self.has_patch() and view_type != "close_up") or self.has_weed_area() or self.box_cap_hit
```

Replace `overlay` with:

```python
    def overlay(self) -> Image.Image:
        weed = self.union_of(lambda i: i.role == "weed")
        crop = self.union_of(lambda i: i.role == "crop") & ~weed  # weeds win: a weed in the row keeps its pixels
        flagged = [i.box for i in self.instances.values() if i.role == "weed" and i.check_this]
        areas = [self.patch_boxes[i.id] for i in self.instances.values() if i.role == "weed" and i.id in self.patch_boxes]
        return render_overlay(self.image.work, crop, weed, self.row_band(), flagged, areas)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_state.py tests/test_tools.py tests/test_stop_agent.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scout/agent/state.py tests/helpers.py tests/test_state.py
git commit -m "feat: stop state keeps weed areas, the survey and an unmarked-weeds check" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Tools: survey, mark without the cap, weed areas, feedback

**Files:**
- Modify: `scout/agent/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Update the test helpers and old tests**

In `tests/test_tools.py`:

Change the imports to:

```python
from scout.agent.state import StopFailed, StopState
from scout.agent.tools import OVERLAY_CAPTION, TOOL_SCHEMAS, ToolRunner
from scout.vision.sam_parse import SamRecord
from tests.helpers import FakeSam, field_photo, field_scene, make_image, mask_record, rect, small_weedbook
```

Replace `_runner` with:

```python
def _runner(sam=None, max_calls=12, image=None):
    state = StopState(image or make_image(100, 100), "soybean")
    return ToolRunner(state, sam or FakeSam(), small_weedbook(), max_calls=max_calls), state
```

Replace `test_schemas_name_six_tools` with:

```python
def test_schemas_name_seven_tools():
    assert [t["name"] for t in TOOL_SCHEMAS] == ["survey", "segment", "mark", "zoom", "classify", "measure", "finish"]
    schemas = {t["name"]: t["parameters"] for t in TOOL_SCHEMAS}
    assert schemas["mark"]["properties"]["boxes"]["maxItems"] == 120
    assert schemas["mark"]["properties"]["kind"]["enum"] == ["crop", "weed", "patch"]
    assert "all_weeds_marked" in schemas["finish"]["required"]
    assert schemas["survey"]["required"] == ["crop_seen", "weeds_visible"]
```

In `test_mark_rejects_bad_boxes_before_calling_sam`, change `one * 17` to `one * 121`.

Replace `test_sixteen_weed_boxes_make_the_count_approximate` with:

```python
def test_a_full_mark_call_makes_the_count_a_floor():
    runner, state = _runner()
    _mark(runner, "weed", WEED_A, *[BARE] * 119)
    assert state.box_cap_hit and len(state.unclassified()) == 1
    metrics = json.loads(runner.run("measure", {"view_type": "close_up"}).output)
    assert metrics["weed_count"] == 1 and metrics["count_approximate"] is True
```

- [ ] **Step 2: Write the new failing tests**

Append to `tests/test_tools.py`:

```python
def test_survey_stores_the_crop_seen_and_the_estimate():
    runner, state = _runner()
    assert json.loads(runner.run("survey", {"crop_seen": "corn", "weeds_visible": 52}).output) == {"ok": True}
    assert (state.crop_seen, state.weeds_estimate) == ("corn", 52)
    assert state.steps[-1].summary == "Survey: looks like corn, about 52 weeds"
    runner.run("survey", {"crop_seen": "unsure", "weeds_visible": "3"})
    assert (state.crop_seen, state.weeds_estimate) == ("unsure", 3)
    assert state.steps[-1].summary == "Survey: crop unclear, about 3 weeds"
    for bad in ({"crop_seen": "wheat", "weeds_visible": 1}, {"crop_seen": "corn", "weeds_visible": -1},
                {"crop_seen": "corn", "weeds_visible": None}):
        assert runner.run("survey", bad).output.startswith("error:")
    assert (state.crop_seen, state.weeds_estimate) == ("unsure", 3)


def test_mark_splits_many_boxes_into_requests_of_sixteen():
    sam = FakeSam()
    runner, state = _runner(sam)
    out = _mark(runner, "weed", *[BARE] * 20, WEED_B, *[BARE] * 15)
    assert sorted(len(boxes) for kind, boxes in sam.calls if kind == "boxes") == [4, 16, 16]
    assert [(f["box_no"], f["box"]) for f in out["found"]] == [(21, WEED_B)]
    assert out["notes"][0] == "box 1: nothing found" and out["notes"][-1] == "box 36: nothing found"
    assert not state.box_cap_hit and state.steps[-1].summary == "Mark weed: 1 of 36 boxes kept"


def test_mark_answers_with_counts_a_hint_and_a_picture():
    runner, state = _runner()
    runner.run("survey", {"crop_seen": "soybean", "weeds_visible": 2})
    result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "waterhemp"}]})
    out = json.loads(result.output)
    assert out["weeds_marked"] == {"plants": 1, "patches": 0} and out["weeds_estimate"] == 2
    assert out["hint"] == ("You estimated 2 weeds and have marked 1 plant and 0 patches. Mark the rest: more boxes, "
                           "or kind patch where they are too dense.")
    assert [caption for caption, _ in result.images] == [OVERLAY_CAPTION]
    assert result.images[0][1].startswith("data:image/jpeg;base64,")
    out = _mark(runner, "weed", WEED_B)
    assert out["weeds_marked"] == {"plants": 2, "patches": 0} and "hint" not in out


def test_patch_kind_measures_green_pixels_and_follows_the_crop():
    sam = FakeSam(scene={"row": rect(10, 0, 30, 100), "leaf": rect(70, 60, 75, 80)})
    runner, state = _runner(sam, image=field_photo())
    _mark(runner, "crop", ROW)
    out = _mark(runner, "patch", [500, 500, 900, 900], [50, 100, 350, 900], [350, 100, 550, 400])
    assert [(f["box_no"], f["plants"], f["area_pct"]) for f in out["found"]] == [(1, 2, 2.0)]
    assert out["notes"] == ["box 2: mostly crop: box the weeds between the rows, not the row",
                            "box 3: no green weed pixels in this area"]
    assert out["weeds_marked"] == {"plants": 0, "patches": 1}
    assert state.steps[-1].summary == "Mark patch: 1 of 3 areas kept"
    assert ("boxes", ((50, 50, 90, 90),)) not in sam.calls  # weed areas never go to SAM
    _mark(runner, "crop", [700, 600, 750, 800])  # a crop leaf over the second shoot
    area = state.instances[out["found"][0]["id"]]
    assert area.area_px == 100 and area.plant_count == 1


def test_finish_records_whether_every_weed_is_marked():
    runner, state = _runner()
    runner.run("measure", {"view_type": "close_up"})
    assert runner.run("finish", {"crop_stage": None, "notes": "", "all_weeds_marked": False}).finished
    assert state.all_weeds_marked is False
    runner.run("finish", {"crop_stage": None, "notes": "", "all_weeds_marked": "yes"})
    assert state.all_weeds_marked is None  # only a real true or false counts
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: FAIL (`ImportError: cannot import name 'OVERLAY_CAPTION'`).

- [ ] **Step 4: Implement**

In `scout/agent/tools.py`:

Replace the module docstring and imports with:

```python
"""The seven tools Muse Spark can call at a stop, and the code that runs them. Every box in or out is 0-1000."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from scout.agent.state import MAX_BOXES, MAX_MARK_BOXES, UNMARKED_SHARE, StopFailed, StopState, to_pixels, to_thousandths
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import to_data_url
from scout.models import Instance
from scout.retry import BadRequest
from scout.vision import masks as mk
from scout.vision.measure import compute_metrics
from scout.vision.sam_parse import SamRecord
```

Replace the constants block (from `VIEW_TYPES` to `PATCH_NOTE`) with:

```python
VIEW_TYPES = ("row", "top_down", "close_up", "wide")
KINDS = ("crop", "weed")
MARK_KINDS = ("crop", "weed", "patch")
CROPS_SEEN = ("corn", "soybean", "alfalfa", "other", "unsure")
CONFIDENCES = ("high", "medium", "low")
ZOOM_MAX_PX = 768
MAX_LISTED = 100
SAM_WORKERS = 4  # SAM requests in flight per mark call
NOTHING = "nothing found"
MOSTLY_CROP_NOTE = "mostly crop"
PATCH_NOTE = "large patch; counted as one, approximate"
OVERLAY_CAPTION = ("What is marked now: green is crop, magenta is weed, blue boxes are weed areas, "
                   "yellow boxes need a check")
```

In `TOOL_SCHEMAS`, insert as the first entry:

```python
    _fn("survey", "Look over the photo first: the crop you see, and how many weed plants are visible.",
        {"crop_seen": {"type": "string", "enum": list(CROPS_SEEN)},
         "weeds_visible": {"type": "integer", "minimum": 0,
                           "description": "Your estimate of separate weed plants visible, small ones included"}},
        ["crop_seen", "weeds_visible"]),
```

Replace the `mark` entry with:

```python
    _fn("mark", "Mark what you see with boxes. crop and weed boxes go to SAM 3.1, any number (code splits them into "
        "requests of 16). patch boxes are areas of weeds too dense or small to box one by one; code measures them as "
        "green pixels that are not crop. The answer comes with a picture of everything marked so far.",
        {"kind": {"type": "string", "enum": list(MARK_KINDS),
                  "description": "crop: boxes around crop rows or crop plants (one tall box per visible row is "
                                 "fine). weed: a tight box around each weed, one per plant. patch: a box around an "
                                 "area of dense or tiny weeds, with as little crop as possible."},
         "boxes": {"type": "array", "minItems": 1, "maxItems": MAX_MARK_BOXES, "items": {
             "type": "object",
             "properties": {"box": _BOX, "label": {"type": "string", "description": "What it is, in 2-3 words"}},
             "required": ["box", "label"], "additionalProperties": False}}},
        ["kind", "boxes"]),
```

Replace the `finish` entry with:

```python
    _fn("finish", "Close this stop after measure.",
        {"crop_stage": {"type": ["string", "null"], "description": "Crop growth stage such as V3, or null"},
         "notes": {"type": "string"},
         "all_weeds_marked": {"type": "boolean",
                              "description": "true only if every weed you can see is marked, as boxes or areas"}},
        ["crop_stage", "notes", "all_weeds_marked"]),
```

Add these methods to `ToolRunner` (after `_add`):

```python
    def _sam_boxes(self, pixel_boxes: list) -> dict[str, SamRecord]:
        """Every box to SAM in requests of at most 16, sent in parallel. Keys are box numbers from 1."""
        starts = list(range(0, len(pixel_boxes), MAX_BOXES))
        chunks = [pixel_boxes[start:start + MAX_BOXES] for start in starts]
        with ThreadPoolExecutor(max_workers=min(SAM_WORKERS, len(chunks))) as pool:
            answers = list(pool.map(lambda chunk: self.sam.segment_boxes(self.state.image, chunk), chunks))
        records: dict[str, SamRecord] = {}
        unexpected = []
        for start, chunk, answer in zip(starts, chunks, answers):
            for record in answer:  # object i + 1 is the chunk's box i
                n = int(record.object_id) if str(record.object_id).isdigit() else 0
                if 1 <= n <= len(chunk):
                    records.setdefault(str(start + n), record)
                else:
                    unexpected.append(record.object_id)
        if unexpected:
            self.state.log("mark", f"SAM returned unexpected object ids {unexpected[:5]}", ok=False)
        return records

    def _feedback(self, out: dict) -> ToolResult:
        """Counts against the survey estimate, a hint when marking is short of it, and a picture of the marks."""
        plants, areas = self.state.weeds_marked()
        out["weeds_marked"] = {"plants": plants, "patches": areas}
        estimate = self.state.weeds_estimate
        if estimate is not None:
            out["weeds_estimate"] = estimate
            if self.state.plants_marked() < UNMARKED_SHARE * estimate:
                out["hint"] = (f"You estimated {estimate} weeds and have marked {plants} plant{'' if plants == 1 else 's'}"
                               f" and {areas} patch{'' if areas == 1 else 'es'}. Mark the rest: more boxes, or kind "
                               "patch where they are too dense.")
        try:
            images = [(OVERLAY_CAPTION, to_data_url(self.state.overlay()))]
        except Exception:  # the picture helps the agent check its work; the counts above still stand without it
            images = []
        return ToolResult(json.dumps(out), images=images)

    def _mark_patches(self, pixel_boxes: list, labels: list[str]) -> ToolResult:
        found, notes = [], []
        for n, (box, label) in enumerate(zip(pixel_boxes, labels), start=1):
            inst, note = self.state.add_patch(box, label)
            if inst is not None:
                found.append({"box_no": n, **self._brief(inst), "plants": inst.plant_count})
            if note:
                notes.append(f"box {n}: {note}")
        self.state.log("mark", f"Mark patch: {len(found)} of {len(pixel_boxes)} areas kept", [f["id"] for f in found])
        return self._feedback({"found": found, "notes": notes})

    def _tool_survey(self, crop_seen: str, weeds_visible) -> ToolResult:
        if crop_seen not in CROPS_SEEN:
            raise ValueError(f"crop_seen must be one of {list(CROPS_SEEN)}")
        count = _number(weeds_visible, "weeds_visible")
        if count is None:
            raise ValueError("weeds_visible must be a number")
        self.state.crop_seen, self.state.weeds_estimate = crop_seen, int(count)
        look = "crop unclear" if crop_seen == "unsure" else f"looks like {crop_seen}"
        self.state.log("survey", f"Survey: {look}, about {int(count)} weeds")
        return ToolResult(json.dumps({"ok": True}))
```

Replace `_tool_mark` with:

```python
    def _tool_mark(self, kind: str, boxes: list[dict]) -> ToolResult:
        if kind not in MARK_KINDS:
            raise ValueError(f"kind must be one of {list(MARK_KINDS)}")
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= MAX_MARK_BOXES:
            raise ValueError(f"give 1 to {MAX_MARK_BOXES} boxes")
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
        self.state.log("mark", f"Mark {kind}: {len(found)} of {len(boxes)} boxes kept", [f["id"] for f in found])
        return self._feedback({"found": found, "notes": notes})
```

In `_tool_segment`, just before `self.state.log("segment", summary, ...)`, add:

```python
        if kind == "crop":
            self.state.refresh_patches()
```

In `_tool_classify`, just before `self.state.log("classify", summary, ids)`, add:

```python
        self.state.refresh_patches()  # an instance reclassified as crop changes what weed areas leave out
```

Replace `_tool_finish` with:

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_tools.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scout/agent/tools.py tests/test_tools.py
git commit -m "feat: survey tool, mark without the 16-box cap, weed areas and marking feedback" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Stop agent: survey first, one picture per turn, new flags

**Files:**
- Modify: `scout/agent/stop_agent.py`
- Test: `tests/test_stop_agent.py`

- [ ] **Step 1: Update the old tests**

In `tests/test_stop_agent.py`:

Change the imports to:

```python
from scout.agent.state import MAX_MARK_BOXES
from scout.agent.stop_agent import UNMARKED_FLAG, crop_flag, run_stop
from scout.agent.tools import OVERLAY_CAPTION
```

(keep the other existing imports).

Replace `_close` with:

```python
def _close(view_type="row", all_marked=True):
    return [ToolCall("c-m", "measure", {"view_type": view_type}),
            ToolCall("c-f", "finish", {"crop_stage": None, "notes": "done", "all_weeds_marked": all_marked})]
```

In `test_full_stop`, change the finish call to

```python
                       ToolCall("c6", "finish", {"crop_stage": "V3", "notes": "Two weeds between rows.",
                                                 "all_weeds_marked": True})], ""),
```

and change `assert [t["name"] for t in first["tools"]][:2] == ["segment", "mark"]` to
`assert [t["name"] for t in first["tools"]][:3] == ["survey", "segment", "mark"]`.

In `test_call_limit_closes_stop_and_flags_unclassified_weeds`, the stop never finishes, so the unmarked flag joins:

```python
    assert result.flags == ["Check s0-i3: marked but never classified", "Check s0-i4: marked but never classified",
                            UNMARKED_FLAG]
    assert result.count_approximate is True
```

Marks now answer with a picture, so a turn of marks is followed by one picture message in the next request. In `test_crop_marks_run_first_within_a_turn`, replace the `call_id` assertion with:

```python
    outputs = [p["call_id"] for p in llm.requests[1]["input"] if p.get("type") == "function_call_output"]
    assert outputs == ["c2", "c1"]  # outputs keep the model's order
```

In `test_a_lost_conversation_is_resent_in_full_and_the_stop_goes_on_stateless`, replace the assertions from `assert _kinds(retry["input"]) ...` through `assert [i["call_id"] for i in last["input"][5:7]] ...` with:

```python
    assert _kinds(retry["input"]) == ["message", "function_call", "function_call",
                                      "function_call_output", "function_call_output", "message"]
    assert [i["call_id"] for i in retry["input"][1:3]] == [i["call_id"] for i in retry["input"][3:5]] == ["c1", "c2"]
    assert retry["input"][1]["name"] == "mark" and json.loads(retry["input"][1]["arguments"])["kind"] == "crop"
    assert retry["input"][3:] == lost["input"]  # exactly the items the lost request carried: outputs and the picture
    assert last["previous_response_id"] is None  # the rest of the stop stays stateless
    assert last["input"][:6] == retry["input"]  # the history only grows
    assert _kinds(last["input"][6:]) == ["function_call", "function_call_output", "message"]
    assert [i["call_id"] for i in last["input"][6:8]] == ["c3", "c3"]  # the zoom call, its output, then the close-up
```

- [ ] **Step 2: Write the new failing tests**

Append to `tests/test_stop_agent.py`:

```python
SURVEY = ToolCall("c0", "survey", {"crop_seen": "corn", "weeds_visible": 2})


def test_survey_runs_first_and_a_mismatched_crop_is_flagged():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS, SURVEY], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm)
    assert result.steps[0].tool == "survey" and result.crop_seen == "corn"
    assert result.flags == ["This photo looks like corn, not soybean"]
    weeds = next(p for p in llm.requests[1]["input"] if p.get("call_id") == "c2")
    assert json.loads(weeds["output"])["weeds_estimate"] == 2  # the survey ran before the weed mark


def test_only_the_latest_marking_picture_rides_along():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    _run(llm)
    messages = [p for p in llm.requests[1]["input"] if p.get("type") == "message"]
    assert len(messages) == 1
    images = [c for c in messages[0]["content"] if c["type"] == "input_image"]
    assert len(images) == 1 and OVERLAY_CAPTION in messages[0]["content"][0]["text"]


def test_marking_well_under_the_estimate_is_flagged():
    survey = ToolCall("c0", "survey", {"crop_seen": "soybean", "weeds_visible": 10})
    llm = FakeLlm([LlmTurn("r1", [survey, MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm)
    assert result.flags == [UNMARKED_FLAG] and result.count_approximate is True


def test_the_agent_saying_weeds_remain_is_flagged():
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close(all_marked=False)], "")])
    result, _ = _run(llm)
    assert result.flags == [UNMARKED_FLAG]


def test_crop_flag_wording():
    assert crop_flag("corn", "soybean") == "This photo looks like corn, not soybean"
    assert crop_flag("other", "corn") == "This photo doesn't look like corn"
    assert crop_flag("unsure", "corn") is None and crop_flag(None, "corn") is None
    assert crop_flag("corn", "corn") is None


def test_the_floor_flag_names_the_mark_limit():
    many = ToolCall("c2", "mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "w"}] +
                                   [{"box": [400, 500, 500, 600], "label": "w"}] * (MAX_MARK_BOXES - 1)})
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, many], ""), LlmTurn("r2", [_classify("s0-i3"), *_close("close_up")], "")])
    result, _ = _run(llm)
    assert f"Weed count is a floor: a mark call used all {MAX_MARK_BOXES} boxes" in result.flags
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `.venv/Scripts/python -m pytest tests/test_stop_agent.py -v`
Expected: FAIL (`ImportError: cannot import name 'UNMARKED_FLAG'`).

- [ ] **Step 4: Implement**

In `scout/agent/stop_agent.py`:

Change the imports:

```python
from scout.agent.state import MAX_MARK_BOXES, StopState
from scout.agent.tools import OVERLAY_CAPTION, TOOL_SCHEMAS, ToolRunner
```

Below `FAILED_FLAG = ...` add:

```python
UNMARKED_FLAG = "Some weeds may be unmarked: cover and count are minimums"
```

Replace `run_order` with:

```python
def run_order(calls: list[ToolCall]) -> list[ToolCall]:
    """The survey runs first, so marks can be checked against its estimate; crop marks and segments next, so the
    weed check sees the crop; then the other tools; measure and finish last, so they see the turn's verdicts. The
    order is stable within each rank."""
    def rank(call: ToolCall) -> int:
        if call.name == "survey":
            return -1
        if call.name in ("mark", "segment") and call.arguments.get("kind") == "crop":
            return 0
        return {"measure": 2, "finish": 3}.get(call.name, 1)
    return sorted(calls, key=rank)


def latest_overlay_only(images: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Each mark answers with a picture of everything marked so far; within one turn only the last one is news."""
    last = max((n for n, (caption, _) in enumerate(images) if caption == OVERLAY_CAPTION), default=None)
    return [image for n, image in enumerate(images) if image[0] != OVERLAY_CAPTION or n == last]


def crop_flag(crop_seen: str | None, crop: str) -> str | None:
    if crop_seen in (None, "unsure", crop):
        return None
    if crop_seen == "other":
        return f"This photo doesn't look like {crop}"
    return f"This photo looks like {crop_seen}, not {crop}"
```

In `run_stop`, replace

```python
                if images:
                    pending.append(user_message("Close-ups you asked for: " + "; ".join(c for c, _ in images),
                                                [url for _, url in images]))
```

with

```python
                images = latest_overlay_only(images)
                if images:
                    pending.append(user_message("Pictures for your tool calls: " + "; ".join(c for c, _ in images),
                                                [url for _, url in images]))
```

In `finalize`, replace the block from `flags = [_flag(i) ...]` through the `StopResult(...)` call with:

```python
    flags = [_flag(i) for i in instances if i.role == "weed" and i.check_this]
    mismatch = crop_flag(state.crop_seen, state.crop)
    if mismatch:
        flags.insert(0, mismatch)
    if m.count_approximate or (state.has_patch() and view != "close_up") or state.has_weed_area():
        flags.append("Weed count is approximate: some weeds merged into patches")
    if state.box_cap_hit:
        flags.append(f"Weed count is a floor: a mark call used all {MAX_MARK_BOXES} boxes")
    unmarked = state.maybe_unmarked()
    if unmarked:
        flags.append(UNMARKED_FLAG)
    has_crop = any(i.role == "crop" for i in instances)
    # Alfalfa is solid-seeded and its crop boxes are optional, so a worked alfalfa stop counts without them.
    worked = state.view_type is not None or any(i.role == "weed" for i in instances)
    return StopResult(
        stop_index=state.image.index, image_hash=state.image.hash, view_type=view, gps=state.image.gps,
        crop_stage=state.crop_stage, weed_cover_pct=m.weed_cover_pct, crop_cover_pct=m.crop_cover_pct,
        weed_count=m.weed_count,
        count_approximate=m.count_approximate or state.count_approximate(view) or unmarked,
        density_per_m2=m.density_per_m2, in_row_weed_share=m.in_row_weed_share, species=m.species,
        species_cover_pct=m.species_cover_pct, species_density=m.species_density,
        species_max_height_in=m.species_max_height_in, max_height_in=m.max_height_in, flags=flags,
        notes=state.notes, crop_seen=state.crop_seen, steps=state.steps,
        status="ok" if has_crop or (state.crop == "alfalfa" and worked) else "no_crop_found",
    )
```

(Keep the lines above `flags = ...` in `finalize` unchanged: the failed check, unclassified weeds flagged, `view`, `ensure_sky`, `compute_metrics`, `instances`.)

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_stop_agent.py tests/test_pipeline.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scout/agent/stop_agent.py tests/test_stop_agent.py
git commit -m "feat: survey first, one marking picture per turn, unmarked and crop-mismatch flags" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Instructions ask for every weed

**Files:**
- Modify: `scout/agent/prompts.py:4-37`
- Test: `tests/test_stop_agent.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stop_agent.py`:

```python
def test_instructions_ask_for_every_weed_not_a_sample():
    text = stop_instructions("corn", ["field_horsetail"], 30)
    assert "survey" in text and 'kind "patch"' in text and "all_weeds_marked" in text
    assert "not a sample" in text and "up to 6" in text and "30 tool calls" in text and "picture" in text
    assert "at most 16" not in text and "6 turns" not in text and "largest first" not in text
    assert "at most 2" not in text
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_stop_agent.py::test_instructions_ask_for_every_weed_not_a_sample -v`
Expected: FAIL (`assert 'survey' in text`).

- [ ] **Step 3: Rewrite the instructions**

Replace `stop_instructions` in `scout/agent/prompts.py` with:

```python
def stop_instructions(crop: str, species_ids: list[str], max_calls: int) -> str:
    if crop == "alfalfa":
        crop_boxes = "Alfalfa is solid-seeded, so crop boxes are optional."
    else:
        crop_boxes = (f"For the crop, box each visible {crop} row (one tall box per row is fine) or each {crop} "
                      'plant. In a view along the rows, segment "crop row" with kind "crop" also works.')
    return f"""You are an agronomist scouting a {crop} field. You see one photo. Mark every weed you can see, identify \
it, then have code measure it. A missed weed makes the farmer's numbers wrong, so be thorough.

Coordinates: every box you give or get is [x1, y1, x2, y2] from 0 to 1000 on both axes. x runs from the left edge \
(0) to the right edge (1000), and y from the top (0) to the bottom (1000).

Procedure:
1. In your first turn, call survey with the crop you see and your estimate of how many separate weed plants are \
visible, small ones included. In the same turn call mark with kind "crop" and mark with kind "weed". {crop_boxes} \
Crop boxes hug each row's plants; don't stretch them over the soil between rows, but do include crop leaves that \
cross it. For the weeds, give a tight box around every weed plant you can tell apart: any number of boxes, all of \
them, not a sample. Where weeds grow too densely or are too small to box one by one (a carpet of shoots, a mat of \
seedlings), call mark with kind "patch": one box around each such area, holding weeds and soil and as little crop \
as possible; code measures it as green pixels that are not crop. If you see no weeds, skip the weed marks: never \
box crop, soil or residue as a weed. A distinctive flowering weed can also be found by phrase: segment \
"dandelion" or "yellow flower" with kind "weed". Crop leaves close to the camera (the bottom of a row photo) are \
crop: box a weed where you see the weed plant, distinct from the crop.
2. Every mark answer comes with a picture of what is marked now: green is crop, magenta is weed, blue boxes are \
weed areas, yellow boxes need a check. Compare it with the photo. Mark weeds that are still missing, as boxes or \
areas. Where a crop leaf shows magenta inside a weed area, mark that leaf with kind "crop". "mostly crop" means \
that box was on crop and was dropped; "nothing found" means SAM found no plant there. Zoom on doubtful instances \
(check_this is true), up to 6; each close-up arrives at your next turn.
3. classify every weed instance, weed areas included, in one call. Use a species_id from this list only: \
{", ".join(species_ids)}. Use null when you cannot tell. Give leaf_count when you can count the leaves. height_in \
is an estimate from visible size cues (a {crop} plant, a leaf, the row spacing): use null when there is no cue. Use \
role "crop" for an instance that is really crop and "ignore" for soil, residue, or shadow.
4. measure with the view_type that fits: row (looking along rows), top_down (camera pointing straight down), \
close_up (a few plants fill the frame), wide (field to the horizon). Then finish with the crop growth stage if \
you can tell (for example V3), one sentence of notes, and all_weeds_marked: true only if every weed you can see is \
marked.

Rules: never invent measurements; cover, counts and density come only from measure. You have at most {max_calls} \
tool calls; use the turns you need to mark every weed. Always respond with tool calls, not prose. Every coordinate \
is 0 to 1000."""
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_stop_agent.py -q`
Expected: all pass, including the existing `test_instructions_use_thousandths_and_crop_hints` (it still finds "0 to 1000", the species list, "12 tool calls", '"crop row"', "crop boxes are optional", "no weeds", '"yellow flower"', "estimate", "null", "close to the camera" and "hug each row's plants").

- [ ] **Step 5: Commit**

```bash
git add scout/agent/prompts.py tests/test_stop_agent.py
git commit -m "feat: instructions ask for every weed, weed areas and a check of the marking picture" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Report warns about a crop mismatch

**Files:**
- Modify: `scout/agent/field_agent.py`
- Modify: `scout/report/templates/report.html`
- Test: `tests/test_field_agent.py`, `tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_field_agent.py`:

```python
def test_report_warns_when_photos_show_another_crop():
    stops = [_stops()[0].model_copy(update={"crop_seen": "corn"}),
             _stops()[1].model_copy(update={"crop_seen": "unsure"})]
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found. Read the label before use.")])
    report = build_report("w1", "soybean", stops, BOOK, [RECORD], llm)
    assert report.crop_mismatch == ("1 of 2 photos looks like corn, but this walk was scouted as soybean. "
                                    "Thresholds and label options below are for soybean. Re-run with the right crop.")
    assert report.decision == "treat"  # the farmer's pick still decides


def test_no_crop_warning_when_the_crop_matches_or_is_unclear():
    stops = [_stops()[0].model_copy(update={"crop_seen": "soybean"}), _stops()[1]]
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found. Read the label before use.")])
    assert build_report("w1", "soybean", stops, BOOK, [RECORD], llm).crop_mismatch is None


def test_crop_warning_wording_for_several_photos_and_other_crops():
    two = [StopResult(stop_index=i, image_hash=str(i), crop_seen="corn") for i in range(2)]
    assert crop_mismatch("soybean", two).startswith("2 of 2 photos look like corn")
    other = [StopResult(stop_index=0, image_hash="a", crop_seen="other")]
    assert crop_mismatch("corn", other).startswith("1 of 1 photos doesn't look like corn. Thresholds")
```

and add `crop_mismatch` to its import: `from scout.agent.field_agent import build_report, crop_mismatch, numbers_ok`.

Append to `tests/test_render.py`:

```python
def test_crop_mismatch_banner_comes_before_the_decision():
    warning = "1 of 1 photos looks like corn, but this walk was scouted as soybean."
    html = _html(_report().model_copy(update={"crop_mismatch": warning}))
    assert warning in html and html.index(warning) < html.index("Decision")
    assert "looks like corn" not in _html(_report())


def test_legend_explains_blue_weed_areas():
    assert "Blue box: weed area measured by green pixels" in _html(_report())
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `.venv/Scripts/python -m pytest tests/test_field_agent.py tests/test_render.py -v`
Expected: FAIL (`ImportError: cannot import name 'crop_mismatch'`).

- [ ] **Step 3: Implement**

In `scout/agent/field_agent.py`, add after `template_explanation`:

```python
def crop_mismatch(crop: str, stops: list[StopResult]) -> str | None:
    """A warning when photos show another crop than the walk was scouted as. The farmer's pick still decides."""
    ok = [s for s in stops if s.status == "ok"]
    seen = [s.crop_seen for s in ok if s.crop_seen not in (None, "unsure", crop)]
    if not seen:
        return None
    other, n = Counter(seen).most_common(1)[0][0], len(seen)
    if other == "other":
        head = f"{n} of {len(ok)} photos {'does' if n == 1 else 'do'}n't look like {crop}"
    else:
        head = (f"{n} of {len(ok)} photos {'looks' if n == 1 else 'look'} like {other}, but this walk was scouted as "
                f"{crop}")
    return f"{head}. Thresholds and label options below are for {crop}. Re-run with the right crop."
```

In `build_report`, change the `return FieldReport(...)` call to pass the warning:

```python
    return FieldReport(
        walk_id=walk_id, crop=crop, stops=stops, species_summary=d.summaries, decision=d.decision,
        treat_mode=d.treat_mode, urgency=urgency_text, rule_trace=d.rule_trace,
        explanation=explanation or template_explanation(d.decision, d.treat_mode, d.rule_trace), options=options,
        crop_mismatch=crop_mismatch(crop, stops),
    )
```

In `scout/report/templates/report.html`:

Add to the `<style>` block, after the `.flag` rule:

```css
  .warning { background: #fdecc8; border: 1px solid #e0a43a; border-radius: 10px; padding: 12px; font-weight: 600; }
```

After the `<h1>...</h1>` line add:

```html
{% if report.crop_mismatch %}<p class="warning">{{ report.crop_mismatch }}</p>{% endif %}
```

Replace the legend line with:

```html
<p>Green: crop · Magenta: weeds · Gray: in-row zone · Blue box: weed area measured by green pixels · Yellow box: check this plant</p>
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_field_agent.py tests/test_render.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/agent/field_agent.py scout/report/templates/report.html tests/test_field_agent.py tests/test_render.py
git commit -m "feat: report warns when photos show another crop than the walk" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The crop menu has no default

**Files:**
- Modify: `scout/web/index.html:14-18`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_main.py`:

```python
def test_crop_menu_makes_the_farmer_choose(tmp_path):
    html = _client(tmp_path).get("/").text
    assert '<option value="" disabled selected>Choose the crop</option>' in html
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `.venv/Scripts/python -m pytest tests/test_main.py::test_crop_menu_makes_the_farmer_choose -v`
Expected: FAIL.

- [ ] **Step 3: Change the menu**

In `scout/web/index.html` replace the `<select>` element with:

```html
      <select name="crop" required>
        <option value="" disabled selected>Choose the crop</option>
        <option value="soybean">Soybean</option>
        <option value="corn">Corn</option>
        <option value="alfalfa">Alfalfa</option>
      </select>
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_main.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scout/web/index.html tests/test_main.py
git commit -m "fix: the crop menu makes the farmer choose instead of defaulting to soybean" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Full suite

- [ ] **Step 1: Run every test**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (233 before this plan plus the new ones). Any failure is fixed before going on; do not skip or loosen a test to get green.

---

### Task 13: Live check on the real photos

**Files:**
- Modify: `scripts/run_stop.py`
- Modify: `docs/spike-notes.md`

- [ ] **Step 1: Let the run script work photos in parallel**

Replace `scripts/run_stop.py` with:

```python
"""Run the stop agent on photos, in parallel like a walk, and save overlays plus JSON results.

Usage: python scripts/run_stop.py --crop soybean --out run_out_v2 field-images/IMG_4219.JPG field-images/IMG_4220.JPG
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scout.agent.llm import Llm
from scout.agent.stop_agent import run_stop
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import load_image
from scout.config import get_settings
from scout.vision.sam_client import SamClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--crop", required=True, choices=["corn", "soybean", "alfalfa"])
    parser.add_argument("--out", default="run_out")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    settings = get_settings()
    llm, sam, weeds = Llm.from_settings(settings), SamClient.from_settings(settings), WeedBook.load()

    def one(photo: str) -> str:
        stem, started = Path(photo).stem, time.monotonic()
        try:
            image = load_image(0, Path(photo).read_bytes(), settings.full_max_px, settings.work_max_px)
            result, state = run_stop(image, args.crop, llm, sam, weeds, settings)
            state.overlay().save(out / f"{stem}.png")
        except Exception as error:  # an unreadable photo or a dropped connection must not end the batch
            return f"!! {stem} failed: {type(error).__name__}: {error}"
        (out / f"{stem}.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        return (f"{stem}: {result.status} in {time.monotonic() - started:.0f} s, cover {result.weed_cover_pct}%, "
                f"{result.weed_count} weeds, species {result.species}, crop seen {result.crop_seen}, "
                f"flags {result.flags}")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(one, args.photos):
            print(line, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all test photos at once, as one walk would**

Start these four in parallel (background), from the repo root:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_stop.py --crop corn --out run_out_v2 field-images/IMG_4214.JPG field-images/IMG_4215.JPG field-images/IMG_4547.JPG field-images/IMG_6564.JPG
```

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_stop.py --crop soybean --out run_out_v2 field-images/3AEE1239-C89C-4BD5-905B-C2FCBC01E161.JPG field-images/IMG_4219.JPG field-images/IMG_4220.JPG
```

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_stop.py --crop alfalfa --out run_out_v2 field-images/IMG_3922.JPG field-images/IMG_3926.JPG field-images/IMG_6942.JPG
```

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_stop.py --crop soybean --out run_out_mismatch field-images/IMG_4215.JPG
```

Expected: every photo prints a line; the slowest finishes within about 6 minutes of the start. `IMG_4215` in `run_out_mismatch` prints `crop seen corn` and its flags start with `This photo looks like corn, not soybean`.

- [ ] **Step 3: Compare before and after**

Run:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python - <<'EOF'
import json
from pathlib import Path
from PIL import Image, ImageDraw
rows, tiles = [], []
for new in sorted(Path("run_out_v2").glob("*.json")):
    old = Path("run_out") / new.name
    a = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    b = json.loads(new.read_text(encoding="utf-8"))
    secs = b["steps"][-1]["t_ms"] / 1000 if b["steps"] else 0
    marks = [s["summary"] for s in b["steps"] if s["tool"] in ("survey", "mark")]
    rows.append(f"| {new.stem} | {a.get('weed_cover_pct', '-')}% / {a.get('weed_count', '-')} | "
                f"{b['weed_cover_pct']}% / {b['weed_count']} | {secs:.0f} s | {b.get('crop_seen')} | "
                f"{'; '.join(marks)} | {'; '.join(b['flags'][:3])} |")
    pair = [Image.open(p).convert("RGB") for p in (Path("run_out") / f"{new.stem}.png", Path("run_out_v2") / f"{new.stem}.png") if p.exists()]
    if len(pair) == 2:
        h = 480
        pair = [im.resize((round(im.width * h / im.height), h)) for im in pair]
        tile = Image.new("RGB", (pair[0].width + pair[1].width + 10, h + 30), "white")
        tile.paste(pair[0], (0, 30)); tile.paste(pair[1], (pair[0].width + 10, 30))
        ImageDraw.Draw(tile).text((5, 8), f"{new.stem}: before | after", fill="black")
        tiles.append(tile)
print("| Photo | Before cover / count | After cover / count | Seconds | Crop seen | Survey and marks | Flags |")
print("|---|---|---|---|---|---|---|")
print("\n".join(rows))
if tiles:
    sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y)); y += t.height
    sheet.save("run_out_v2/compare_sheet.jpg", quality=85)
EOF
```

Expected: a Markdown table printed, and `run_out_v2/compare_sheet.jpg` written. On `IMG_4215` the after cover is well above 5.87% and the near inter-row horsetail is magenta on the sheet.

- [ ] **Step 4: Record the results**

Append to `docs/spike-notes.md` a section `## Complete marking (2026-11-02)` with:
- one paragraph on what changed (survey, no 16-box cap, weed areas, feedback picture, unmarked and crop-mismatch flags, budget 30 calls / 360 s / 10 stops at once);
- the table printed in Step 3;
- the slowest stop's seconds and whether a 10-photo walk would fit in about 6 minutes;
- the `run_out_mismatch` line for `IMG_4215` scouted as soybean;
- anything that looks wrong on the compare sheet, stated plainly.

- [ ] **Step 5: Commit**

`run_out*/` and `field-images/` stay out of git; commit only the script and the notes:

```bash
git add scripts/run_stop.py docs/spike-notes.md
git commit -m "docs: live check of complete marking on the test photos" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Show the user**

Send `run_out_v2/compare_sheet.jpg` to the user with the table, and ask them to judge the after overlays (the open questions on IMG_4220, IMG_4547 and 3AEE1239 still stand).
