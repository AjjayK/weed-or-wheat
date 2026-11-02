# Complete weed marking: design

Date: 2026-11-01. Status: approved in brainstorming, awaiting spec review.

## Problem

On a corn photo carpeted with field horsetail (the walk `store/1b2dd669140fbf64`), the agent drew 12 weed boxes
around the largest clumps and left most of the horsetail unmarked. Unmarked weeds are never outlined or measured,
so the reported 5.62% cover is far too low, and no flag said so. The same walk was scouted as soybean although the
photo is corn; the agent noticed ("Crop appears to be corn not soybean") but only in the stop notes.

The model can see the weeds. Our own instructions and tool schema stopped it:

1. `mark` caps boxes at 16 (`maxItems: MAX_BOXES`), SAM's per-request limit passed straight to the model.
2. The prompt says "one box per plant, nearest or largest first, at most 16": with 100+ shoots it boxes a sample.
3. "Box a weed only where you see the whole weed plant, distinct from the crop" skips thin or tangled shoots.
4. "Aim for 6 turns", "zoom on at most 2", "mark them once more" were written to save time and credit.
5. Nothing checks completeness: the reply "12 of 12 boxes kept" never asks whether that is all of them, and the only
   completeness flag needs all 16 boxes used.
6. There is no way to mark a carpet of weeds: only one box per plant.

Probe evidence (scratchpad scripts, 2026-11-01):

| Probe | Result |
|---|---|
| SAM text prompts on IMG_4215: "horsetail", "field horsetail", "equisetum", "seedling", "grass", "weeds" | 0 found each ("pine seedling" found 14 by lookalike) |
| Muse Spark told to estimate first and box every weed, up to 120 (IMG_4215) | estimate 52, 52 boxes in 22 s; SAM in 4 parallel chunks of 16: 1.8 s |
| Same on IMG_4220 (soybean, waterhemp) | 28 boxes (the test run had 8) |
| SAM inside those 52 boxes | fills most of each box, soil included: 14.2% cover, inflated |
| Whole-photo green pixels minus crop (no-till soybean 3AEE1239) | distant crop rows and trees turn magenta: 11.6% false cover |
| Green pixels inside an agent-drawn weed-area box, minus crop (IMG_4215) | traces the horsetail shoots; a crossing corn leaf missing from the crop mask was counted |
| Agent asked which crop it sees (IMG_4215, IMG_4220, 3AEE1239) | corn, soybean, soybean: all correct |

## Goal

Every weed the agent can see is marked, either one box per plant or as a weed-area patch; what can't be marked is
flagged; a crop mismatch is impossible to miss. Credits are not a constraint; a 10-photo walk should still finish in
about 3 to 6 minutes, never 20.

Out of scope: the UI redesign (live step pictures, reasoning text), which is a separate spec that follows this one.

## Design

### 1. How the agent marks a photo

**New tool `survey`**, called first:

| Field | Type | Meaning |
|---|---|---|
| `crop_seen` | enum `corn`, `soybean`, `alfalfa`, `other`, `unsure` | The crop the photo shows |
| `weeds_visible` | integer ≥ 0 | Estimate of separate weed plants visible, small ones included |

The handler stores both on `StopState` (`crop_seen`, `weeds_estimate`) and logs `Survey: looks like corn, about 52
weeds`. Calling it again overwrites the values.

**`mark` without the 16 cap.** `boxes` takes up to `MAX_MARK_BOXES = 120`. The handler splits the boxes into chunks of
at most `MAX_BOXES = 16` (SAM's limit, unchanged in `SamClient.segment_boxes`) and sends the chunks in parallel (a
thread pool of up to 8). Results are registered in box order, so instance ids stay stable. `box_cap_hit` is set when
a call uses all 120, and its flag reads "Weed count is a floor: a mark call used all 120 boxes".

**New mark kind `patch`**: a box around an area where weeds are too dense or too small to box one by one. Patches go
to no SAM request; they are measured from pixels (section 2).

**Feedback after every mark**, in the tool output JSON (existing fields kept):

- `weeds_marked`: weed instances from boxes plus patches, e.g. `{"plants": 12, "patches": 1}`.
- `weeds_estimate`: the survey value, when given.
- `hint`, only when plants marked + patch plant counts are under 80% of the estimate: "You estimated 52 weeds and
  have marked 12 plants and 1 patch. Mark the rest: more boxes, or kind patch where they are too dense."

The result also carries one image (like `zoom`): the current overlay at the work size, captioned "What is marked now:
green crop, magenta weeds, blue patch boxes, yellow boxes need a check." The agent compares it with the photo and
fixes slips: marks missed weeds, boxes a crop leaf that crosses a patch as crop (which removes it from the patch,
section 2), or classifies a wrong instance as `ignore`.

**Instructions rewritten** (`stop_instructions`):

- Procedure: (1) `survey`, then `mark` crop and `mark` weeds in the same turn: box every weed plant you can tell
  apart, any number; use kind `patch` for areas too dense or small to box; box crop rows tightly, including crop
  leaves that cross the inter-row. (2) Check the returned picture against the photo; mark what was missed; zoom on
  doubtful instances, up to 6. (3) `classify` every instance, patches included, in one call. (4) `measure`, then
  `finish`.
- Removed: "nearest or largest first, at most 16", "aim for 6 turns or fewer", "zoom on at most 2", "mark them once
  more", "finish in your second turn".
- Kept: never box crop, soil or residue as a weed; crop leaves near the camera are crop; species ids from the list
  only; never invent measurements; respond with tool calls only.

**`finish` gains `all_weeds_marked` (boolean, required)**, stored on `StopState`.

**Budget** (`Settings` defaults): `max_tool_calls` 12 → 30, `stop_timeout_s` 240 → 360, `max_concurrent_stops`
4 → 10. Expected 2 to 4 minutes per stop; with all photos in parallel, a 10-photo walk takes about 3 to 6 minutes.

### 2. How patches, counts and flags are measured

**Patch mask**, computed by code in `scout/vision/patches.py`:

```
green  = excess green on chromatic coordinates: 2g - r - b > GREEN_EXG (0.08), with r, g, b = R, G, B / (R + G + B)
patch  = green AND inside the patch box AND NOT dilate(crop union, 3 px)
patch  = patch without connected pieces under MIN_PIECE_PX (20 px)
```

Soil, residue and shadow are not green, so they never count.

**Recomputed when crop changes.** A patch keeps its box. `StopState.refresh_patches()` recomputes every patch mask
after any crop mark, so boxing a crossing leaf as crop removes it from the patch at once. `Instance.area_px` and
`box` are updated with the mask.

**Mostly-crop patches are refused.** If more than `PATCH_BOX_MOSTLY_CROP = 0.5` of the patch box is crop, no
instance is made and the agent is told: "patch N is mostly crop: box the weeds between the rows, not the row".

**Cover**: patch pixels join the weed union in `compute_metrics` like any weed mask: each pixel counts once, weeds
win over crop, sky is excluded.

**Counts**: `Instance` gains `plant_count: int = 1`. A patch's `plant_count` is the number of connected green pieces
in its mask, at least 1. `compute_metrics` sums `plant_count` instead of counting instances. A patch always makes the
stop's count approximate. The "second box on a plant already counted" rule (`MIN_OWN_FRAC`) applies to patches as to
any weed.

**Species**: patches are classified like other instances. A patch without a species becomes unidentified broadleaf
or grass from its label, as today.

**New flag**: "Some weeds may be unmarked: cover and count are minimums." It fires when `all_weeds_marked` is false,
or when there are no patches and the marked plant count is under 80% of `weeds_estimate`. It also sets
`count_approximate`.

**Unchanged**: the decision rules. They already grade cover when a count is approximate.

**Overlay**: `render_overlay` draws patch boxes in blue (0, 160, 255), as the agent's feedback image and the stop
overlay in the report both use it. The report legend adds "Blue box: weed area measured by green pixels".

### 3. Crop-mismatch warning

- When `crop_seen` is `corn`, `soybean`, `alfalfa` or `other` and differs from the walk's crop, the stop gets the
  flag "This photo looks like corn, not soybean" (`other`: "This photo doesn't look like soybean"). `unsure` raises
  nothing. `StopResult` gains `crop_seen: str | None`.
- `FieldReport` gains `crop_mismatch: str | None`, set by `build_report` when any ok stop mismatches: "1 of 1 photos
  looks like corn, but this walk was scouted as soybean. Thresholds and label options below are for soybean. Re-run
  with the right crop." The report template shows it at the top, above the decision.
- The decision is not changed: the farmer's pick stays authoritative.
- The upload form's crop menu starts on a disabled "Choose the crop" option and is required.

## Error handling

- A SAM chunk that fails follows the existing paths: a `BadRequest` is reported to the agent and counted toward the
  "rejected twice" stop failure; retries for 429 and 5xx are unchanged. Chunks that succeeded are still registered.
- `survey` with an unknown `crop_seen` or a negative `weeds_visible` is rejected like any invalid tool call.
- A stop that ends at a limit without `survey` or `finish` has no estimate or `all_weeds_marked`: the unmarked flag
  is decided from what exists (no estimate means only the `all_weeds_marked` trigger applies, and a missing
  `finish` counts as "not all marked").
- An overlay image that fails to render does not fail the mark: the JSON feedback is still returned.

## Testing

Offline unit tests, written first, no API calls (fake SAM and recorded responses as today):

- More than 16 boxes split into chunks of at most 16, sent in parallel, registered in box order.
- `survey` validation and storage; a repeated call overwrites.
- Mark feedback: `weeds_marked`, `weeds_estimate`, the hint at under 80%, and the overlay image attached.
- Patch mask: green AND box AND NOT dilated crop; specks under 20 px dropped; soil-colored pixels never counted.
- `refresh_patches` after a crop mark removes a crossing leaf from a patch.
- A mostly-crop patch is refused with its note.
- A patch's `plant_count` equals its green pieces; it makes the count approximate.
- The unmarked flag: both triggers, and neither when the estimate is met.
- Crop mismatch: the stop flag, `unsure` silent, the report banner, the decision unchanged.
- The upload form's crop menu has no default.
- New `Settings` defaults: 30 calls, 360 s, 10 stops at once.

All existing tests keep passing.

Live check: rerun the 10 test photos plus the horsetail photo through the real agent. Record per photo boxes,
patches, estimate, cover before and after, flags, `crop_seen` and seconds per stop, in `docs/spike-notes.md`. Pass
when a 10-photo walk finishes in under about 6 minutes, the horsetail in the near inter-row of the horsetail photo
is magenta, and its report warns that the photo looks like corn. Before-and-after overlays go to the user for review.
