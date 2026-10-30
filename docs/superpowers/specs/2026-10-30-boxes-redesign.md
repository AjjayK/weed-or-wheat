# Redesign: weeds found with boxes (2026-10-30)

This note replaces the "plant minus crop" detection in the design spec (§8.2), based on the live tests in `docs/spike-notes.md`. It is the brief for rewriting plan Tasks 15–17 and touching Tasks 1, 7, 13 and 18.

## Why

SAM 3.1 text prompts don't find crop plants, leaves or weeds in crop canopies. "corn plant", "soybean plant", "leaf" and "weed" all return 0. Only "sky", "crop row", "vegetation" (the whole frame) and distinct objects such as "dandelion" and "flower" work.

Box prompts work:
- Muse Spark gives a tight box around each weed, in 0–1000 coordinates.
- All of a photo's boxes go to SAM in one multi-object request (up to 16 objects). That request took 1.3–3.1 s and returned one record per box.

Single points were worse: a point that lands on crop returns the whole row canopy. Pixel coordinates were less accurate than 0–1000.

Muse Spark takes 13–16 s per turn, so the stop loop must use few turns.

## Decisions

1. **Coordinates the agent sees and gives are 0–1000 on both axes.** x runs from the left to the right edge of the work image and y from top to bottom. Tools convert to work-image pixels, rounding, clamping to the image, and making every box at least 2 px on each side. Every box a tool reports back to the agent is also 0–1000.
2. **SAM client (Task 7)** gains `segment_boxes(image, boxes) -> list[SamRecord]`:
   - It takes 1–16 pixel boxes and raises `ValueError` outside that range.
   - It builds one multi-object prompt: object `i + 1` holds one box, with values coerced to int.
   - The result is cached like the other calls.
   - Record `object_id` is `str(i + 1)`. Boxes SAM returned nothing for are simply absent.
   - `segment_at` stays in the client, but no tool exposes it.
3. **Candidates are removed (Task 15).** That means `refresh_candidates`, `CANDIDATE_MIN_AREA_FRAC`, the `all_plants` kind and `source="candidate"` handling. `Instance.source` keeps `"candidate"` in its Literal, unused, since models are committed.
4. **Tools (Task 16)** are `segment`, `mark`, `zoom`, `classify`, `measure` and `finish`:
   - **`segment(phrase, kind)`**, kind `crop` or `weed`, takes a SAM concept:
     - "crop row" works as a crop phrase for row views.
     - Only distinct things work as weed phrases, for example "dandelion" or "flower".
     - Weed results pass the same crop-overlap check as `mark`.
   - **`mark(kind, boxes)`**, kind `crop` or `weed`, takes `boxes` as 1–16 items `{box: [x1, y1, x2, y2] in 0–1000, label: short text}` and makes one SAM request. The label is stored as the instance's `prompt`.
     - **crop:** each record becomes a crop instance.
     - **weed:** for each record, compute the overlap with crop, counting only crop instances that are *not* canopy-sized. Canopy-sized means spanning at least a third of the image's long side; otherwise every weed growing inside a row canopy would look like crop.
       - If at least 60% of the mask lies on that non-canopy crop, nothing is added and the agent is told "box N: mostly crop".
       - If the mask covers more than 12% of the frame, it is kept as one patch with `check_this`. The agent is told "box N: large patch; counted as one, approximate", and the stop's count is marked approximate.
       - Otherwise the full mask becomes a weed instance with role `weed`, id_confidence unset, and `check_this` when more than 30% of it overlaps any crop (a weed inside a canopy, which the agent should zoom on) or SAM's confidence is below 0.5.
       - Boxes SAM returned nothing for are reported as "box N: nothing found".
       - A weed `mark` that uses all 16 boxes marks the count approximate, because it is then a floor.
     - **Ordering:** within one turn, crop `mark`/`segment` calls run before the others, so the weed check always sees the crop.
   - **`zoom(instance_id | box)`**, where the box is in 0–1000.
   - **`classify`, `measure` and `finish`** are unchanged.
   - The tool's instance summary shows each box in 0–1000.
5. **Weeds win over crop.** In `compute_metrics` (Task 13) and `StopState.overlay`, crop pixels are `crop & ~weed_union`. A weed growing inside the row canopy keeps its pixels and counts as in-row.
6. **In-row zone (Task 15 `row_band`):**
   - Return None for alfalfa, which is solid-seeded, or when there is no crop.
   - If any crop instance spans at least a third of the image's long side, use `canopy_band(crop_union)`, which is already in Task 12.
   - Otherwise use `fit_rows` through crop-instance centers.
   - Remove sky pixels from the band when the sky mask is known.
7. **Stop procedure (Task 17 prompt),** aiming for 6 turns or fewer within the 12-call budget:
   1. In the first turn, call `mark` with kind `crop` (boxes around crop rows or plants; one tall box per visible row is fine) and `mark` with kind `weed` (a tight box around every weed, one per plant, nearest or largest first). Both calls can be in the same turn. For alfalfa, crop boxes are optional. A distinctive flowering weed can also be segmented by phrase.
   2. Read the results, zoom on up to 2 doubtful instances, and mark missed weeds once more if needed.
   3. Classify every weed instance in one call.
   4. Call `measure`, then `finish`.

   Always answer with tool calls. Every coordinate is 0–1000.
8. **`finalize` (Task 17):** a weed instance that is marked but never classified stays a weed with `check_this`. There are no candidate leftovers.
9. **Settings (Task 1):** `stop_timeout_s` goes from 90 to 240, because of 13–16 s turns.
10. **Robustness from earlier reviews:**
    - `run_stop` catches `StopFailed`, `BadRequest` and `LlmError`, so a failed stop keeps its step log.
    - `scripts/run_stop.py` (Task 18) and `evaluation/run_eval.py` (Task 33) catch per photo and continue.
