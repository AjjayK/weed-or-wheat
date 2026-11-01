# Spike notes (2026-10-30)

## SAM 3.1

- **Model ids confirmed:** `muse-image-1.0`, `muse-spark-1.1`, `muse-spark-1.2`, `muse-spark-1.2-contributor`, `muse-spark-1.3`, `muse-spark-1.3-contributor`, `muse-voice-transcribe-1.0`, `muse-wonder-1.0`, `sam-3.1`.
- **Mask raster equals box size:** yes. There were 0 resized masks across 24 calls.
- **Timing:** API calls took 0.5–3.6 s each. Decoding took at most 1.8 s (a full-sky mask) and was usually under 0.3 s.
- **Empty result:** "no match" comes back as a normal `message` with one `output_text` part whose text is `""`. It is not an error.
- **Metadata:** `include_confidence` and `mask_encoding` are honored. Confidences came back, and every mask was one-bit (`!`).

### Concept (text) prompts: 6 photos, `scripts/spike_sam.py`

| Photo | Crop prompt | "plant" | Extras |
|---|---|---|---|
| IMG_4214 (corn row, dense) | "corn plant" 0 | 0 | "grass" 0, "sky" 0 |
| IMG_4215 (corn row) | "corn plant" 0 | 7 small pieces (conf 0.50–0.57) | "grass" 0, "sky" 0 |
| IMG_4219 (soybean rows) | "soybean plant" 0 | 2 (0.56, 0.63) | "grass" 0, "sky" 1 (0.95) |
| IMG_4220 (soybean, wide) | "soybean plant" 0 | 0 | "grass" 1 (0.79), "sky" 1 (0.98) |
| IMG_3922 (alfalfa, wide) | "alfalfa plant" 0 | 0 | "dandelion" 36, "sky" 1 (0.98) |
| IMG_3926 (alfalfa, close) | "alfalfa plant" 0 | 0 | "dandelion" 18, "sky" 0 |

A follow-up probe (`scripts/probe_sam_phrases.py`) tried more phrases:

- **IMG_4214:**
  - "corn", "maize", "corn leaf", "leaf", "weed" and "grass blade" all returned 0.
  - "vegetation" returned 1 mask covering 99.9% of the frame.
- **IMG_4219:**
  - "soybean", "soybean leaf", "leaf", "vegetation" and "weed" all returned 0.
  - "crop row" returned 5 rows covering 61% (conf 0.64–0.70).
  - "seedling" returned 1 (0.51).
- **IMG_3926:**
  - "alfalfa", "clover", "leaf" and "weed" all returned 0.
  - "vegetation" returned the whole frame.
  - "flower" returned 15 (conf 0.92).

**"Plant" minus crop is a no-go.** In crop canopies SAM 3.1 does not pick out individual crop plants, leaves or weeds from a text prompt. Text prompts are still useful for:

- "sky", to find the ground area.
- "crop row", for the crop in row views.
- Visually distinct weeds such as "dandelion" or "flower" in alfalfa.

### Point and box prompts: IMG_4219 (`spike_out/point_prompts_IMG_4219.jpg`)

| Prompt | Result |
|---|---|
| Point on a weed in open soil (285, 640) | That weed only, 2,812 px, conf 0.87 |
| Box around the same weed | Same mask, 3,073 px, conf 0.87 |
| Point on a weed at the row edge (240, 745) | The weed plant, 9,516 px, conf 0.14 (low, but the mask is right) |
| Point on a soybean (430, 490) | The whole connected right-hand row canopy, conf 0.79 |
| Point on a soybean (100, 650) | The whole left-hand row canopy, conf 0.75 |

**Decision (user, 2026-10-30): go with pointing** (spec §8.2.3 fallback, now the primary path):

- Muse Spark looks at the photo and gives a point on each weed.
- SAM segments each point as its own object; one request can carry up to 16 objects.
- Points on the crop (or "crop row") give the crop mask.
- Muse Spark verifies each weed by zooming in.
- Task 11 checks whether Muse Spark's points actually land on weeds.

## Muse Spark (Task 11, `scripts/spike_agent.py`, model `muse-spark-1.3`)

- **Tool call round trip: works.**
  - `report_rows` came back with `{"count": 7, "notes": ...}`.
  - The schema left `notes` optional and there was no 400, so Meta does not force strict mode on tools.
- **Image in a follow-up user message (chained with `previous_response_id`): works.** It correctly described the corn close-up.
- **Structured output (strict json_schema): works.** It returned `{"crop": "corn", "weeds_visible": true}`.
- **Latency: 13–16 s per turn.** The agent loop has to use few turns. At about 12 turns a stop would take 2–3 minutes.

### Pointing accuracy (`spike_out/pointing_sheet.jpg`)

One point per weed, each point sent to SAM separately:

| Photo | Pixel coordinates | 0–1000 coordinates |
|---|---|---|
| IMG_4219 (soybean) | 8 points. Most sit on the soybean rows, so SAM returned whole rows: unusable. | 7 points. About half sit on inter-row weeds; the rest hit rows. |
| IMG_4215 (corn) | 36 points in the inter-row. SAM merged them into one large strip. | 30 points on the inter-row weeds. The pieces are mostly weeds, plus one corn leaf. |
| IMG_3926 (alfalfa) | 14 points next to dandelions | 12 points next to dandelions |

**0–1000 coordinates are clearly better than pixels.** A single point on crop is the worst case, because SAM returns the whole row.

### Boxes instead of points (`scripts/probe_boxes.py`, `spike_out/boxes_sheet.jpg`)

Muse Spark gave a tight box per weed in 0–1000 coordinates, and all of a photo's boxes went to SAM as one multi-object request (up to 16):

| Photo | Muse Spark | SAM (one request) | Result |
|---|---|---|---|
| IMG_4219 | 8 boxes, 14 s ("waterhemp", "pigweed", "common ragweed") | 8 records, 1.3 s | 4 are tight masks on inter-row weeds, 3 sit on soybean foliage at the row edges |
| IMG_4215 | 16 boxes, 11 s ("horsetail") | 16 records, 1.4 s | Most are on the inter-row weeds, 1 is on a corn leaf |
| IMG_3926 | 14 boxes, 8 s ("dandelion flower") | 14 records, 3.1 s | Small boxes sit slightly off the flowers, and SAM fills the box |

**Decision: weeds are found with boxes in 0–1000 coordinates, all sent in one SAM request.** It needs 2 turns instead of up to 16 SAM calls, and boxes bound SAM's mask so it can't swallow a whole row. Remaining errors (boxes on crop, small boxes off target) are handled by:

- Subtracting the crop mask.
- Treating a "weed" that mostly overlaps crop as a crop hit.
- The agent zooming in to verify.


## Stop agent on real photos (Task 18, `scripts/run_stop.py`)

**First run, 10 photos, three crops in parallel:**
- 7 stops finished `ok`. Each used 3–5 turns and took 47–111 s.
- 3 stops failed with `400 referenced response not found or expired`. Meta sometimes loses the previous response a couple of seconds after returning it.
- Weed boxes drawn on soybean foliage near the camera passed as large "patches" (IMG_4219, IMG_4220).

**Fixes:**
- **Lost responses:** when this error happens, the turn is resent with the full conversation and no `previous_response_id`, and the rest of the stop stays stateless. A live smoke test confirmed Meta accepts replayed `function_call` items.
- **Crop-sized patches:** a patch-sized weed mask that lies at least 50% on any crop (row canopies included) is now vetoed as "mostly crop".
- **Prompt:** two new hints. Crop leaves close to the camera are crop, and crop boxes should hug the rows.

**Rerun of the 5 affected photos:** all `ok`. IMG_4220 hit the lost-response error once and recovered with the full resend.

| Photo | Crop | Weed cover | Weeds | Species |
|---|---|---|---|---|
| IMG_4214 | corn | 11.8% | 12 | unidentified grass |
| IMG_4215 | corn | 5.9% | 12 | field horsetail |
| IMG_6564 | corn | 6.8% | 14 | unidentified broadleaf (all low confidence) |
| IMG_4547 | corn (wide) | 11.2% | 5 | giant ragweed |
| IMG_4219 | soybean | 9.6% | 6 | waterhemp |
| IMG_4220 | soybean | 10.2% | 8 | waterhemp |
| 3AEE1239 | soybean (no-till) | 0% | 0 | none |
| IMG_3922 | alfalfa (wide) | 1.4% | 45 | dandelion (16-box cap hit) |
| IMG_3926 | alfalfa | 2.4% | 23 | dandelion |
| IMG_6942 | alfalfa | 0% | 0 | none |

**Open questions for review by eye** (overlays in `run_out/`):
- Is the strip along the left soybean row in IMG_4220 really weeds?
- Are the yellowish plants in the 3AEE1239 furrows weeds the agent missed?
- In the no-till soybean photo, the crop masks spill over the soil between rows.

## Label table (Task 23, 2026-11-01)

**Trial, 2 products (old prompt):** Roundup PowerMAX 3 kept 57 of 58 records, Callisto 18 of 18, in 7.5 minutes. Two prompt fixes followed:
- Roundup's soybean section points to a weeds table elsewhere in the label, and only 6 soybean records came back. The prompt now says to follow such a pointer. Result after the fix: 25 soybean records.
- Callisto's table rates some weeds "PC" (partial control, which the label calls below commercially acceptable) when Callisto is used alone. The prompt now skips partial-control and suppression-only weeds. Palmer amaranth, kochia, horseweed, morningglory, yellow nutsedge and common ragweed then dropped out of Callisto's corn records.

**Full build, 20 products:** 483 records kept, 44 dropped by the quote check, in 45 minutes.
- **Cadet kept 0 of 16:** its weed table (Table 2) is an image in the PDF, so none of the quotes appear in the PDF's text layer and the quote check dropped them all.
- **Partial drops:** Accent Q 6, Raptor 12, Butyrac 200 6, Roundup 3, Pursuit 1.
- **Not shown as options:** 16 records are tagged something other than postemergence (Callisto on soybean is preemergence; one Select Max corn record is burndown), and the lookup only lists postemergence uses.
- **Weak quotes:** 239 of the 467 postemergence records quote a weed's row in a label table (for example "Waterhemp, common"). This proves the weed is listed on that page. The height limit and rate come from elsewhere on the label, so the hand check should look at those.
- **Demo pairs (postemergence products):**
  - waterhemp in soybean: 6
  - giant ragweed in corn: 8
  - giant ragweed in soybean: 9
  - giant foxtail in corn: 5
  - dandelion in alfalfa: 1 (Sencor)
  - field horsetail in corn: 0

**Hand check (Step 5):** pending, done by the user.
