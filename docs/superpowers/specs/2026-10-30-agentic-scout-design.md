# Agentic Weed Scout — Design Spec

- **Date:** 2026-10-30
- **Status:** Approved in brainstorming; ready for implementation planning
- **Working name:** weed-or-wheat (the product is a general weed-vs-crop scout; the name is not final)
- **Event:** Meta Global AI Developer Hackathon, Indie Developer track, theme "Build for Your World"
- **Deadline:** Wed 2026-11-04, 9:00 PM ET (target: submit by noon ET)

## 1. Summary

A farmer or agronomist uploads the photos from one walk through one field. For each photo (a "stop"), a Muse Spark agent drives SAM 3.1 to separate crop from weeds, zooms in on doubtful plants, identifies weed species and size, and has deterministic code measure weed intensity. Across the walk, code aggregates the measurements and applies agronomic threshold rules. The agent then writes a decision (no action, scout again, or treat) and looks up EPA herbicide labels to list products whose label covers this crop, this weed, and this weed size, each with a page citation.

The pitch: *point your phone at a field, see weeds vs crop, and get a decision you could act on, grounded in the label.*

## 2. Context and constraints

From the official participant guide (cerebralvalley.ai/e/meta-ai-developer-hackathon/details, read 2026-10-28):

- **Deliverables:** public GitHub repo, 3-minute demo video, live product.
- **New work only:** the demo may show only what was built during the event, and judges must be able to tell what is new. Prior repos (Kronia, sf-epa-data-ingestion) are inspiration only; no code is copied from them.
- **Projects to avoid** (listed by organizers) include "Basic RAG Applications" and "Simple Image Analyzers." The design avoids both: the core is an agent that measures and decides, and label lookup is a structured filter, not chat over documents.
- **Judging rubric:** Impact 30%, Use of Meta AI tooling 25%, Demo 25%, Technical depth 20%. Stage one is pass/fail on theme fit and use of the Meta Model API.
- **Bonus prize in reach:** Best Agentic Use-Case ($5K).
- **Credits:** $150 Meta Model API credit per participant (card on file required; claim by Nov 4).
- **SAM 3.1 API facts** (dev.meta.ai/docs/sam/segmenting): one short noun-phrase concept per request (no negation, no lists), or box/point prompts for up to 16 objects; returns a box and mask per match; optional per-record confidence; `one_bit` mask encoding available; video tracks up to 16 objects per frame; $2.50 per 1,000 images. The docs recommend using Muse Spark for scene understanding and passing concepts to SAM.

## 3. Goals and non-goals

**Goals (v1)**
1. Per-stop weed vs crop segmentation with mask overlays.
2. Per-stop intensity measures computed by code: weed cover %, weed count, density per m² for straight-down shots, species, size and stage.
3. Field-level summary across stops: frequency, mean cover, mean density, max size per species.
4. A decision with stated reasons: no action, scout again in N days, or treat (spot or broadcast) with urgency.
5. Label-backed product options with herbicide group, limits, rate, harvest interval, and page citation.
6. A visible, live log of the agent's steps.
7. Exportable report (print-to-PDF page plus per-stop CSV).
8. Deployed live product and a 3-minute demo video.

**Non-goals (v1)**
- Insects and diseases.
- Weather-based spray timing.
- Video input.
- Field maps beyond GPS pins on stops.
- Density from angled shots using row spacing as a scale.
- Native mobile app. The web page works in a phone browser.
- Any claim of being a prescription; options are framed as label-backed and "read the label before use."

## 4. User flow

1. **Start a walk.** Pick the crop: corn, soybean, or alfalfa. Optionally name the field.
2. **Upload 1–10 photos** from one visit (JPG, PNG, HEIC). If a photo carries GPS, its stop gets a map pin.
3. **Watch the agent work.** The page streams each stop's steps: segment, find candidates, zoom, identify, measure.
4. **Read the report:**
   - Per stop: overlay (crop green, weeds red, in-row zone shaded), measures, "check this" flags.
   - Field summary: metric cards (weed cover, density, frequency, decision).
   - Species table.
   - Label-backed options with citations.
5. **Export** with print-to-PDF of the report page, or CSV of per-stop metrics.

## 5. Intensity measures

| Measure | Source | When reported |
|---|---|---|
| Weed cover % | Weed mask pixels ÷ ground pixels | Always |
| Crop cover % | Crop mask pixels ÷ ground pixels | Always |
| Weed count | Separate weed pieces after crop subtraction | Always; marked approximate when masks merge (grass mats) |
| Density (plants/m²) | Count ÷ ground footprint | Straight-down shots only (see §8.4) |
| Species | Agent identification from zoom crops or close-ups | When identified; else "unidentified broadleaf" or "unidentified grass" |
| Height (in) and leaf count | Agent estimate from zoom crops | When visible; marked estimate |
| In-row vs between-row share | Weed area inside vs outside fitted row bands | When ≥2 crop rows are detected |
| Frequency | Share of stops where a species appears | Field level |

**Ground pixels:**
- For `top_down` and `close_up` views, ground pixels are all image pixels.
- For `row` and `wide` views, code makes one extra SAM call with the concept "sky" and removes that mask from the image. The ground is what remains.

## 6. Decision logic

Rules run in code (`agronomy/rules.py`) on the field summary. Muse Spark explains the outcome but does not choose it.

**Per species:**
1. **Zero tolerance.** If the species is flagged `zero_tolerance` (e.g., waterhemp, Palmer amaranth, because of resistance and the seeds they add to the soil) and appears at any stop → **treat**.
2. **Over threshold.** Else if mean density ≥ `action_density` (or mean cover ≥ `action_cover_pct` when density is unknown) → **treat**.
3. **Near threshold.** Else if mean density or cover ≥ 50% of the threshold → **scout again in 5–7 days**.
4. Else → **no action**.

**Field outcome:**
- The most severe species outcome wins.
- **Treat mode:** *spot* when the treated species appear at ≤ 1/3 of stops and mean weed cover < 5%; otherwise *broadcast*.

**Urgency for treat:** compare the largest weed height against the smallest `max_weed_height_in` among the matching label options.
- Height ≥ limit → "beyond label size limit" (options shown with that warning).
- Height ≥ 75% of limit → "treat within ~3 days."
- Otherwise → "treat within ~7 days."

All threshold values are configurable in `weeds.yaml` and must cite a university extension source. The 50% and 75% factors and the 1/3 and 5% spot-mode cutoffs are named constants in `rules.py`.

## 7. Architecture

### 7.1 Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, NumPy, Pillow with `pillow-heif`, the `openai` SDK, and `meta-sam-parser` ≥ 0.0.6.
- **Frontend:** one static HTML page with plain JS. It uses Server-Sent Events for the live step feed and shows server-rendered overlay PNGs.
- **Models:** through the Meta Model API (`base_url=https://api.meta.ai/v1`, key in `MODEL_API_KEY`).
  - Muse Spark: latest available id, confirmed on day 1 from dev.meta.ai/docs/models; default `muse-spark-1.3`.
  - SAM 3.1: id `sam-3.1`.
- **Hosting:** Hugging Face Spaces (Docker) with the API key as a secret.

### 7.2 Modules

```
app/
  main.py              FastAPI routes: POST /walks, GET /walks/{id}/events (SSE), GET /walks/{id}/report, GET /walks/{id}/csv
  config.py            env vars, model ids, limits
  models.py            Pydantic schemas (§7.3)
  ingest.py            upload handling: HEIC→RGB, EXIF orientation, resize ≤2048px, GPS extraction, image hash
vision/
  sam_client.py        segment_concept(image, phrase) / segment_prompts(image, objects) → list[Instance]; retries; cache by (image hash, prompt)
  masks.py             decode masks to bool arrays, union/subtract, connected pieces, area
  rows.py              fit crop rows as lines; in-row band masks
  measure.py           per-stop metrics from current instance labels
agent/
  tools.py             tool schemas and handlers (§8.1), bound to a StopState
  stop_agent.py        Muse Spark tool loop for one stop
  field_agent.py       field-level explanation and label lookup call
  prompts.py           system prompts
labels/
  focus_products.csv   EPA registration numbers chosen by the user (~20 corn/soy/alfalfa herbicides)
  build_labels.py      offline: EPA PPLS API → product facts + label PDF → Muse Spark extraction → use_records.json
  use_records.json     extracted, validated label use records (committed)
  lookup.py            find_label_options(crop, species, weed_height_in, crop_stage)
agronomy/
  weeds.yaml           ~33 species: ids, names, synonyms, type, thresholds, zero_tolerance, sources
  rules.py             decision logic (§6)
report/
  render.py            overlay PNGs, report HTML, CSV
  templates/report.html
web/
  index.html, app.js, style.css
tests/
  fixtures/            recorded SAM and Muse Spark responses for 5 photos
  test_*.py
eval/
  labels.csv           user's quick labels for ~15 photos
  run_eval.py          agreement report
```

### 7.3 Core schemas

```python
class Instance(BaseModel):
    id: str                      # "s3-i7"
    source: Literal["concept", "prompt"]
    prompt: str                  # noun phrase or "point"/"box"
    box: tuple[int, int, int, int]
    sam_confidence: float | None
    area_px: int
    role: Literal["crop", "weed", "ignore", "unknown"] = "unknown"
    species_id: str | None = None        # key into weeds.yaml
    height_in: float | None = None
    leaf_count: int | None = None
    id_confidence: Literal["high", "medium", "low"] | None = None
    check_this: bool = False

class StopResult(BaseModel):
    stop_index: int
    image_hash: str
    view_type: Literal["row", "top_down", "close_up", "wide"]
    gps: tuple[float, float] | None
    crop_stage: str | None       # e.g. "V3", agent estimate
    weed_cover_pct: float
    crop_cover_pct: float
    weed_count: int
    count_approximate: bool
    density_per_m2: float | None
    in_row_weed_share: float | None
    species: dict[str, int]      # species_id → count
    max_height_in: float | None
    flags: list[str]
    steps: list[AgentStep]       # the visible log
    status: Literal["ok", "no_crop_found", "failed"]

class UseRecord(BaseModel):
    reg_no: str
    product_name: str
    active_ingredients: list[str]
    herbicide_groups: list[str]          # WSSA/HRAC group numbers as on label
    crop: Literal["corn", "soybean", "alfalfa"]
    species_id: str
    weed_label_text: str
    timing: Literal["preemergence", "postemergence", "burndown"]
    max_weed_height_in: float | None
    crop_stage_min: str | None
    crop_stage_max: str | None
    rate_min: float | None
    rate_max: float | None
    rate_unit: str | None
    phi_days: int | None                 # pre-harvest interval
    rei_hours: int | None                # restricted-entry interval
    page: int
    quote: str                           # ≤200 chars, verbatim from that page

class AgentStep(BaseModel):
    t_ms: int                    # time since stop start
    tool: str                    # tool name, or "model" for a reasoning message
    summary: str                 # one line shown in the live step feed, e.g. 'Segment "plant": 14 found'
    instance_ids: list[str] = []
    ok: bool = True

class SpeciesSummary(BaseModel):
    species_id: str
    stops_present: int
    frequency: float             # stops_present ÷ stops with status "ok"
    mean_cover_pct: float        # mean over ok stops, 0 where absent
    mean_density_per_m2: float | None   # over top_down stops only; None if there are none
    max_height_in: float | None
    outcome: Literal["no_action", "scout_again", "treat"]

class FieldReport(BaseModel):
    walk_id: str
    crop: str
    stops: list[StopResult]
    species_summary: list[SpeciesSummary]   # frequency, mean cover, mean density, max height
    decision: Literal["no_action", "scout_again", "treat"]
    treat_mode: Literal["spot", "broadcast"] | None
    urgency: str | None
    rule_trace: list[str]                   # which rule fired, with numbers
    explanation: str                        # Muse Spark prose; contains no new numbers
    options: list[UseRecord]
```

## 8. Agent design

### 8.1 Stop agent tools

| Tool | Arguments | Returns | Notes |
|---|---|---|---|
| `segment` | `phrase: str` | instance summaries (id, box, confidence, area %) | One SAM concept call. Plural or case changes don't matter. |
| `segment_at` | `points: list[{x,y,positive}] \| box` | one instance summary | SAM box/point prompt; used to fix one doubtful plant |
| `zoom` | `instance_id` or `box` | image crop (padded, ≤768px) | Sent back to the model to look at |
| `classify` | `instance_id, role, species_id?, height_in?, leaf_count?, id_confidence` | ok / validation error | `species_id` must exist in `weeds.yaml` |
| `measure` | none | current metrics (§5) | Pure code |
| `finish` | `view_type, crop_stage?, notes` | ends loop | Rejected if any weed instance is still `unknown` and calls remain |

### 8.2 Loop

1. **Inputs.** The system prompt gives the crop, the allowed `species_id` list, and the procedure: segment crop concept → segment "plant" → review candidates → zoom/re-prompt doubtful ones → classify → measure → finish.
2. **Candidate generation in code.** After the crop and "plant" segments exist, `segment` responses include `candidates`: pieces of (union of "plant" masks − union of crop masks) larger than a minimum area (0.05% of image). These are registered as instances with `source="concept"` and `prompt="plant−crop"`.
3. **Fallback.** If the "plant" concept returns nothing useful, the agent marks weed points in a full-image view and calls `segment_at`.
4. **Limits.**
   - Max 12 tool calls per stop and 90 seconds per stop.
   - Stops run concurrently, at most 4 at a time.
   - On hitting a limit, code calls `measure` and closes the stop, with unclassified pieces counted as "unidentified" and flagged.
5. **Zoom images.** If the API rejects images in tool results, `zoom` returns a text ack and the loop appends the crop as a new user message with the image.

### 8.3 Field step

1. Code builds `species_summary` and runs `rules.py`, which fixes `decision`, `treat_mode`, `urgency`, `rule_trace`.
2. For each treated species, code calls `lookup.find_label_options`. The options are deduplicated by product and sorted to show different herbicide groups first.
3. Muse Spark gets the summary, the rule trace, and the options. It writes `explanation` (≤120 words) with no new numbers. A check rejects any number in the prose that does not appear in the summary or trace, and regenerates once. If it fails again, a template explanation is used.

### 8.4 Vision details

- **Mask decoding:** `meta-sam-parser` decodes one-bit masks to boolean arrays at the processed image size.
- **Rows:**
  - Take crop instance centroids and cluster them into lines with RANSAC, at most 6 lines.
  - Accept the row structure only if at least 2 lines each have at least 3 crop instances.
  - The in-row band is a line ± the median crop instance width.
- **Straight-down footprint:**
  - Assume a capture height of 1.2 m (user-adjustable) and a horizontal field of view of 69° (typical phone main camera).
  - Footprint width = 2·h·tan(FOV/2); height follows from the aspect ratio.
  - Density is shown only when `view_type == "top_down"`.

## 9. Label knowledge base (offline, built during the event)

1. The user fills `focus_products.csv` with ~20 common herbicides labeled for corn, soybean, or alfalfa: EPA registration number plus a short note.
2. `build_labels.py`, written fresh, does the following:
   - Calls the EPA PPLS API (`https://ordspub.epa.gov/ords/pesticides/cswu/ppls/<reg_no>`) for product name, active ingredients, sites, and pests.
   - Downloads the latest label PDF. The EPA server needs a legacy-TLS adapter.
   - Sends each PDF to Muse Spark with a JSON schema for a list of `UseRecord`, restricted to the three crops and to `species_id` values from `weeds.yaml` (synonyms provided).
   - Drops any record without a `page` and a `quote`.
   - Verifies each `quote` appears in the text of that page (PDF text extraction; normalized whitespace). It drops records that fail and logs the drop counts.
3. `use_records.json` is committed. EPA labels are public documents.
4. `lookup.find_label_options` keeps records where:
   - crop matches;
   - species matches;
   - the crop stage is within `[crop_stage_min, crop_stage_max]` when the stage and both limits are known;
   - timing is `postemergence`.

   It does not drop records on weed size. A record whose `max_weed_height_in` is below the current weed height is still returned, marked "over size limit", and sorted last.

## 10. Agronomy data

`weeds.yaml` entry:

```yaml
waterhemp:
  common_name: waterhemp
  scientific_name: Amaranthus tuberculatus
  type: broadleaf          # broadleaf | grass | sedge
  synonyms: [common waterhemp, tall waterhemp]
  action_density: null     # plants/m², from a cited source; null if zero_tolerance
  action_cover_pct: null
  zero_tolerance: true
  sources: []              # filled on Nov 1 with the extension publication(s) each threshold comes from
```

A species may ship with `sources: []` only if both thresholds are null and `zero_tolerance` is false. The catch-all defaults then apply.

Species list (v1, 33 ids): waterhemp, palmer_amaranth, redroot_pigweed, common_lambsquarters, giant_ragweed, common_ragweed, velvetleaf, horseweed, kochia, common_cocklebur, pennsylvania_smartweed, morningglory, jimsonweed, eastern_black_nightshade, giant_foxtail, green_foxtail, yellow_foxtail, barnyardgrass, large_crabgrass, fall_panicum, shattercane, woolly_cupgrass, volunteer_corn, canada_thistle, dandelion, common_milkweed, wild_mustard, field_pennycress, henbit, purple_deadnettle, common_chickweed, yellow_nutsedge, quackgrass. Plus two catch-alls: `unidentified_broadleaf` and `unidentified_grass`, which always carry a "check this" flag.

**Default threshold:** any species with no sourced threshold, including the catch-alls, uses `DEFAULT_ACTION_COVER_PCT = 5.0` in `rules.py`. This is a tool default, not an agronomic source, and the report labels it that way in the rule trace.

## 11. Error handling

| Case | Behavior |
|---|---|
| SAM returns zero matches | Normal result. For the crop concept the agent may try up to 2 alternative phrases (e.g., "soybean seedling"); then the stop is `no_crop_found` and is excluded from field averages, with a note. |
| HTTP 429 / 5xx | Retry up to 3 times with exponential backoff (1s, 2s, 4s). |
| HTTP 400 | No retry (per docs). The tool returns the error message to the agent once; a second 400 on the same tool fails the stop. |
| Stop fails | Marked `failed` in the report; other stops continue. Field step runs if ≥1 stop is `ok`. |
| Malformed tool arguments | Validation error returned to the agent; counts toward the call limit. |
| Low confidence | SAM confidence < 0.5 or `id_confidence == "low"` sets `check_this` and a visible flag. |
| No label match | Report states "No product in the current label set covers <species> on <crop>." Nothing invented. |
| Unsupported or huge upload | Rejected with a clear message. Max 10 photos, 25 MB each. |
| Cost guard (live site) | Max 10 photos per walk, 3 walks per IP per hour, results cached by image hash. Two preloaded demo walks run from cache. |

## 12. Testing and evaluation

- **Unit tests (pytest):** mask decode and area math on synthetic masks; crop subtraction and piece extraction; row fitting on synthetic centroids; footprint math; every branch of `rules.py`; `lookup.py` filters; quote verification in `build_labels.py`; the number check on the explanation.
- **Recorded fixtures:** real SAM and Muse Spark responses for 5 photos, saved on day 1–2, so the vision and agent plumbing can be tested offline at no cost.
- **Agreement check:** the user labels ~15 photos in `eval/labels.csv`: weeds present (y/n), main species, cover bucket (<2%, 2–10%, >10%). `run_eval.py` reports agreement per field. This is used to tune prompts and quoted in the demo if it is good.
- **Label check:** hand-verify 10 random `UseRecord`s against their PDF pages; target ≥9/10 correct before the demo.

## 13. Compliance and privacy

- New public repo created now; all code is written during the event. The README has a "Built during the hackathon (Oct 30 – Nov 4, 2026)" section and credits prior projects as inspiration only.
- `field-images/` and `.env` are git-ignored. Only the photos chosen for the demo walks are added, after review.
- GPS pins in the public demo and video are offset or blurred so the farm's location is not exposed.
- Every option shows "Read the label before use. The label is the law." The report says it is decision support, not a prescription.

## 14. Schedule

| Day | Work | Checkpoint |
|---|---|---|
| Fri Oct 30 | Claim credits and API key; confirm model ids; spike SAM on 6 photos ("soybean plant", "corn plant", "plant", "grass"); mask decode and overlay; first Muse Spark tool call; record fixtures. | **Go/no-go:** does "plant" minus crop give usable weed candidates? If not, switch to agent-marked points plus `segment_at` (§8.2.3). |
| Sat Oct 31 | Stop agent loop, tools, `measure`, rows; run on 10 photos. User picks focus products and demo walks. | Per-stop results look sensible |
| Sun Nov 1 | `build_labels.py`, `use_records.json`, `lookup.py`, `weeds.yaml`, `rules.py` | Lookup returns cited options for the demo species |
| Mon Nov 2 | Field step, report rendering, FastAPI routes, web page with SSE; end-to-end locally | Full walk runs end to end |
| Tue Nov 3 | Deploy to Spaces; agreement check; prompt tuning; record the 3-minute video; README | Live URL works; last day for organizer support |
| Wed Nov 4 | Buffer; final checks; submit by noon ET | Submitted |

## 15. Risks and fallbacks

1. **SAM "plant" merges grass mats or misses tiny seedlings.** Cover % stays valid; counts are marked approximate. Fallback: agent-marked points with `segment_at`.
2. **Images in tool results may be unsupported.** Fallback in §8.2.5.
3. **Label extraction errors.** Small product set, quote verification, hand check, citations visible to the user.
4. **Schedule slip.** Cut order if behind: CSV export → GPS pins → in-row/between-row split → straight-down density. Never cut: overlays, agent step log, decision with rule trace, cited options, live URL, video.

## 16. Stretch goals (only after the Mon Nov 2 checkpoint is green)

- Video row-walk: SAM 3.1 tracking for weeds per meter (sparse weeds only; 16 objects/frame cap).
- Density from angled shots using 30-inch row spacing as a scale.
- Disease severity on leaf close-ups (lesion area ÷ leaf area).
- Spray-condition check from a weather API.
