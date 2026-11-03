# Live redesign: design

Date: 2026-11-02. Status: approved in brainstorming, awaiting spec review.

## Goal

Make the app look modern and make the wait worth watching. While a walk runs, viewers see what the agent sees and
thinks, photo by photo: the photo, each marking picture, each close-up, the final overlay, and the agent's own
reasoning. Sample walks replay that live view from saved events, so judges and first-time visitors see the agent
work without spending API credit.

Decisions made with the user:

- Scope: the upload page, the live walk view and the report page, in one look, plus sample-walk replay.
- Look: "field notebook". Cream page (#F5F2E8), ink green text (#1D2A1F, #3F4A3C), magenta accent (#B0127F),
  Domine serif headings and Public Sans body (Google Fonts), the same palette as the two decks. Overlay colors are
  unchanged: green crop, magenta weeds, blue weed areas, yellow checks.
- Live layout: a stage for one photo plus a strip of all photos.
- Approach: plain HTML, CSS and JavaScript with no build step (technical decision 2). Pictures are saved as JPEG
  files and streamed as URLs. Every event is saved for replay.

Probe (2026-11-02): with `reasoning={"summary": "auto"}`, Muse Spark returns reasoning items whose `summary` holds
readable lines, for example "Analyzing corn field image to count inter-row weeds and box visible crop rows". It also
returns an assistant message with a one-line narration, for example "Scouting the corn rows — marking crop, weeds,
and dense patches to measure". Without the parameter, the summaries are empty.

## 1. What the agent sends, and how it is stored and replayed

### Reasoning

- `Llm.turn` gains `reasoning_summary: bool = False`. When true, the request carries
  `reasoning={"summary": "auto"}`. `LlmTurn` gains `reasoning: list[str]`, the text of every reasoning summary part,
  in order. `LlmTurn.text` (the narration) is already kept.
- `run_stop` passes `reasoning_summary=True`. Label extraction and the field explanation do not.
- After each turn that produced narration or reasoning, the stop loop logs one step: `tool="think"`, `summary` = the
  narration (or the first reasoning line when there is no narration), `detail` = the reasoning lines (at most 4, each
  at most 200 characters). This happens before the turn's tool calls run. `AgentStep` gains
  `detail: list[str] = []`.
- "think" steps are kept in `StopResult.steps` like any other step.

### Pictures

- `StopState` gains `picture_sink: Callable[[Image.Image], str] | None`. It is given to `run_stop` and passed through
  to the state. `StopState.log(..., picture: Image.Image | None = None)` calls the sink when both are present and
  stores the returned name on the step. `AgentStep` gains `picture: str | None = None`.
- Pictures are logged at these points:
  - a stop starts: a "start" step with the work photo;
  - after every crop, weed or patch mark: the overlay, the same picture the agent gets;
  - every zoom: the close-up crop;
  - when the stop is finalized: a "done" step with the final overlay. This happens in the pipeline, after
    `run_stop`, using `state.overlay()`.
- The pipeline's sink saves pictures to `store/<walk>/steps/s<stop>-<nn>.jpg`: JPEG quality 80, long side at most
  1024 px, `nn` counting up per stop from 00. It returns the file name.
- `GET /walks/{walk_id}/steps/{name}` serves a picture from the walk's folder, either the store or demo. It only
  accepts names matching `s\d{1,2}-\d{2,3}\.jpg` and returns 404 otherwise.

### Events

Every event is a dict with `type` and `t`, the seconds since the walk started (float, 0.1 s resolution). `Walk.emit`
stamps `t`.

| type | fields |
|---|---|
| `walk_start` | `crop`, `stops`: list of `{stop, name}` (upload file names) |
| `stop_start` | `stop`, `picture` (URL) |
| `step` | `stop`, `tool`, `summary`, `ok`, `picture` (URL or null) |
| `thought` | `stop`, `text`, `detail` (list) |
| `stop_done` | `stop`, `status`, `cover`, `weeds`, `picture` (URL of the final overlay) |
| `report_building` | none |
| `report_ready` | none |
| `error` | `message` |

The pipeline's `on_step` turns a "start" step into `stop_start`, a "think" step into `thought`, and a "done" step
into `stop_done` with the stop's numbers. Every other step becomes `step`. Picture URLs are
`/walks/{walk_id}/steps/{name}`. `walk_start` is emitted by `Pipeline.run` before any stop starts. It needs the
upload names, which `main.py` passes into `Pipeline.run`. `report_building` is emitted after the last stop.

### Replay

- When a walk finishes and its report is written, the pipeline also writes `events.json` atomically, as it does for
  `report.json`. A walk where every photo failed writes neither.
- `GET /walks/{walk_id}/events` for a walk that is not running but has a saved `events.json`, in the store or demo,
  streams those events. It waits between events by the gap in `t`, divided by `REPLAY_SPEED` (4) and capped at
  `REPLAY_MAX_GAP_S` (3 s). A `replay=1` field is added to every replayed event. A cached walk without `events.json`
  (reports from before this change) keeps today's behaviour: a single `report_ready`.
- Tests set the replay speed so there is no waiting.

## 2. Upload page

- Header: the kicker "Weed Scout", a one-line promise in Domine, and one sentence on what the agent does.
- Step 1, crop: three large radio cards (corn, soybean, alfalfa). None is checked at first, and the radio group is
  required. This replaces the select.
- Step 2, photos: a drop zone wrapping the file input. It also opens the picker when tapped. It shows thumbnails
  (object URLs), the count and the total size. In the browser it stops more than 10 photos, or any photo over 25 MB,
  with an inline message. The server checks everything again.
- A primary button, "Scout this field". It is the only filled button on the page.
- Sample walks, from `/samples`: one card each, with a thumbnail, the title, the stop count and the decision.
  "Watch the agent" opens the live view in replay; "Report" opens the report. `/samples` entries gain `thumbnail`
  (a step picture URL), `stops` and `decision`, read from the sample's saved report and events.
- Footer: "Pins are shifted for privacy. Decision support, not a prescription."
- Phone: the columns stack, with the form first.

## 3. Live walk view

- It is the same page in a live mode. The URL becomes `/?walk=<id>`. Loading that URL opens the live view and
  connects to the events. EventSource's Last-Event-ID resume works as today.
- Header: "Scouting your <crop> field", a progress bar, "N of M photos done" and an elapsed timer. In replay, a badge
  reads "Replay, 4× speed" with a "Skip to report" link.
- Photo strip: one thumbnail per photo. Before `stop_start` it is a gray placeholder; after that it shows the photo.
  Each has a status: waiting (gray), working (pulsing magenta outline), done (green check) or failed (red mark).
  - The stage follows the photo with the newest event. Tapping a thumbnail pins it; "Follow live" unpins.
  - The strip scrolls sideways on a phone.
- Stage, left:
  - the photo's latest picture, crossfading when a new one arrives, with its caption: "Photo", "What is marked now",
    "Close-up" or "Final overlay";
  - below it, the latest thought in Domine italic, and up to three detail lines in small text.
- Stage, right: a checklist of phases, built from the photo's steps:
  - survey;
  - crop;
  - weeds (weed or patch marks);
  - check the picture (a mark after the first weed mark, or a reclassify);
  - zoom;
  - name (classify);
  - measure.

  Each phase is pending, current or done, with the latest step summary for that phase. Under the checklist goes the
  latest counts line from step summaries, plus "calls and seconds left" when known.
- End:
  - after the last `stop_done`, the header reads "Building the report";
  - on `report_ready`, a "See the report" button appears;
  - the page then navigates to the report after 3 s, unless a photo is pinned.
- Errors:
  - a failed photo shows its last step in red on the stage;
  - a connection error shows "Reconnecting…";
  - an EventSource that closes for good (unknown walk) shows "This walk isn't running any more" with a link back to
    upload.
- Accessibility: the thought area is `aria-live="polite"`. Strip thumbnails are buttons labeled "Photo N, status".

## 4. Report page

The content is unchanged; the layout and look follow section 2.

- Top: the crop-mismatch banner when present, the title "<Crop> field, N stops", and a large decision card in the
  outcome color with treat mode and urgency. Under it, the number cards (mean cover, density, stops with weeds).
- "Why": the explanation in Domine, with the rule trace behind a "How the rules decided" `<details>`.
- The weeds table and the options table, restyled as clean rows. Seed-trait and over-size warnings show as small
  amber tags, and page citations are links.
- Stops: a card per stop with the overlay, its flags as tags, and the legend (which includes blue weed areas). New:
  a "How the agent worked" `<details>` per stop. It shows that stop's steps in order: thoughts in italic, tool steps
  as plain lines, and a 64 px thumbnail for each step picture.
- Map and footer as today.
- Print and PDF: `<details>` print open, colors keep their fill, and step thumbnails are hidden. The CSV is unchanged.
- Old cached reports, whose steps have no `picture` or `detail`, render without thumbnails or thoughts.

## 5. Testing, sample walks and checking it works

Offline tests, written first, no API calls:

- `Llm.turn`:
  - sends `reasoning={"summary": "auto"}` only when asked;
  - fills `LlmTurn.reasoning` from fake response objects whose reasoning items carry summary parts;
  - leaves it empty when there are none.
- `run_stop` logs one "think" step per turn with narration and detail, before that turn's tool steps.
- Pictures:
  - the sink is called for the start step, after each crop, weed or patch mark, and on each zoom;
  - the returned names are on the steps;
  - with no sink, no picture is recorded and nothing breaks.
- The pipeline (with a fake stop runner and a real temp store):
  - emits `walk_start` with names, `stop_start` and `stop_done` with picture URLs, `report_building` and
    `report_ready`, all with `t`;
  - writes the step pictures and `events.json`;
  - writes neither for an all-failed walk.
- `main.py`:
  - the step-picture route serves a saved picture;
  - it returns 404 for a bad name, a path trick or an unknown walk;
  - a cached walk with `events.json` replays its events (with the replay speed patched to instant), with
    `replay` set;
  - one without it sends `report_ready` only;
  - `/samples` returns `thumbnail`, `stops` and `decision`;
  - `/` serves the radio crop cards (required, none checked), the drop zone and the live view's containers.
- Report:
  - renders the decision card, the banner and the stop timeline with thumbnails and thoughts;
  - an old report without pictures renders.
- The existing tests keep passing. Tests that assert the old select markup are updated to the radio cards.

The front end is plain JavaScript without a test framework. It is checked in the browser.

Sample walks (formerly Task 31):

- Run two real walks through the app on the user's photos: corn (including IMG_4215, horsetail) and soybean
  (waterhemp).
- With the GPS offset on, copy each walk's folder (report, events, overlays, step pictures) to `demo/<walk_id>/`.
- List them in `demo/samples.json` with a title.
- Only photos the user approves are used, because `demo/` is committed and published with the repo.

Browser check, using the built-in browser against the local server:

- the upload page at desktop and phone widths;
- one live walk from upload to report;
- a sample replay;
- the report, including print preview.

Screenshots of each go to the user.

## Out of scope

- Dark mode.
- A JavaScript framework.
- Changes to measuring, rules or labels.
- Thumbnails in the CSV.
- Live view pages for walks that failed outright.
