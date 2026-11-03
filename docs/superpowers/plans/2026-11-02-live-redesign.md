# Live Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A modern "field notebook" app where viewers watch the agent work photo by photo, with every marking picture, close-up and reasoning line, and where sample walks replay that live view from saved events.

**Architecture:**
- **Reasoning:** Muse Spark's reasoning summaries and narration become "think" steps.
- **Pictures:** the stop state saves step pictures through a sink the pipeline provides: JPEG files under `store/<walk>/steps/`.
- **Events and replay:**
  - Events carry `t`, picture URLs and new event types: `walk_start`, `stop_start`, `thought`, `stop_done`, `report_building`.
  - The pipeline saves events to `events.json`.
  - The events endpoint replays them, paced, for cached walks.
- **Front end:** plain HTML, CSS and JS with no build step. The upload view and the live view (stage plus strip) share one page. The report template is restyled in the same look.

**Tech Stack:**
- Python 3.10, FastAPI, Pydantic v2, Pillow, Jinja2, pytest.
- Plain JavaScript (EventSource, fetch).
- CSS with custom properties.
- Google Fonts: Domine and Public Sans.

**Spec:** `docs/superpowers/specs/2026-11-02-live-redesign-design.md`.

**Conventions for every task:**
- Run tests with `.venv/Scripts/python -m pytest ...`. Tests never call the network.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never print, echo or log `.env` or `MODEL_API_KEY`. Do not push. `docs/drafts/` stays untracked.
- Keep lines within 120 characters, and keep files LF in the working tree.
- One deviation from the spec, agreed while planning: the stop-start photo and the final overlay are saved by the pipeline itself, not through "start" and "done" steps. So `run_stop`'s step list gains only "think" steps.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `scout/agent/llm.py` | Modify | Ask for reasoning summaries; `LlmTurn.reasoning` |
| `scout/models.py` | Modify | `AgentStep.detail`, `AgentStep.picture` |
| `scout/agent/state.py` | Modify | `picture_sink`; `log(..., detail, picture)` |
| `scout/agent/tools.py` | Modify | Marks and zooms log their picture; the overlay is rendered once per mark |
| `scout/agent/stop_agent.py` | Modify | Reasoning on; a "think" step per turn; `picture_sink` passthrough |
| `scout/app/pipeline.py` | Modify | Event `t`; new events; step picture files; `events.json`; `step_picture`; `saved_events`; richer `samples` |
| `scout/app/main.py` | Modify | Pass photo names; step-picture route; paced replay |
| `scout/report/render.py`, `scout/report/templates/report.html` | Modify | Field notebook report; decision card; agent timeline |
| `scout/web/index.html`, `scout/web/style.css`, `scout/web/app.js` | Rewrite | Upload view and live view |
| `tests/test_*.py` | Modify | Tests listed per task |
| `demo/` | Create (Task 10) | Two sample walks plus `samples.json` |

---

### Task 1: Ask Muse Spark for reasoning summaries

**Files:** Modify `scout/agent/llm.py`. Test: `tests/test_llm.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_llm.py`:

```python
def test_reasoning_summaries_are_asked_for_and_read():
    seen = {}

    def create(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return _response([NS(type="reasoning", summary=[NS(type="summary_text", text="Counting weeds."),
                                                         NS(type="summary_text", text="Boxing the rows.")]),
                          NS(type="message", role="assistant",
                             content=[NS(type="output_text", text="Scouting the rows.")]),
                          NS(type="function_call", call_id="c1", name="survey", arguments="{}")],
                         text="Scouting the rows.")

    turn = Llm(create, "m").turn("x", reasoning_summary=True)
    assert seen["reasoning"] == {"summary": "auto"}
    assert turn.reasoning == ["Counting weeds.", "Boxing the rows."] and turn.text == "Scouting the rows."
    Llm(create, "m").turn("x")
    assert "reasoning" not in seen


def test_reasoning_without_summaries_is_empty():
    turn = Llm(lambda **k: _response([NS(type="reasoning")]), "m").turn("x")
    assert turn.reasoning == []
```

- [ ] **Step 2: Run them.** `.venv/Scripts/python -m pytest tests/test_llm.py -v`. Expected: FAIL (`unexpected keyword argument 'reasoning_summary'`).

- [ ] **Step 3: Implement.** In `scout/agent/llm.py`:

Add a field at the end of `LlmTurn`:

```python
    reasoning: list[str] = field(default_factory=list)  # the reasoning summary lines, when asked for
```

Change the `turn` signature and the request:

```python
    def turn(self, input, instructions: str | None = None, tools: list[dict] | None = None,
             previous_response_id: str | None = None, text_format: dict | None = None,
             reasoning_summary: bool = False) -> LlmTurn:
        kwargs: dict[str, Any] = {"model": self.model, "input": input}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if text_format:
            kwargs["text"] = {"format": text_format}
        if reasoning_summary:  # readable lines of the model's thinking, shown to people watching the agent
            kwargs["reasoning"] = {"summary": "auto"}
```

In the output loop, collect reasoning before the existing branches, and return it:

```python
        calls, items, reasoning = [], [], []
        for item in response.output:
            kind = getattr(item, "type", None)
            if kind == "reasoning":
                reasoning += [part.text for part in getattr(item, "summary", None) or [] if getattr(part, "text", None)]
            elif kind == "message" and getattr(item, "role", "assistant") == "assistant":
```

Keep the rest of the loop unchanged: the message branch and the `elif kind == "function_call"` branch. End with:

```python
        return LlmTurn(response.id, calls, getattr(response, "output_text", "") or "", items, reasoning)
```

- [ ] **Step 4: Run the tests.** `tests/test_llm.py`, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit.**
```bash
git add scout/agent/llm.py tests/test_llm.py
git commit -m "feat: ask Muse Spark for reasoning summaries when watching matters" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Steps carry detail lines and a saved picture

**Files:** Modify `scout/models.py` and `scout/agent/state.py`. Test: `tests/test_state.py`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_state.py`. Add `make_image` to its `tests.helpers` import if it is missing.

```python
def test_log_saves_a_picture_through_the_sink_and_names_it_on_the_step():
    saved = []
    state = StopState(make_image(), "soybean",
                      picture_sink=lambda image: saved.append(image.size) or f"s0-{len(saved):02d}.jpg")
    step = state.log("mark", "Mark weed: 1 of 1 boxes kept", picture=state.image.work, detail=["a"])
    assert step.picture == "s0-01.jpg" and step.detail == ["a"] and saved == [(100, 100)]
    assert state.log("zoom", "Zoom on s0-i1", picture=state.image.work).picture == "s0-02.jpg"
    assert state.log("classify", "Classified 1 plants (1 weeds)").picture is None
    assert StopState(make_image(), "soybean").log("mark", "x", picture=make_image().work).picture is None


def test_a_picture_that_fails_to_save_does_not_end_the_stop():
    def broken(image):
        raise OSError("disk full")

    step = StopState(make_image(), "soybean", picture_sink=broken).log("mark", "x", picture=make_image().work)
    assert step.picture is None and step.summary == "x"
```

- [ ] **Step 2: Run it.** `.venv/Scripts/python -m pytest tests/test_state.py -v`. Expected: FAIL (`unexpected keyword argument 'picture_sink'`).

- [ ] **Step 3: Implement.**

In `scout/models.py`, `AgentStep` becomes:

```python
class AgentStep(BaseModel):
    t_ms: int
    tool: str
    summary: str
    instance_ids: list[str] = Field(default_factory=list)
    ok: bool = True
    detail: list[str] = Field(default_factory=list)  # a "think" step's reasoning lines
    picture: str | None = None  # file name of the picture saved for this step, if any
```

In `scout/agent/state.py`:
- Add `import logging` to the imports and `_log = logging.getLogger(__name__)` after them.
- Give `StopState.__init__` a new keyword parameter, `picture_sink: Callable[[Image.Image], str] | None = None`, after `clock`. Store it as `self._picture_sink = picture_sink`.
- Replace `log` with:

```python
    def log(self, tool: str, summary: str, instance_ids=(), ok: bool = True, detail: Sequence[str] = (),
            picture: Image.Image | None = None) -> AgentStep:
        name = None
        if picture is not None and self._picture_sink is not None:
            try:
                name = self._picture_sink(picture)
            except Exception:  # pictures are for people watching; a failed save must not end the stop
                _log.warning("step picture could not be saved", exc_info=True)
        step = AgentStep(t_ms=int((self._clock() - self.started) * 1000), tool=tool, summary=summary,
                         instance_ids=list(instance_ids), ok=ok, detail=list(detail), picture=name)
        self.steps.append(step)
        if self._on_step:
            self._on_step(step)
        return step
```

- [ ] **Step 4: Run the tests.** `tests/test_state.py`, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit.**
```bash
git add scout/models.py scout/agent/state.py tests/test_state.py
git commit -m "feat: steps carry reasoning lines and a saved picture" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Marks and zooms log their picture

**Files:** Modify `scout/agent/tools.py`. Test: `tests/test_tools.py`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_tools.py`:

```python
def test_marks_and_zooms_hand_their_pictures_to_the_step():
    saved = []
    state = StopState(make_image(100, 100), "soybean",
                      picture_sink=lambda image: saved.append(image) or f"s0-{len(saved):02d}.jpg")
    runner = ToolRunner(state, FakeSam(), small_weedbook())
    result = runner.run("mark", {"kind": "weed", "boxes": [{"box": WEED_A, "label": "w"}]})
    assert state.steps[-1].tool == "mark" and state.steps[-1].picture == "s0-01.jpg"
    assert saved[0].size == state.size and [c for c, _ in result.images] == [OVERLAY_CAPTION]
    runner.run("zoom", {"instance_id": state.steps[-1].instance_ids[0]})
    assert state.steps[-1].tool == "zoom" and state.steps[-1].picture == "s0-02.jpg"
```

- [ ] **Step 2: Run it.** `.venv/Scripts/python -m pytest tests/test_tools.py -v`. Expected: the new test FAILS (`picture` is None).

- [ ] **Step 3: Implement.** In `scout/agent/tools.py`:

Add a method to `ToolRunner`, just before `_feedback`:

```python
    def _render_overlay(self):
        """The marking picture, rendered once per mark: for the agent's feedback and for people watching."""
        try:
            return self.state.overlay()
        except Exception:  # the picture helps; the counts in the answer still stand without it
            log.warning("overlay picture failed", exc_info=True)
            return None
```

Change `_feedback` to take the picture instead of rendering it. Its signature becomes
`def _feedback(self, out: dict, kind: str, picture) -> ToolResult:`. Replace its final `try ... except ... return` block with:

```python
        if picture is None:
            out["picture"] = "unavailable"
            return ToolResult(json.dumps(out))
        return ToolResult(json.dumps(out), images=[(OVERLAY_CAPTION, to_data_url(picture, quality=OVERLAY_QUALITY))])
```

In `_mark_patches`, replace its last two lines with:

```python
        picture = self._render_overlay()
        self.state.log("mark", f"Mark patch: {len(found)} of {len(pixel_boxes)} areas kept", [f["id"] for f in found],
                       picture=picture)
        return self._feedback({"found": found, "notes": notes}, "patch", picture)
```

In `_tool_mark`, replace its last two lines with:

```python
        picture = self._render_overlay()
        self.state.log("mark", f"Mark {kind}: {len(found)} of {len(boxes)} boxes kept", [f["id"] for f in found],
                       picture=picture)
        return self._feedback({"found": found, "notes": notes}, kind, picture)
```

In `_tool_zoom`, change the log line to:

```python
        self.state.log("zoom", f"Zoom on {label}", [instance_id] if instance_id else [], picture=crop)
```

The existing test that makes `state.overlay` raise still passes: `_render_overlay` logs the same warning, and `_feedback` sets `picture` to "unavailable".

- [ ] **Step 4: Run the tests.** `tests/test_tools.py`, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit.**
```bash
git add scout/agent/tools.py tests/test_tools.py
git commit -m "feat: marks and zooms log their picture for people watching" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Each turn logs the agent's thought; the stop passes the picture sink on

**Files:** Modify `scout/agent/stop_agent.py`. Test: `tests/test_stop_agent.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_stop_agent.py`:

```python
def test_each_turn_logs_the_agents_thought_before_its_tools():
    turn = LlmTurn("r1", [MARK_CROP, MARK_WEEDS], " Marking the rows and the weeds. ", [], ["Counting weeds.", "x" * 300])
    llm = FakeLlm([turn, LlmTurn("r2", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm)
    think = result.steps[0]
    assert think.tool == "think" and think.summary == "Marking the rows and the weeds."
    assert think.detail == ["Counting weeds.", "x" * 200] and result.steps[1].tool == "mark"
    assert [s.tool for s in result.steps].count("think") == 1
    assert llm.requests[0]["reasoning_summary"] is True


def test_a_turn_with_only_reasoning_uses_its_first_line():
    turn = LlmTurn("r1", [MARK_CROP], "", [], ["Boxing the crop rows."])
    llm = FakeLlm([turn, LlmTurn("r2", _close("top_down"), "")])
    result, _ = _run(llm)
    assert (result.steps[0].tool, result.steps[0].summary) == ("think", "Boxing the crop rows.")


def test_marks_and_zooms_save_pictures_through_the_sink():
    saved = []
    llm = FakeLlm([LlmTurn("r1", [MARK_CROP, MARK_WEEDS], ""),
                   LlmTurn("r2", [ToolCall("c3", "zoom", {"instance_id": "s0-i3"})], ""),
                   LlmTurn("r3", [_classify("s0-i3", "s0-i4"), *_close()], "")])
    result, _ = _run(llm, picture_sink=lambda image: saved.append(image.size) or f"s0-{len(saved):02d}.jpg")
    assert [(s.tool, s.picture) for s in result.steps if s.picture] == [
        ("mark", "s0-01.jpg"), ("mark", "s0-02.jpg"), ("zoom", "s0-03.jpg")]
```

- [ ] **Step 2: Run them.** `.venv/Scripts/python -m pytest tests/test_stop_agent.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.** In `scout/agent/stop_agent.py`:

Add these constants below `MAX_CHECK_FLAGS`:

```python
THOUGHT_LINES = 4  # reasoning lines kept per turn
THOUGHT_MAX_CHARS = 200
```

Add `from PIL import Image` to the imports. Give `run_stop` a new keyword parameter, after `clock`:
`picture_sink: Callable[[Image.Image], str] | None = None`. Create the state with
`StopState(image, crop, on_step=on_step, clock=clock, picture_sink=picture_sink)`.

Change `ask` to request reasoning summaries:

```python
    ask = dict(instructions=stop_instructions(crop, weeds.ids(), settings.max_tool_calls), tools=TOOL_SCHEMAS,
               reasoning_summary=True)
```

Right after `history += turn.output_items` in the loop, add:

```python
            thought = (turn.text or "").strip()
            if thought or turn.reasoning:  # what the agent is thinking, for people watching; before the turn's tools
                state.log("think", thought or turn.reasoning[0],
                          detail=[line[:THOUGHT_MAX_CHARS] for line in turn.reasoning[:THOUGHT_LINES]])
```

- [ ] **Step 4: Run the tests.** `tests/test_stop_agent.py`, then the full suite. Expected: all pass. Existing tests use turns with empty text, so their step lists are unchanged.

- [ ] **Step 5: Commit.**
```bash
git add scout/agent/stop_agent.py tests/test_stop_agent.py
git commit -m "feat: each turn logs the agent's thought; stops pass the picture sink on" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pipeline events, step pictures, saved events and richer samples

**Files:** Modify `scout/app/pipeline.py`. Test: `tests/test_pipeline.py`.

- [ ] **Step 1: Update the fake runner and write the failing tests.**

In `tests/test_pipeline.py`, replace `fake_runner` with:

```python
def fake_runner(image, crop, llm, sam, weeds, settings, on_step=None, picture_sink=None):
    state = StopState(image, crop, on_step=on_step, picture_sink=picture_sink)
    state.log("think", "Looking over the rows", detail=["Counting weeds"])
    state.log("mark", "Mark weed: 1 of 1 boxes kept", picture=image.work)
    return StopResult(stop_index=image.index, image_hash=image.hash, species={"waterhemp": 1},
                      species_cover_pct={"waterhemp": 1.0}, weed_cover_pct=1.0, weed_count=1), state
```

Add `import json` to its imports. Then append:

```python
def test_events_tell_the_whole_walk_with_pictures_and_time(tmp_path):
    images = [make_image(index=0), make_image(index=1, color=(10, 10, 10))]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images, names=["a.jpg", "b.jpg"])
    events = walk.events
    assert all("t" in e for e in events)
    first = events[0]
    assert first["type"] == "walk_start" and first["crop"] == "soybean"
    assert first["stops"] == [{"stop": 0, "name": "a.jpg"}, {"stop": 1, "name": "b.jpg"}]
    assert first["budget"] == {"calls": 30, "seconds": 360.0}
    starts = [e for e in events if e["type"] == "stop_start"]
    assert len(starts) == 2 and all(e["picture"].startswith(f"/walks/{walk.id}/steps/s") for e in starts)
    thoughts = [e for e in events if e["type"] == "thought"]
    assert thoughts[0]["text"] == "Looking over the rows" and thoughts[0]["detail"] == ["Counting weeds"]
    marks = [e for e in events if e["type"] == "step"]
    assert len(marks) == 2 and all(e["picture"].endswith(".jpg") and e["tool"] == "mark" for e in marks)
    done = [e for e in events if e["type"] == "stop_done"]
    assert {e["status"] for e in done} == {"ok"} and all(e["cover"] == 1.0 and e["weeds"] == 1 for e in done)
    assert all(e["picture"].endswith(".jpg") for e in done)
    assert [e["type"] for e in events][-2:] == ["report_building", "report_ready"]
    folder = pipeline.find_dir(walk.id)
    assert sorted(p.name for p in (folder / "steps").iterdir()) == [
        "s0-00.jpg", "s0-01.jpg", "s0-02.jpg", "s1-00.jpg", "s1-01.jpg", "s1-02.jpg"]
    saved = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    assert saved == events and pipeline.saved_events(walk.id) == events


def test_step_pictures_are_found_by_valid_name_only(tmp_path):
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    assert pipeline.step_picture(walk.id, "s0-01.jpg").is_file()
    for bad in ("s0-99.jpg", "../report.json", "s0-01.png", "x.jpg"):
        assert pipeline.step_picture(walk.id, bad) is None
    assert pipeline.step_picture("not-a-walk", "s0-01.jpg") is None


def test_samples_carry_a_thumbnail_the_stop_count_and_the_decision(tmp_path):
    images = [make_image(index=0)]
    pipeline = _pipeline(tmp_path, fake_runner)
    walk = Walk(id=walk_id_for("soybean", images), crop="soybean")
    pipeline.run(walk, images)
    demo = tmp_path / "demo"
    demo.mkdir()
    (tmp_path / "store" / walk.id).rename(demo / walk.id)
    (demo / "samples.json").write_text(json.dumps([{"walk_id": walk.id, "name": "Soybean, waterhemp"}]))
    [sample] = pipeline.samples()
    assert sample["name"] == "Soybean, waterhemp" and sample["stops"] == 1
    assert sample["thumbnail"] == f"/walks/{walk.id}/steps/s0-00.jpg"
    assert sample["decision"] == "No action"  # small_weedbook's waterhemp: 1% cover against the 5% tool default
```

Also, in `test_walk_where_every_stop_fails_is_failed_and_not_cached`, add `assert not (tmp_path / "store" / walk.id / "events.json").exists()`.

- [ ] **Step 2: Run them.** `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.** In `scout/app/pipeline.py`:

Add these imports: `import itertools`, `import time`, `from PIL import Image`, `from scout.report.render import OUTCOME_TEXT`.

Add these constants below `_WALK_ID`:

```python
STEP_NAME = re.compile(r"s\d{1,2}-\d{2,3}\.jpg")  # step picture file names; the route serves nothing else
STEP_MAX_PX = 1024
STEP_QUALITY = 80
```

`Walk` gains a start time and stamps every event:

```python
@dataclass
class Walk:
    id: str
    crop: str
    status: str = "running"  # running | done | failed
    events: list[dict] = field(default_factory=list)
    report: FieldReport | None = None
    error: str | None = None
    started: float = field(default_factory=time.monotonic)

    def emit(self, event: dict) -> None:
        self.events.append({**event, "t": round(time.monotonic() - self.started, 1)})
```

Add a module helper below `walk_id_for`:

```python
def _write_atomic(path: Path, text: str) -> None:
    """find_dir and replays treat these files as complete, so they appear whole or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
```

Add these methods to `Pipeline`, after `cached`:

```python
    def saved_events(self, walk_id: str) -> list[dict] | None:
        folder = self.find_dir(walk_id)
        path = folder / "events.json" if folder else None
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def step_picture(self, walk_id: str, name: str) -> Path | None:
        """A running walk's pictures are in the store before its report exists, so this doesn't use find_dir."""
        if not _WALK_ID.fullmatch(walk_id) or not STEP_NAME.fullmatch(name):
            return None
        for base in (self.store_dir, self.demo_dir):
            path = base / walk_id / "steps" / name
            if path.is_file():
                return path
        return None
```

Replace `samples` with:

```python
    def samples(self) -> list[dict]:
        index = self.demo_dir / "samples.json"
        if not index.exists():
            return []
        out = []
        for sample in json.loads(index.read_text(encoding="utf-8")):
            report = self.cached(sample["walk_id"])
            events = self.saved_events(sample["walk_id"]) or []
            thumbnail = next((e.get("picture") for e in events if e.get("type") == "stop_start"), None)
            decision = None
            if report is not None:
                decision = OUTCOME_TEXT[report.decision] + (f" ({report.treat_mode})" if report.treat_mode else "")
            out.append({**sample, "thumbnail": thumbnail, "decision": decision,
                        "stops": len(report.stops) if report else None})
        return out
```

Replace `run` with:

```python
    def run(self, walk: Walk, images: list[StopImage], names: list[str] | None = None) -> None:
        """Events are emitted before status changes, so a reader that sees a final status has every event."""
        try:
            report = self.cached(walk.id)
            fresh = report is None
            if fresh:
                out_dir = self.store_dir / walk.id
                (out_dir / "steps").mkdir(parents=True, exist_ok=True)
                names = names or [f"Photo {image.index + 1}" for image in images]
                walk.emit({"type": "walk_start", "crop": walk.crop,
                           "budget": {"calls": self.settings.max_tool_calls, "seconds": self.settings.stop_timeout_s},
                           "stops": [{"stop": image.index, "name": name} for image, name in zip(images, names)]})
                with ThreadPoolExecutor(max_workers=self.settings.max_concurrent_stops) as pool:
                    stops = list(pool.map(lambda image: self._stop(walk, image, out_dir), images))
                if all(stop.status == "failed" for stop in stops):  # an outage or a bad key: don't cache it
                    raise RuntimeError("no stop could be analyzed")
                walk.emit({"type": "report_building"})
                report = build_report(walk.id, walk.crop, stops, self.weeds, self.records, self.llm)
                _write_atomic(out_dir / "report.json", report.model_dump_json(indent=1))
            walk.report = report
            walk.emit({"type": "report_ready"})
            if fresh:  # the saved feed is what a replay plays back
                _write_atomic(self.store_dir / walk.id / "events.json", json.dumps(walk.events))
            walk.status = "done"
        except Exception as error:
            log.exception("walk %s failed", walk.id)
            walk.error = str(error)
            walk.emit({"type": "error", "message": "The walk couldn't be analyzed. Try again."})
            walk.status = "failed"
```

Replace `_stop` with:

```python
    def _stop(self, walk: Walk, image: StopImage, out_dir: Path) -> StopResult:
        numbers = itertools.count()

        def save(picture: Image.Image) -> str:
            name = f"s{image.index}-{next(numbers):02d}.jpg"
            copy = picture.convert("RGB")
            copy.thumbnail((STEP_MAX_PX, STEP_MAX_PX))
            copy.save(out_dir / "steps" / name, format="JPEG", quality=STEP_QUALITY)
            return name

        def url(name: str | None) -> str | None:
            return f"/walks/{walk.id}/steps/{name}" if name else None

        def try_save(picture: Image.Image) -> str | None:
            try:
                return save(picture)
            except Exception:  # pictures are for people watching; a failed save must not end the stop
                log.warning("step picture could not be saved", exc_info=True)
                return None

        def on_step(step):
            if step.tool == "think":
                walk.emit({"type": "thought", "stop": image.index, "text": step.summary, "detail": step.detail})
            else:
                walk.emit({"type": "step", "stop": image.index, "tool": step.tool, "summary": step.summary,
                           "ok": step.ok, "picture": url(step.picture)})

        walk.emit({"type": "stop_start", "stop": image.index, "picture": url(try_save(image.work))})
        final = None
        try:
            result, state = self._stop_runner(image, walk.crop, self.llm, self.sam, self.weeds, self.settings,
                                              on_step=on_step, picture_sink=save)
            overlay = state.overlay()
            overlay.save(out_dir / f"stop_{image.index}.png")
            final = url(try_save(overlay))
        except Exception:
            log.exception("stop %s of walk %s failed", image.index, walk.id)
            result = StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, status="failed",
                                flags=["This stop could not be analyzed"])
            image.work.save(out_dir / f"stop_{image.index}.png")
        walk.emit({"type": "stop_done", "stop": image.index, "status": result.status,
                   "cover": result.weed_cover_pct, "weeds": result.weed_count, "picture": final})
        return result
```

The existing `test_run_writes_report_and_overlays` still asserts `kinds.count("stop_done") == 2 and kinds.count("step") == 2`. With the new fake runner, each stop logs one "mark" step (a `step` event) and one "think" step (a `thought` event), so it still holds.

- [ ] **Step 4: Run the tests.** `tests/test_pipeline.py`, then the full suite. Expected: all pass. If `tests/test_main.py` fails because `FakePipeline.run` doesn't accept `names`, leave it: Task 6 updates it. Report it instead.

- [ ] **Step 5: Commit.**
```bash
git add scout/app/pipeline.py tests/test_pipeline.py
git commit -m "feat: walks emit pictures, thoughts and time, and save their events for replay" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The app passes photo names, serves step pictures and replays saved walks

**Files:** Modify `scout/app/main.py`. Test: `tests/test_main.py`.

- [ ] **Step 1: Update the fakes and write the failing tests.**

In `tests/test_main.py`:
- Add the imports `import json` and `from scout.app import main as main_module`.
- Add `from scout.app.pipeline import STEP_NAME`.
- In `FakePipeline`, change `run(self, walk, images)` to `run(self, walk, images, names=None)`, store `self.names = names`, and add:

```python
    def saved_events(self, walk_id):
        folder = self.find_dir(walk_id)
        path = folder / "events.json" if folder else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def step_picture(self, walk_id, name):
        path = self.root / walk_id / "steps" / name
        return path if STEP_NAME.fullmatch(name) and path.is_file() else None
```

- `GpsCapture.run(self, walk, images)` becomes `run(self, walk, images, names=None)` and calls `super().run(walk, images, names)`.

Append these tests:

```python
def _saved_walk(root, walk_id, events):
    folder = root / walk_id
    (folder / "steps").mkdir(parents=True)
    report = FieldReport(walk_id=walk_id, crop="corn", stops=[], species_summary=[], decision="no_action",
                         rule_trace=[], explanation="x", options=[])
    (folder / "report.json").write_text(report.model_dump_json())
    if events is not None:
        (folder / "events.json").write_text(json.dumps(events))
    return folder


def _event_data(text):
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_upload_passes_photo_names_to_the_pipeline(tmp_path):
    pipeline = FakePipeline(tmp_path)
    client = _client(tmp_path, pipeline)
    walk_id = _upload(client, n=2).json()["walk_id"]
    client.get(f"/walks/{walk_id}/events")
    assert pipeline.names == ["p0.png", "p1.png"]


def test_step_pictures_are_served_by_valid_name_only(tmp_path):
    folder = _saved_walk(tmp_path, "0123456789abcdef", None)
    (folder / "steps" / "s0-01.jpg").write_bytes(b"jpeg")
    client = _client(tmp_path)
    ok = client.get("/walks/0123456789abcdef/steps/s0-01.jpg")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg" and ok.content == b"jpeg"
    for bad in ("s0-02.jpg", "report.json", "..%2Freport.json"):
        assert client.get(f"/walks/0123456789abcdef/steps/{bad}").status_code == 404
    assert client.get("/walks/ffffffffffffffff/steps/s0-01.jpg").status_code == 404


def test_a_saved_walk_replays_its_events(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "REPLAY_SPEED", 1e9)
    events = [{"type": "walk_start", "crop": "corn", "stops": [{"stop": 0, "name": "a.jpg"}], "t": 0.0},
              {"type": "stop_start", "stop": 0, "picture": None, "t": 0.5},
              {"type": "report_ready", "t": 40.0}]
    _saved_walk(tmp_path, "0123456789abcdef", events)
    replayed = _event_data(_client(tmp_path).get("/walks/0123456789abcdef/events").text)
    assert [e["type"] for e in replayed] == ["walk_start", "stop_start", "report_ready"]
    assert all(e["replay"] is True for e in replayed)


def test_a_saved_walk_without_events_just_reports_ready(tmp_path):
    _saved_walk(tmp_path, "0123456789abcdef", None)
    replayed = _event_data(_client(tmp_path).get("/walks/0123456789abcdef/events").text)
    assert replayed == [{"type": "report_ready"}]
```

- [ ] **Step 2: Run them.** `.venv/Scripts/python -m pytest tests/test_main.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** In `scout/app/main.py`:

Add these constants below `KEEP_ALIVE_S`:

```python
REPLAY_SPEED = 4.0  # a saved walk replays this many times faster than it ran
REPLAY_MAX_GAP_S = 3.0  # and never waits longer than this between two events
```

In `create_walk`, build the names before starting the thread, and pass them:

```python
        names = [upload.filename or f"Photo {index + 1}" for index, upload in enumerate(files)]
```

```python
        threading.Thread(target=pipeline.run, args=(walk, images), kwargs={"names": names}, daemon=True).start()
```

Replace the start of `events` (up to `async def stream():`) with:

```python
    @app.get("/walks/{walk_id}/events")
    async def events(walk_id: str, request: Request):
        walk = walks.get(walk_id)
        replaying = False
        if walk is None:
            if pipeline.cached(walk_id) is None:
                raise HTTPException(404, "Unknown walk.")
            saved = pipeline.saved_events(walk_id)
            replaying = bool(saved)
            events = [{**event, "replay": True} for event in saved] if saved else [{"type": "report_ready"}]
            walk = Walk(id=walk_id, crop="", status="done", events=events)
        last = request.headers.get("last-event-id", "")
        start = int(last) + 1 if last.isdigit() else 0  # EventSource sends it when it reconnects: resume, don't replay
```

Inside `stream()`, before each `yield f"id: ..."`, pace a replay:

```python
                while sent < len(walk.events):
                    if replaying and sent > start:  # a saved walk plays back at REPLAY_SPEED, gaps capped
                        gap = walk.events[sent].get("t", 0) - walk.events[sent - 1].get("t", 0)
                        await asyncio.sleep(min(max(gap, 0) / REPLAY_SPEED, REPLAY_MAX_GAP_S))
                    yield f"id: {sent}\ndata: {json.dumps(walk.events[sent])}\n\n"
```

Add the route below `overlay`:

```python
    @app.get("/walks/{walk_id}/steps/{name}")
    def step_picture(walk_id: str, name: str):
        path = pipeline.step_picture(walk_id, name)
        if path is None:
            raise HTTPException(404, "No such picture.")
        return FileResponse(path, media_type="image/jpeg")
```

Leave `report_page` unchanged here. Task 7 changes it together with `report_html`'s new parameter.

- [ ] **Step 4: Run the tests.** `tests/test_main.py`, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit.**
```bash
git add scout/app/main.py tests/test_main.py
git commit -m "feat: serve step pictures and replay saved walks at 4x" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The report in the field notebook look, with the agent's timeline

**Files:** Modify `scout/report/render.py`, `scout/report/templates/report.html` and `scout/app/main.py` (`report_page` only). Test: `tests/test_render.py`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_render.py` (and add `AgentStep` to its `scout.models` import):

```python
def test_decision_card_rules_and_the_agents_timeline():
    report = _report()
    report.stops[0].steps = [
        AgentStep(t_ms=10, tool="think", summary="Scouting the rows.", detail=["Counting weeds"]),
        AgentStep(t_ms=20, tool="mark", summary="Mark weed: 2 of 2 boxes kept", picture="s0-01.jpg")]
    html = report_html(report, small_weedbook(), lambda i: f"/o/{i}.png", lambda name: f"/walks/w1/steps/{name}")
    assert 'class="decision treat"' in html and "How the rules decided" in html and "How the agent worked" in html
    assert "Scouting the rows." in html and "Mark weed: 2 of 2 boxes kept" in html
    assert 'src="/walks/w1/steps/s0-01.jpg"' in html


def test_reports_without_step_pictures_still_render():
    report = _report()
    report.stops[0].steps = [AgentStep(t_ms=1, tool="mark", summary="Mark weed: 1 of 1 boxes kept")]
    html = _html(report)
    assert "Mark weed: 1 of 1 boxes kept" in html and "/steps/" not in html
```

- [ ] **Step 2: Run them.** `.venv/Scripts/python -m pytest tests/test_render.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.**

In `scout/report/render.py`, change `report_html` to:

```python
def report_html(report: FieldReport, weeds: WeedBook, overlay_url: Callable[[int], str],
                step_url: Callable[[str], str] | None = None) -> str:
```

In the `render(...)` call, add `step_url=step_url or (lambda name: None),`.

In `scout/app/main.py`, change `report_page` to pass a step-picture URL maker:

```python
    @app.get("/walks/{walk_id}/report", response_class=HTMLResponse)
    def report_page(walk_id: str):
        return report_html(report_for(walk_id), pipeline.weeds, lambda i: f"/walks/{walk_id}/overlay/{i}.png",
                           lambda name: f"/walks/{walk_id}/steps/{name}")
```

Replace `scout/report/templates/report.html` with:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scouting report: {{ report.crop }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Domine:wght@400..700&family=Public+Sans:ital,wght@0,400..700;1,400&display=swap">
<style>
  :root { --page: #F5F2E8; --card: #FBFAF5; --line: #DDD8C8; --ink: #1D2A1F; --text: #3F4A3C; --muted: #5F675A;
          --accent: #B0127F; --green: #4E7A1E; --serif: "Domine", Georgia, serif; --sans: "Public Sans", system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--text); font: 16px/1.55 var(--sans); }
  main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 48px; }
  h1, h2, h3 { font-family: var(--serif); color: var(--ink); line-height: 1.2; margin: 28px 0 10px; }
  h1 { font-size: clamp(26px, 4vw, 36px); margin-top: 8px; } h2 { font-size: 22px; } h3 { font-size: 17px; margin: 8px 0 4px; }
  a { color: var(--accent); }
  .kicker { font-size: 13px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent); font-weight: 600; margin: 0; }
  .nav { display: flex; gap: 16px; font-size: 14px; font-weight: 600; margin-bottom: 12px; }
  .warning { background: #fdecc8; border: 1px solid #e0a43a; border-radius: 10px; padding: 12px; font-weight: 600; color: #633806; }
  .decision { border-radius: 14px; padding: 18px 20px; margin: 16px 0; border: 1px solid var(--line); }
  .decision .label { font-size: 13px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
  .decision b { display: block; font-family: var(--serif); font-size: clamp(26px, 4vw, 34px); color: var(--ink); }
  .decision .urgency { font-weight: 600; }
  .treat { background: #FAEEDA; } .treat .label { color: #854F0B; }
  .scout_again { background: #E6F1FB; } .scout_again .label { color: #0C447C; }
  .no_action { background: #EAF3DE; } .no_action .label { color: #27500A; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
  .metric { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; font-size: 14px; color: var(--muted); }
  .metric b { display: block; font-size: 24px; color: var(--ink); }
  .why { font-family: var(--serif); font-size: 18px; color: var(--ink); max-width: 760px; }
  details { margin: 8px 0; } summary { cursor: pointer; font-weight: 600; color: var(--ink); }
  .table-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--line); border-radius: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  td, th { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; } tr:last-child td { border-bottom: 0; }
  .flag { display: inline-block; background: #FAEEDA; color: #633806; border-radius: 6px; padding: 2px 8px; font-size: 13px; margin: 3px 4px 0 0; }
  .muted { color: var(--muted); font-size: 13px; }
  .legend { font-size: 14px; }
  .stops { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .stop { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
  .stop > img { width: 100%; border-radius: 10px; display: block; }
  .timeline { list-style: none; padding: 0; margin: 8px 0 0; font-size: 13px; display: grid; gap: 6px; }
  .timeline .think { font-family: var(--serif); font-style: italic; color: var(--ink); }
  .timeline .tool { font-weight: 600; color: var(--ink); }
  .timeline .detail { color: var(--muted); }
  .timeline img { width: 64px; height: 48px; object-fit: cover; border-radius: 6px; vertical-align: middle; margin-left: 6px; }
  .timeline .failed { color: #A32D2D; }
  @media print { .noprint, .timeline img { display: none; } details > * { display: block; } .warning { -webkit-print-color-adjust: exact; print-color-adjust: exact; } .decision, .flag { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
</style>
</head>
<body>
<main>
<p class="nav noprint"><a href="/">New walk</a><a href="stops.csv">Download CSV</a><a href="#" onclick="window.print(); return false;">Save as PDF</a></p>
<p class="kicker">Weed Scout · scouting report</p>
<h1>{{ report.crop|capitalize }} field, {{ report.stops|length }} stops</h1>
{% if report.crop_mismatch %}<p class="warning">{{ report.crop_mismatch }}</p>{% endif %}
<div class="decision {{ report.decision }}">
  <span class="label">Decision</span>
  <b>{{ decision_text }}</b>
  {% if report.urgency %}<span class="urgency">{{ report.urgency }}</span>{% endif %}
</div>
<div class="metrics">
  <div class="metric">Mean weed cover<b>{{ "%.1f"|format(mean_cover) }}%</b></div>
  <div class="metric">Density<b>{{ density_text }}</b></div>
  <div class="metric">Stops with weeds<b>{{ stops_with_weeds }} of {{ ok_count }}</b></div>
</div>

<h2>Why</h2>
<p class="why">{{ report.explanation }}</p>
<details><summary>How the rules decided</summary><ul>{% for line in report.rule_trace %}<li>{{ line }}</li>{% endfor %}</ul></details>

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
  <tr><th>For</th><th>Product (EPA reg. no.)</th><th>Group</th><th>Limits</th><th>Rate</th><th>Pre-harvest interval</th><th>Source</th></tr>
  {% for o in report.options %}{% set r = o.record %}
  <tr>
    <td>{{ names.get(r.species_id, r.species_id) }}</td>
    <td>{{ r.product_name }} ({{ r.reg_no }}){% if r.requires_trait %}<div class="flag">Requires {{ r.requires_trait }} seed</div>{% endif %}</td>
    <td>{{ r.herbicide_groups|join(", ") }}</td>
    <td>{% if r.max_weed_height_in is not none %}Weeds up to {{ r.max_weed_height_in }} in{% endif %}{% if r.crop_stage_max %} · crop up to {{ r.crop_stage_max }}{% endif %}{% if o.over_size_limit %}<div class="flag">Weeds are over this size limit</div>{% endif %}</td>
    <td>{% if r.rate_min is not none %}{{ r.rate_min }}{% if r.rate_max is not none and r.rate_max != r.rate_min %}–{{ r.rate_max }}{% endif %} {{ r.rate_unit or "" }}{% else %}—{% endif %}</td>
    <td>{{ (r.phi_days|string + " days") if r.phi_days is not none else "—" }}</td>
    <td>{% if r.pdf_file %}<a href="{{ pdf_base }}{{ r.pdf_file }}#page={{ r.page }}">p. {{ r.page }}</a>{% else %}p. {{ r.page }}{% endif %}<div class="muted">“{{ r.quote }}”</div></td>
  </tr>
  {% endfor %}
</table></div>
{% else %}<p>No product in the current label set matches.</p>{% endif %}
<p class="muted">Read the label before use. The label is the law. This report is decision support, not a prescription.</p>

{% set pins = report.stops | selectattr("gps") | list %}
{% if pins %}
<h2>Stops on the map</h2>
<div id="map" style="height: 280px; border-radius: 12px;"></div>
<p class="muted">Pins are shifted for privacy.</p>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
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
<p class="legend">Green: crop · Magenta: weeds · Gray: in-row zone · Blue box: weed area measured by green pixels · Yellow box: check this plant</p>
<div class="stops">
{% for s in report.stops %}
  <div class="stop">
    <img src="{{ overlay_url(s.stop_index) }}" alt="Stop {{ s.stop_index + 1 }}: crop in green, weeds in magenta">
    <h3>Stop {{ s.stop_index + 1 }}</h3>
    {% if s.status == "failed" %}<p class="muted">Not analyzed; not used in the field numbers.</p>{% else %}
    <span class="muted">{{ s.view_type.replace("_", " ") }}{% if s.crop_stage %} · crop {{ s.crop_stage }}{% endif %} · {{ s.status.replace("_", " ") }}</span>
    <p>Weed cover {{ s.weed_cover_pct }}% · {{ s.weed_count }} weeds{% if s.count_approximate %} (approx.){% endif %}{% if s.density_per_m2 is not none %} · {{ s.density_per_m2 }}/m²{% endif %}</p>{% endif %}
    {% for f in s.flags %}<div class="flag">{{ f }}</div>{% endfor %}
    {% if s.steps %}
    <details><summary>How the agent worked</summary>
      <ol class="timeline">
      {% for step in s.steps %}
        {% if step.tool == "think" %}
        <li class="think">“{{ step.summary }}”{% for line in step.detail %}<div class="detail">{{ line }}</div>{% endfor %}</li>
        {% else %}
        <li{% if not step.ok %} class="failed"{% endif %}><span class="tool">{{ step.tool }}</span> {{ step.summary }}{% if step.picture and step_url(step.picture) %}<img src="{{ step_url(step.picture) }}" alt="" loading="lazy">{% endif %}</li>
        {% endif %}
      {% endfor %}
      </ol>
    </details>
    {% endif %}
  </div>
{% endfor %}
</div>
</main>
</body>
</html>
```

Strings the existing tests check, all kept:
- "Treat (broadcast)", "Product A (100-1)", the overlay URL, "Read the label before use", the PDF `#page=` link, "Check s0-i6: low confidence";
- the `<td>waterhemp</td>\s*<td>Product A (100-1)` pattern;
- "Requires ... seed</div>" only with a trait;
- the leaflet tags twice with integrity, "Pins are shifted for privacy.";
- "Mean weed cover<b>2.0%</b>", "Not analyzed; not used in the field numbers.";
- the escaping, the banner before "Decision", and the blue legend;
- the one-line `@media print` rule containing `.warning { ... print-color-adjust: exact; }`.

The word "Decision" appears only in the decision card, after the banner.

- [ ] **Step 4: Run the tests.** `tests/test_render.py`, then the full suite. Expected: all pass.

- [ ] **Step 5: Commit.**
```bash
git add scout/report/render.py scout/report/templates/report.html scout/app/main.py tests/test_render.py
git commit -m "feat: field-notebook report with a decision card and the agent's timeline" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Upload view and live view

**Files:** Rewrite `scout/web/index.html`, `scout/web/style.css` and `scout/web/app.js`. Test: `tests/test_main.py`.

- [ ] **Step 1: Write the failing smoke test.** In `tests/test_main.py`, replace `test_crop_menu_makes_the_farmer_choose` with:

```python
def test_page_has_crop_cards_drop_zone_and_live_view(tmp_path):
    html = _client(tmp_path).get("/").text
    for crop in ("corn", "soybean", "alfalfa"):
        assert f'<input type="radio" name="crop" value="{crop}" required>' in html
    assert "checked" not in html
    for part in ('id="drop-zone"', 'id="live-view"', 'id="strip"', 'id="stage-img"', 'aria-live="polite"',
                 'id="checklist"', "Scout this field"):
        assert part in html
```

- [ ] **Step 2: Run it.** `.venv/Scripts/python -m pytest tests/test_main.py -v`. Expected: FAIL.

- [ ] **Step 3: Write `scout/web/index.html`:**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weed Scout</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Domine:wght@400..700&family=Public+Sans:ital,wght@0,400..700;1,400&display=swap">
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main id="upload-view">
  <header class="hero">
    <p class="kicker">Weed Scout</p>
    <h1>Walk the field. Upload the photos. Get a spray call you can check.</h1>
    <p class="lede">An agent finds and measures every weed, checks it against extension thresholds, and lists EPA label options with page citations.</p>
  </header>
  <div class="columns">
    <form id="walk-form" class="card" novalidate>
      <fieldset>
        <legend>1 · Which crop?</legend>
        <div class="crop-cards">
          <label class="crop-card"><input type="radio" name="crop" value="corn" required><span>Corn</span></label>
          <label class="crop-card"><input type="radio" name="crop" value="soybean" required><span>Soybean</span></label>
          <label class="crop-card"><input type="radio" name="crop" value="alfalfa" required><span>Alfalfa</span></label>
        </div>
      </fieldset>
      <div>
        <p class="field-label" id="photos-label">2 · Photos from one walk (1 to 10)</p>
        <label class="drop-zone" id="drop-zone">
          <input type="file" id="files" accept="image/*,.heic,.heif" multiple aria-labelledby="photos-label">
          <span id="drop-text">Drop photos here or tap to choose</span>
        </label>
        <div id="thumbs" class="thumbs"></div>
      </div>
      <button type="submit" class="primary" id="scout-button">Scout this field</button>
      <p id="form-error" class="error" role="alert" hidden></p>
    </form>
    <section id="samples" class="samples" hidden>
      <h2>Or watch a sample walk</h2>
      <div id="sample-list"></div>
    </section>
  </div>
  <footer class="footnote">Pins are shifted for privacy. Decision support, not a prescription.</footer>
</main>

<main id="live-view" hidden>
  <header class="live-header">
    <div>
      <p class="kicker">Weed Scout</p>
      <h1 id="live-title">Scouting the field</h1>
      <p class="muted" id="live-progress-text">Starting…</p>
    </div>
    <div class="live-meta">
      <span id="replay-badge" class="badge" hidden>Replay, 4× speed</span>
      <a id="skip-link" href="#" hidden>Skip to report</a>
      <span id="elapsed" class="muted">0:00</span>
    </div>
  </header>
  <div class="progress"><div id="progress-bar"></div></div>
  <div class="strip-row">
    <div id="strip" class="strip"></div>
    <button type="button" id="follow-live" class="ghost" hidden>Follow live</button>
  </div>
  <section class="stage">
    <figure class="stage-picture">
      <div class="picture-frame"><img id="stage-img" alt=""></div>
      <figcaption id="stage-caption" class="muted"></figcaption>
    </figure>
    <aside class="stage-side">
      <ol id="checklist" class="checklist"></ol>
      <p id="stage-counts" class="counts"></p>
    </aside>
    <div class="thought" aria-live="polite">
      <p id="thought-text"></p>
      <ul id="thought-detail"></ul>
    </div>
  </section>
  <div id="live-end" class="live-end" hidden>
    <a id="report-link" class="primary button" href="#">See the report</a>
  </div>
  <p id="live-status" class="muted"></p>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `scout/web/style.css`:**

```css
:root {
  --page: #F5F2E8; --card: #FBFAF5; --line: #DDD8C8; --ink: #1D2A1F; --text: #3F4A3C; --muted: #5F675A;
  --accent: #B0127F; --accent-soft: #F4C0D1; --green: #4E7A1E; --green-bg: #EAF3DE; --red: #A32D2D;
  --serif: "Domine", Georgia, serif; --sans: "Public Sans", system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text); font: 16px/1.55 var(--sans); }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 48px; }
main[hidden] { display: none; }
h1, h2, h3 { font-family: var(--serif); color: var(--ink); font-weight: 700; line-height: 1.2; margin: 0 0 8px; }
h1 { font-size: clamp(26px, 4vw, 36px); }
h2 { font-size: 20px; }
a { color: var(--accent); font-weight: 600; }
.kicker { font-size: 13px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent); font-weight: 600; margin: 0 0 6px; }
.lede, .muted { color: var(--muted); }
.lede { max-width: 660px; margin: 0 0 24px; }
.columns { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 24px; align-items: start; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }
form.card { display: grid; gap: 20px; }
fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
legend, .field-label { font-weight: 600; color: var(--ink); margin: 0 0 8px; padding: 0; }
.crop-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.crop-card { position: relative; border: 1px solid var(--line); border-radius: 10px; padding: 14px 8px; text-align: center;
             cursor: pointer; background: #fff; font-weight: 600; color: var(--ink); }
.crop-card input { position: absolute; inset: 0; opacity: 0; margin: 0; cursor: pointer; }
.crop-card:has(input:checked) { border: 2px solid var(--green); background: var(--green-bg); color: #27500A; }
.crop-card:has(input:focus-visible) { outline: 3px solid var(--accent); outline-offset: 2px; }
.drop-zone { position: relative; display: grid; place-items: center; min-height: 112px; border: 2px dashed #B4B2A9;
             border-radius: 10px; padding: 16px; text-align: center; color: var(--muted); background: #fff; cursor: pointer; }
.drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-zone.dragging { border-color: var(--accent); background: #FBEAF0; }
.drop-zone:focus-within { outline: 3px solid var(--accent); outline-offset: 2px; }
.thumbs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; align-items: center; font-size: 13px; color: var(--muted); }
.thumbs img { width: 52px; height: 52px; object-fit: cover; border-radius: 6px; }
button, .button { font: inherit; font-weight: 600; border-radius: 10px; padding: 12px 18px; cursor: pointer;
                  border: 1px solid var(--ink); text-decoration: none; display: inline-block; text-align: center; }
.primary { background: var(--ink); color: var(--page); }
.primary:disabled { opacity: .6; cursor: progress; }
.ghost { background: transparent; color: var(--ink); border-color: var(--line); padding: 6px 12px; font-size: 14px; }
.error { color: var(--red); margin: 0; }
.sample { display: grid; grid-template-columns: 88px minmax(0, 1fr); gap: 12px; align-items: center; background: var(--card);
          border: 1px solid var(--line); border-radius: 12px; padding: 10px; margin-bottom: 10px; }
.sample img { width: 88px; height: 66px; object-fit: cover; border-radius: 8px; background: var(--line); }
.sample .title { color: var(--ink); font-weight: 600; }
.sample .meta { font-size: 13px; color: var(--muted); }
.sample .links a { font-size: 14px; margin-right: 14px; }
.footnote { margin-top: 32px; font-size: 13px; color: var(--muted); }
.live-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap; }
.live-header p { margin: 0; }
.live-meta { display: flex; gap: 14px; align-items: center; font-size: 14px; }
.badge { background: #FBEAF0; color: #72243E; border-radius: 999px; padding: 3px 10px; font-size: 13px; font-weight: 600; }
.progress { height: 6px; border-radius: 3px; background: var(--line); margin: 14px 0 16px; overflow: hidden; }
#progress-bar { height: 100%; width: 0; background: var(--green); transition: width .4s; }
.strip-row { display: flex; gap: 10px; align-items: center; margin-bottom: 16px; }
.strip { display: flex; gap: 10px; overflow-x: auto; padding: 4px; flex: 1; }
.strip button { position: relative; flex: 0 0 auto; width: 80px; height: 60px; padding: 0; border-radius: 8px; border: 0;
                background: var(--line) center / cover no-repeat; outline: 2px solid transparent; outline-offset: 1px; }
.strip button[data-status="working"] { outline-color: var(--accent); animation: pulse 1.4s ease-in-out infinite; }
.strip button[data-status="done"] { outline-color: var(--green); }
.strip button[data-status="failed"] { outline-color: var(--red); }
.strip button[aria-current="true"] { outline-width: 4px; }
.strip .mark { position: absolute; right: 5px; bottom: 2px; font-size: 14px; color: #fff; text-shadow: 0 0 3px #000; }
@keyframes pulse { 50% { outline-color: var(--accent-soft); } }
.stage { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); grid-template-areas: "picture side" "thought side";
         gap: 14px 20px; align-items: start; }
.stage-picture { grid-area: picture; margin: 0; }
.picture-frame { background: var(--ink); border-radius: 12px; overflow: hidden; aspect-ratio: 4 / 3; display: grid; place-items: center; }
.picture-frame img { max-width: 100%; max-height: 100%; object-fit: contain; transition: opacity .25s; }
.picture-frame img.fading { opacity: .25; }
figcaption { font-size: 13px; margin-top: 6px; }
figcaption.problem { color: var(--red); }
.stage-side { grid-area: side; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.checklist { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; font-size: 14px; }
.checklist li { display: grid; grid-template-columns: 20px minmax(0, 1fr); column-gap: 4px; color: #888780; }
.checklist li::before { content: "○"; }
.checklist li.current { color: var(--accent); }
.checklist li.current::before { content: "●"; }
.checklist li.done { color: var(--green); }
.checklist li.done::before { content: "✓"; }
.checklist li.failed { color: var(--red); }
.checklist li.failed::before { content: "!"; }
.checklist li.skipped::before { content: "–"; }
.checklist .step-name { font-weight: 600; }
.checklist .step-summary { grid-column: 2; color: var(--muted); font-size: 13px; }
.counts { border-top: 1px solid var(--line); margin: 12px 0 0; padding-top: 10px; font-size: 13px; color: var(--text); }
.counts:empty { display: none; }
.thought { grid-area: thought; }
#thought-text { font-family: var(--serif); font-style: italic; font-size: 18px; color: var(--ink); margin: 0 0 6px; }
#thought-detail { margin: 0; padding-left: 18px; font-size: 13px; color: var(--muted); }
.live-end { margin-top: 20px; }
@media (prefers-reduced-motion: reduce) {
  .strip button { animation: none !important; }
  #progress-bar, .picture-frame img { transition: none; }
}
@media (max-width: 760px) {
  .columns, .stage { grid-template-columns: minmax(0, 1fr); }
  .stage { grid-template-areas: "picture" "thought" "side"; }
}
```

- [ ] **Step 5: Write `scout/web/app.js`:**

```javascript
"use strict";
const $ = (id) => document.getElementById(id);
const MAX_PHOTOS = 10;
const MAX_BYTES = 25_000_000;
const PHASES = [["survey", "Survey"], ["crop", "Crop"], ["weeds", "Weeds"], ["check", "Check the picture"],
                ["zoom", "Zoom"], ["name", "Name the weeds"], ["measure", "Measure"]];
const TOOL_STEPS = new Set(["survey", "segment", "mark", "zoom", "classify", "measure", "finish"]);
const WALK_ID = /^[0-9a-f]{16}$/;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clock(seconds) {
  const s = Math.max(Math.round(seconds), 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// Upload view
const form = $("walk-form");
const filesInput = $("files");
const dropZone = $("drop-zone");
const scoutButton = $("scout-button");
let chosen = [];

function showError(message) {
  $("form-error").textContent = message || "";
  $("form-error").hidden = !message;
}

function setFiles(files) {
  chosen = Array.from(files).filter((f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name));
  const thumbs = $("thumbs");
  thumbs.replaceChildren();
  let bytes = 0;
  for (const file of chosen) {
    bytes += file.size;
    if (thumbs.children.length < MAX_PHOTOS && file.type.startsWith("image/") && !/hei[cf]/i.test(file.type)) {
      const img = el("img");
      img.alt = "";
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      thumbs.append(img);
    }
  }
  if (chosen.length) thumbs.append(el("span", null, `${chosen.length} photo${chosen.length === 1 ? "" : "s"} · ${(bytes / 1e6).toFixed(1)} MB`));
  $("drop-text").textContent = chosen.length ? "Change photos" : "Drop photos here or tap to choose";
  showError("");
}

filesInput.addEventListener("change", () => setFiles(filesInput.files));
for (const type of ["dragenter", "dragover"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
}
for (const type of ["dragleave", "drop"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); });
}
dropZone.addEventListener("drop", (event) => setFiles(event.dataTransfer.files));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const crop = form.elements.crop.value;
  if (!crop) return showError("Choose the crop first.");
  if (!chosen.length) return showError("Add 1 to 10 photos from one walk.");
  if (chosen.length > MAX_PHOTOS) return showError(`That's ${chosen.length} photos; the limit is ${MAX_PHOTOS}.`);
  const big = chosen.find((f) => f.size > MAX_BYTES);
  if (big) return showError(`${big.name} is over 25 MB.`);
  const body = new FormData();
  body.append("crop", crop);
  for (const file of chosen) body.append("files", file, file.name);
  scoutButton.disabled = true;
  scoutButton.textContent = "Uploading…";
  try {
    const response = await fetch("/walks", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      showError(typeof data.detail === "string" ? data.detail : "Upload failed. Try again.");
      return;
    }
    openLive(data.walk_id, crop, true);
  } catch {
    showError("Couldn't reach the server. Try again.");
  } finally {
    scoutButton.disabled = false;
    scoutButton.textContent = "Scout this field";
  }
});

fetch("/samples").then((r) => r.json()).then((samples) => {
  if (!samples.length) return;
  const list = $("sample-list");
  for (const sample of samples) {
    const card = el("div", "sample");
    const img = el("img");
    img.alt = "";
    if (sample.thumbnail) img.src = sample.thumbnail;
    const body = el("div");
    const meta = [sample.stops ? `${sample.stops} stops` : "", sample.decision || ""].filter(Boolean).join(" · ");
    const links = el("div", "links");
    const watch = el("a", null, "Watch the agent");
    watch.href = `/?walk=${sample.walk_id}`;
    const report = el("a", null, "Report");
    report.href = `/walks/${sample.walk_id}/report`;
    links.append(watch, report);
    body.append(el("div", "title", sample.name), el("div", "meta", meta), links);
    card.append(img, body);
    list.append(card);
  }
  $("samples").hidden = false;
}).catch(() => {});

// Live view
const live = { stops: new Map(), focus: null, pinned: false, total: 0, done: 0, replay: false, budget: null,
               reportUrl: "", timer: null, startedAt: 0, source: null };

function openLive(walkId, crop, push) {
  if (push) history.pushState(null, "", `/?walk=${walkId}`);
  $("upload-view").hidden = true;
  $("live-view").hidden = false;
  $("live-title").textContent = crop ? `Scouting your ${crop} field` : "Scouting the field";
  live.reportUrl = `/walks/${walkId}/report`;
  $("report-link").href = live.reportUrl;
  $("skip-link").href = live.reportUrl;
  live.startedAt = Date.now();
  live.timer = setInterval(() => { if (!live.replay) $("elapsed").textContent = clock((Date.now() - live.startedAt) / 1000); }, 1000);
  live.source = new EventSource(`/walks/${walkId}/events`);
  live.source.onmessage = (message) => handle(JSON.parse(message.data));
  live.source.onerror = () => {
    if (live.source.readyState === EventSource.CLOSED) {
      setStatus("This walk isn't running any more.", true);
    } else {
      setStatus("Reconnecting…");
    }
  };
}

function setStatus(text, offerUpload) {
  const status = $("live-status");
  status.replaceChildren(text ? el("span", null, `${text} `) : "");
  if (offerUpload) {
    const link = el("a", null, "Start a new walk");
    link.href = "/";
    status.append(link);
  }
}

function stopFor(index, name) {
  let stop = live.stops.get(index);
  if (!stop) {
    stop = { index, name: name || `Photo ${index + 1}`, status: "waiting", picture: null, caption: "", thumb: null,
             thought: "", detail: [], problem: "", final: null, startT: null, lastT: 0, calls: 0, estimate: null,
             plants: 0, areas: 0,
             phases: Object.fromEntries(PHASES.map(([key]) => [key, { state: "pending", summary: "" }])) };
    const button = el("button");
    button.type = "button";
    button.append(el("span", "mark"));
    button.addEventListener("click", () => pin(index));
    stop.button = button;
    live.stops.set(index, stop);
    const ordered = [...live.stops.values()].sort((a, b) => a.index - b.index);
    $("strip").replaceChildren(...ordered.map((s) => s.button));
    if (live.focus === null) live.focus = index;
  } else if (name) {
    stop.name = name;
  }
  return stop;
}

function pin(index) {
  live.pinned = true;
  live.focus = index;
  $("follow-live").hidden = false;
  render();
}

$("follow-live").addEventListener("click", () => {
  live.pinned = false;
  $("follow-live").hidden = true;
  render();
});

function touch(stop) {
  if (!live.pinned) live.focus = stop.index;
  render();
}

function phaseOf(stop, event) {
  const summary = event.summary || "";
  const weedsDone = stop.phases.weeds.state === "done";
  switch (event.tool) {
    case "survey": return "survey";
    case "mark":
      if (summary.startsWith("Mark crop")) return weedsDone ? "check" : "crop";
      return weedsDone ? "check" : "weeds";
    case "segment": return summary.includes('"crop row"') ? "crop" : (weedsDone ? "check" : "weeds");
    case "zoom": return "zoom";
    case "classify": return "name";
    case "measure": return "measure";
    default: return null;
  }
}

function applyStep(stop, event) {
  const summary = event.summary || "";
  if (TOOL_STEPS.has(event.tool)) stop.calls += 1;
  if (event.t != null) stop.lastT = event.t;
  const estimate = summary.match(/about (\d+) weeds/);
  if (event.tool === "survey" && estimate) stop.estimate = Number(estimate[1]);
  const weeds = summary.match(/^Mark weed: (\d+) of/);
  if (weeds) stop.plants += Number(weeds[1]);
  const areas = summary.match(/^Mark patch: (\d+) of/);
  if (areas) stop.areas += Number(areas[1]);
  if (event.ok === false) stop.problem = summary;
  const key = phaseOf(stop, event);
  if (!key) return;
  const at = PHASES.findIndex(([k]) => k === key);
  for (const [k] of PHASES) if (stop.phases[k].state === "current") stop.phases[k].state = "pending";
  stop.phases[key] = { state: event.ok === false ? "failed" : "done", summary };
  const next = PHASES.slice(at + 1).find(([k]) => stop.phases[k].state === "pending");
  if (next) stop.phases[next[0]].state = "current";
}

function finishPhases(stop) {
  for (const [k] of PHASES) {
    if (stop.phases[k].state === "pending" || stop.phases[k].state === "current") stop.phases[k].state = "skipped";
  }
}

function countsLine(stop) {
  const parts = [];
  if (stop.estimate != null) parts.push(`about ${stop.estimate} weeds seen`);
  if (stop.plants || stop.areas) {
    let marked = `marked ${stop.plants} plant${stop.plants === 1 ? "" : "s"}`;
    if (stop.areas) marked += ` and ${stop.areas} weed area${stop.areas === 1 ? "" : "s"}`;
    parts.push(marked);
  }
  if (live.budget && stop.status === "working" && stop.startT != null) {
    const calls = Math.max(live.budget.calls - stop.calls, 0);
    parts.push(`${calls} calls and ${clock(live.budget.seconds - (stop.lastT - stop.startT))} left`);
  }
  if (stop.final && stop.final.status !== "failed") parts.push(`cover ${stop.final.cover}% · ${stop.final.weeds} weeds`);
  return parts.join(" · ");
}

function showPicture(url, caption) {
  const img = $("stage-img");
  img.alt = caption;
  if (!url || img.dataset.src === url) return;
  img.dataset.src = url;
  img.classList.add("fading");
  const next = new Image();
  next.onload = () => {
    if (img.dataset.src !== url) return;
    img.src = url;
    img.classList.remove("fading");
  };
  next.src = url;
}

function render() {
  for (const stop of live.stops.values()) {
    const b = stop.button;
    b.dataset.status = stop.status;
    b.setAttribute("aria-label", `${stop.name}, ${stop.status}`);
    b.setAttribute("aria-current", String(stop.index === live.focus));
    if (stop.thumb) b.style.backgroundImage = `url("${stop.thumb}")`;
    b.querySelector(".mark").textContent = stop.status === "done" ? "✓" : stop.status === "failed" ? "!" : "";
  }
  const stop = live.stops.get(live.focus);
  if (!stop) return;
  showPicture(stop.picture, `${stop.name}: ${stop.caption}`);
  const caption = $("stage-caption");
  caption.textContent = `${stop.name} · ${stop.problem || stop.caption || "waiting to start"}`;
  caption.classList.toggle("problem", Boolean(stop.problem) && stop.status === "failed");
  $("thought-text").textContent = stop.thought ? `“${stop.thought}”` : "";
  $("thought-detail").replaceChildren(...stop.detail.slice(0, 3).map((line) => el("li", null, line)));
  $("checklist").replaceChildren(...PHASES.map(([key, label]) => {
    const phase = stop.phases[key];
    const item = el("li", phase.state);
    item.append(el("span", "step-name", label));
    if (phase.summary) item.append(el("span", "step-summary", phase.summary));
    return item;
  }));
  $("stage-counts").textContent = countsLine(stop);
}

function progress() {
  $("progress-bar").style.width = live.total ? `${(live.done / live.total) * 100}%` : "0";
  $("live-progress-text").textContent = `${live.done} of ${live.total} photos done`;
}

function ended() {
  clearInterval(live.timer);
  if (live.source) live.source.close();
  $("live-end").hidden = false;
  $("live-progress-text").textContent = "Report ready";
  setTimeout(() => { if (!live.pinned) location.href = live.reportUrl; }, 3000);
}

function handle(event) {
  setStatus("");
  if (event.replay && !live.replay) {
    live.replay = true;
    $("replay-badge").hidden = false;
    $("skip-link").hidden = false;
  }
  if (live.replay && event.t != null) $("elapsed").textContent = clock(event.t);
  switch (event.type) {
    case "walk_start":
      live.total = event.stops.length;
      live.budget = event.budget || null;
      $("live-title").textContent = `Scouting your ${event.crop} field`;
      for (const s of event.stops) stopFor(s.stop, s.name);
      progress();
      render();
      break;
    case "stop_start": {
      const stop = stopFor(event.stop);
      stop.status = "working";
      stop.startT = event.t;
      stop.lastT = event.t;
      stop.thumb = event.picture;
      stop.picture = event.picture;
      stop.caption = "Photo";
      stop.phases.survey.state = "current";
      touch(stop);
      break;
    }
    case "step": {
      const stop = stopFor(event.stop);
      applyStep(stop, event);
      if (event.picture) {
        stop.picture = event.picture;
        stop.caption = event.tool === "zoom" ? "Close-up" : "What is marked now: green crop, magenta weeds, blue weed areas";
      }
      touch(stop);
      break;
    }
    case "thought": {
      const stop = stopFor(event.stop);
      stop.thought = event.text;
      stop.detail = event.detail || [];
      touch(stop);
      break;
    }
    case "stop_done": {
      const stop = stopFor(event.stop);
      stop.status = event.status === "failed" ? "failed" : "done";
      stop.final = event;
      if (event.picture) {
        stop.picture = event.picture;
        stop.caption = "Final overlay";
      }
      finishPhases(stop);
      live.done += 1;
      progress();
      touch(stop);
      break;
    }
    case "report_building":
      $("live-progress-text").textContent = "All photos done. Building the report…";
      break;
    case "report_ready":
      ended();
      break;
    case "error":
      if (live.source) live.source.close();
      clearInterval(live.timer);
      setStatus(event.message, true);
      break;
  }
}

window.addEventListener("popstate", () => location.reload());
const startWalk = new URLSearchParams(location.search).get("walk");
if (startWalk && WALK_ID.test(startWalk)) openLive(startWalk, "", false);
```

- [ ] **Step 6: Run the tests.** `tests/test_main.py`, then the full suite. Expected: all pass. Check `node --check scout/web/app.js` if `node` is on PATH; if it isn't, skip that check and say so.

- [ ] **Step 7: Commit.**
```bash
git add scout/web/index.html scout/web/style.css scout/web/app.js tests/test_main.py
git commit -m "feat: field-notebook upload page and a live view with pictures and the agent's thoughts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Full suite

- [ ] **Step 1:** `.venv/Scripts/python -m pytest -q`. Expected: all pass. Fix any failure before going on; don't loosen a test to get green.

---

### Task 10: Browser check and sample walks (controller, live API)

The controller does this task, not a subagent: it calls the paid live API and needs the user's approval for public photos.

- [ ] **Step 1: Restart the dev server** (`.claude/launch.json` "scout") so it serves the new code.
- [ ] **Step 2: Upload page.** In the built-in browser, check the upload page at desktop width and at 375 px wide. Check that:
  - the crop cards toggle;
  - the drop zone shows thumbnails;
  - submitting with no crop shows "Choose the crop first."
- [ ] **Step 3: Live walk, end to end.**
  - Run one live walk in the browser: corn, 4 photos (IMG_4214, IMG_4215, IMG_4547, IMG_6564.HEIC).
  - Watch the strip, the stage pictures, the thoughts and the checklist.
  - Let it open the report.
  - Note anything broken.
- [ ] **Step 4: Ask the user which photos may go public, then build the samples.**
  - Ask which photos may go in the public `demo/` folder.
  - Make a soybean walk with the approved photos (IMG_4219, IMG_4220, 3AEE1239).
  - Copy each approved walk's folder (`report.json`, `events.json`, `stop_*.png`, `steps/`) from `store/` to `demo/<walk_id>/`.
  - Write `demo/samples.json` as `[{"walk_id": "...", "name": "Corn, horsetail and grass"}, {"walk_id": "...", "name": "Soybean, waterhemp"}]`.
  - The GPS offset is on by default, so the pins are already shifted.
- [ ] **Step 5: Replay and report.**
  - Open `/`. The sample cards should show thumbnails, stop counts and decisions.
  - Click "Watch the agent". It should replay with the "Replay, 4× speed" badge, then open the report.
  - Check the report's "How the agent worked" sections and the print preview.
- [ ] **Step 6: Send the screenshots.** Send the user screenshots of the upload page, the live view mid-walk, a replay, and the report.
- [ ] **Step 7: Commit.**
```bash
git add demo/
git commit -m "feat: two sample walks that replay the agent at work" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
