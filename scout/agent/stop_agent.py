"""Run Muse Spark's tool loop for one stop, then turn the state into a StopResult."""
import time
from typing import Callable

from PIL import Image

from scout.agent.llm import Llm, ToolCall, is_lost_response, user_message
from scout.agent.prompts import stop_instructions
from scout.agent.state import MAX_MARK_BOXES, StopState
from scout.agent.tools import OVERLAY_CAPTION, TOOL_SCHEMAS, ToolRunner
from scout.agronomy.weeds import WeedBook
from scout.app.ingest import StopImage
from scout.config import Settings
from scout.models import AgentStep, Instance, StopResult
from scout.retry import BadRequest
from scout.vision.measure import compute_metrics

FAILED_FLAG = "This stop could not be analyzed"
UNMARKED_FLAG = "Some weeds may be unmarked: cover and count are minimums"
MAX_CHECK_FLAGS = 10  # per-plant "Check ..." lines shown for one stop; the rest are counted in one line
THOUGHT_LINES = 4  # reasoning lines kept per turn
THOUGHT_MAX_CHARS = 200
THOUGHT_SUMMARY_MAX_CHARS = 300  # the summary shows in large type on the live view


def run_order(calls: list[ToolCall]) -> list[ToolCall]:
    """The survey runs first, so marks can be checked against its estimate; crop marks and segments next, so the
    weed check sees the crop; then the other tools; measure and finish last, so they see the turn's verdicts. The
    order is stable within each rank."""
    def rank(call: ToolCall) -> int:
        if call.name == "survey":
            return -1
        if call.name in ("mark", "segment") and call.arguments.get("kind") == "crop":
            return 0
        # identify after the turn's marks; classify after identify, so a weed it leaves unnamed gets the panel's name
        return {"identify": 1.5, "classify": 1.7, "measure": 2, "finish": 3}.get(call.name, 1)
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


def _failed(state: StopState) -> StopResult:
    image = state.image
    return StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, steps=state.steps,
                      crop_seen=state.crop_seen, weeds_estimate=state.weeds_estimate,
                      all_weeds_marked=state.all_weeds_marked, flags=[FAILED_FLAG], status="failed")


def run_stop(image: StopImage, crop: str, llm: Llm, sam, weeds: WeedBook, settings: Settings,
             on_step: Callable[[AgentStep], None] | None = None,
             clock: Callable[[], float] = time.monotonic,
             picture_sink: Callable[[Image.Image], str] | None = None) -> tuple[StopResult, StopState]:
    state = StopState(image, crop, on_step=on_step, clock=clock, picture_sink=picture_sink)
    runner = ToolRunner(state, sam, weeds, max_calls=settings.max_tool_calls, time_limit_s=settings.stop_timeout_s,
                        llm=llm, id_samples=settings.id_samples, id_effort=settings.id_effort,
                        id_timeout_s=settings.id_timeout_s)
    ask = dict(instructions=stop_instructions(crop, weeds.ids(), settings.max_tool_calls), tools=TOOL_SCHEMAS,
               reasoning_summary=True)
    pending: list[dict] = [user_message(f"Stop {image.index + 1}: one photo of a {crop} field. Boxes are 0 to 1000 "
                                        "on both axes.", [image.work_data_url])]
    # Everything said so far, replayable on its own: Meta sometimes forgets a response a request chains to.
    history = list(pending)
    previous = None
    stateless = False
    nudged = False
    hurried = False
    graced = False
    try:
        while not state.finished:
            out_of_calls = state.calls >= settings.max_tool_calls
            if out_of_calls or clock() - state.started > settings.stop_timeout_s:
                if out_of_calls or graced or not _unclassified(state):
                    state.log("limit", "Tool-call or time limit reached; closing the stop", ok=False)
                    break
                # A thorough turn can end past the limit with the weeds named by identify but not classified:
                # one last turn classifies them, instead of a report of unchecked plants.
                graced = True
                state.log("limit", "Time is up; one last turn to classify the weeds", ok=False)
                last = user_message("Time is up. In this turn, classify every weed not classified yet (species_id "
                                    "null for the weeds identify named), then measure and finish. No other tools.")
                pending.append(last)
                history.append(last)
            try:
                turn = llm.turn(list(history) if stateless else pending, previous_response_id=previous, **ask)
            except BadRequest as error:
                if previous is None or not is_lost_response(error):
                    raise
                state.log("retry", "Muse Spark lost the conversation; resent it in full", ok=False)
                stateless = True
                turn = llm.turn(list(history), previous_response_id=None, **ask)  # a second failure ends the stop
            previous = None if stateless else turn.response_id
            history += turn.output_items
            thought = (turn.text or "").strip()
            if thought or turn.reasoning:  # what the agent is thinking, for people watching; before the turn's tools
                lines = turn.reasoning if thought else turn.reasoning[1:]  # no narration: line 1 is the thought
                state.log("think", (thought or turn.reasoning[0])[:THOUGHT_SUMMARY_MAX_CHARS],
                          detail=[line[:THOUGHT_MAX_CHARS] for line in lines[:THOUGHT_LINES]])
            if not turn.calls:
                if nudged:
                    break
                nudged = True
                pending = [user_message("Call a tool. If you are done, call measure and then finish.")]
            else:
                outputs, images = {}, []
                for call in run_order(turn.calls):
                    state.calls += 1
                    result = runner.run(call.name, call.arguments)
                    outputs[call.call_id] = result.output
                    images += result.images
                pending = [{"type": "function_call_output", "call_id": call.call_id, "output": outputs[call.call_id]}
                           for call in turn.calls]  # answered in the model's own order
                images = latest_overlay_only(images)
                if images:
                    pending.append(user_message("Pictures for your tool calls: " + "; ".join(c for c, _ in images),
                                                [url for _, url in images]))
            left = settings.stop_timeout_s - (clock() - state.started)
            if not state.finished and not hurried and 0 < left < settings.wrap_up_s:  # past it, the last turn's note
                # a turn at max effort can take a minute: without this, a thorough stop ends with nothing classified
                hurried = True
                state.log("limit", f"{max(round(left), 0)} s left; asked the agent to wrap up")
                pending.append(user_message(
                    f"Time is nearly up ({max(round(left), 0)} s left). Send everything in this one turn: identify "
                    "(if you haven't yet: one group per kind, every instance of it listed), then classify every "
                    "weed (species_id null for the weeds identify names), then measure and finish."))
            history += pending
        return finalize(state, sam), state
    except Exception as error:  # StopFailed, BadRequest, LlmError, timeouts, dropped connections: this stop only
        state.log("error", f"{type(error).__name__}: {error}", ok=False)
        return _failed(state), state


def _unclassified(state: StopState) -> bool:
    return any(inst.role == "weed" and inst.id_confidence is None for inst in state.instances.values())


def _flag(inst: Instance) -> str:
    if inst.id_confidence is None:
        return f"Check {inst.id}: marked but never classified"
    if inst.id_confidence == "low":
        return f"Check {inst.id}: low confidence"
    return f"Check {inst.id}: may be crop or several plants"


def finalize(state: StopState, sam) -> StopResult:
    """Weeds that were marked but never classified stay weeds, flagged, named by identify's panel when it voted on
    their kind; measurements are recomputed."""
    if not state.instances and state.view_type is None:  # the agent got nowhere: its failure, not a stop without crop
        return _failed(state)
    for inst in state.instances.values():
        if inst.role == "weed" and inst.id_confidence is None:
            inst.check_this = True
            panel = state.identified.get(inst.id)
            if inst.species_id is None and panel and panel[0]:
                inst.species_id = panel[0]  # time ran out after identify, before classify
    view = state.view_type or "row"
    if view in ("row", "wide"):
        state.ensure_sky(sam)
    m = compute_metrics(list(state.instances.values()), state.masks, state.size, view, state.sky, state.row_band())
    instances = list(state.instances.values())
    # Stop-level flags first, then one line per plant to check, capped: up to 120 weeds could each be unclassified.
    flags = []
    mismatch = crop_flag(state.crop_seen, state.crop)
    if mismatch:
        flags.append(mismatch)
    if state.box_cap_hit:
        flags.append(f"Weed count is a floor: a mark call used all {MAX_MARK_BOXES} boxes")
    unmarked = state.maybe_unmarked(plants=m.weed_count)
    if unmarked:
        flags.append(UNMARKED_FLAG)
    if m.count_approximate or (state.has_patch() and view != "close_up") or state.has_weed_area():
        flags.append("Weed count is approximate: some weeds merged into patches or were counted in weed areas")
    checks = [_flag(i) for i in instances if i.role == "weed" and i.check_this]
    flags += checks[:MAX_CHECK_FLAGS]
    if len(checks) > MAX_CHECK_FLAGS:
        flags.append(f"...and {len(checks) - MAX_CHECK_FLAGS} more plants to check")
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
        notes=state.notes, crop_seen=state.crop_seen, weeds_estimate=state.weeds_estimate,
        all_weeds_marked=state.all_weeds_marked, steps=state.steps,
        status="ok" if has_crop or (state.crop == "alfalfa" and worked) else "no_crop_found",
    )
