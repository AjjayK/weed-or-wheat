"""Run Muse Spark's tool loop for one stop, then turn the state into a StopResult."""
import time
from typing import Callable

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


def _failed(state: StopState) -> StopResult:
    image = state.image
    return StopResult(stop_index=image.index, image_hash=image.hash, gps=image.gps, steps=state.steps,
                      crop_seen=state.crop_seen, flags=[FAILED_FLAG], status="failed")


def run_stop(image: StopImage, crop: str, llm: Llm, sam, weeds: WeedBook, settings: Settings,
             on_step: Callable[[AgentStep], None] | None = None,
             clock: Callable[[], float] = time.monotonic) -> tuple[StopResult, StopState]:
    state = StopState(image, crop, on_step=on_step, clock=clock)
    runner = ToolRunner(state, sam, weeds, max_calls=settings.max_tool_calls)
    ask = dict(instructions=stop_instructions(crop, weeds.ids(), settings.max_tool_calls), tools=TOOL_SCHEMAS)
    pending: list[dict] = [user_message(f"Stop {image.index + 1}: one photo of a {crop} field. Boxes are 0 to 1000 "
                                        "on both axes.", [image.work_data_url])]
    # Everything said so far, replayable on its own: Meta sometimes forgets a response a request chains to.
    history = list(pending)
    previous = None
    stateless = False
    nudged = False
    try:
        while not state.finished:
            if state.calls >= settings.max_tool_calls or clock() - state.started > settings.stop_timeout_s:
                state.log("limit", "Tool-call or time limit reached; closing the stop", ok=False)
                break
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
            history += pending
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
        flags.append(f"…and {len(checks) - MAX_CHECK_FLAGS} more plants to check")
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
