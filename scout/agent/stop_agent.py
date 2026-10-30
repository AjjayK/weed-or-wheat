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
