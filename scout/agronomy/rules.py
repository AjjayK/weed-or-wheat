"""Field decision rules (spec §6). Code decides; the model only explains."""
from dataclasses import dataclass

from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import LabelOption, Outcome, SpeciesSummary, StopResult

SCOUT_AGAIN_FACTOR = 0.5
URGENT_HEIGHT_FACTOR = 0.75
SPOT_MAX_FREQUENCY = 1 / 3
SPOT_MAX_COVER_PCT = 5.0
DEFAULT_ACTION_COVER_PCT = 5.0  # tool default, not an agronomic source
SEVERITY = {"no_action": 0, "scout_again": 1, "treat": 2}


@dataclass
class Decision:
    summaries: list[SpeciesSummary]
    decision: Outcome
    treat_mode: str | None
    rule_trace: list[str]


def _grade(value: float, limit: float) -> Outcome:
    if value >= limit:
        return "treat"
    return "scout_again" if value >= SCOUT_AGAIN_FACTOR * limit else "no_action"


def _species_outcome(info: WeedInfo, present: int, n_ok: int, mean_cover: float,
                     mean_density: float | None, also_cover: bool = False) -> tuple[Outcome, str]:
    """Grade density when it is known, else cover. With also_cover (the density sample missed this species, or
    its count is approximate: a mat of plants segmented as one piece), cover is graded too and the more severe
    outcome wins."""
    name = info.common_name
    if info.zero_tolerance and present:
        return "treat", f"{name}: zero tolerance; present at {present} of {n_ok} stops -> treat"
    if info.action_cover_pct is not None:
        cover = (mean_cover, info.action_cover_pct, "% cover", "sourced threshold")
    else:
        cover = (mean_cover, DEFAULT_ACTION_COVER_PCT, "% cover", "tool default, not from a source")
    checks = [cover]
    if info.action_density is not None and mean_density is not None:
        density = (mean_density, info.action_density, "plants/m²", "sourced threshold")
        checks = [density, cover] if also_cover else [density]
    value, limit, unit, source = max(checks, key=lambda check: SEVERITY[_grade(check[0], check[1])])
    outcome = _grade(value, limit)
    return outcome, f"{name}: {value:g} {unit} vs threshold {limit:g} ({source}) -> {outcome.replace('_', ' ')}"


def decide(stops: list[StopResult], weeds: WeedBook) -> Decision:
    ok = [s for s in stops if s.status == "ok"]
    if not ok:
        return Decision([], "scout_again", None, ["No stop could be analyzed -> scout again with new photos"])
    n = len(ok)
    measured = [s for s in ok if s.view_type != "close_up"] or ok  # a close-up shows one plant, not the field
    top_down = [s for s in measured if s.species_density is not None]
    summaries, trace = [], []
    if n < len(stops):
        trace.append(f"Based on {n} of {len(stops)} stops; the rest could not be analyzed")
    for sid in sorted({sid for s in ok for sid in s.species}):
        present = sum(1 for s in ok if sid in s.species)
        mean_cover = round(sum(s.species_cover_pct.get(sid, 0.0) for s in measured) / len(measured), 2)
        mean_density = (round(sum(s.species_density.get(sid, 0.0) for s in top_down) / len(top_down), 2)
                        if top_down else None)
        heights = [s.species_max_height_in[sid] for s in ok if sid in s.species_max_height_in]
        # Cover is graded too when a stop that shows this species has no density measure or only an approximate
        # count; the more severe outcome wins.
        also_cover = any(sid in s.species and (s.species_density is None or s.count_approximate) for s in measured)
        outcome, line = _species_outcome(weeds.get(sid), present, n, mean_cover, mean_density, also_cover)
        summaries.append(SpeciesSummary(species_id=sid, stops_present=present, frequency=round(present / n, 2),
                                        mean_cover_pct=mean_cover, mean_density_per_m2=mean_density,
                                        max_height_in=max(heights) if heights else None, outcome=outcome))
        trace.append(line)
    if not summaries:
        trace.append("No weeds found at any stop -> no action")
    decision = max((s.outcome for s in summaries), key=SEVERITY.get, default="no_action")
    treat_mode = None
    if decision == "treat":
        treated = {s.species_id for s in summaries if s.outcome == "treat"}
        share = sum(1 for s in ok if treated & s.species.keys()) / n
        field_cover = sum(s.weed_cover_pct for s in measured) / len(measured)
        treat_mode = "spot" if share <= SPOT_MAX_FREQUENCY and field_cover < SPOT_MAX_COVER_PCT else "broadcast"
        trace.append(f"Treat mode: {treat_mode} (treated species at {share:.0%} of stops, "
                     f"mean weed cover {field_cover:.1f}%)")
    return Decision(summaries, decision, treat_mode, trace)


def urgency(max_height_in: float | None, options: list[LabelOption]) -> str | None:
    limits = [o.record.max_weed_height_in for o in options if o.record.max_weed_height_in is not None]
    if max_height_in is None or not limits:
        return None
    within = [limit for limit in limits if max_height_in <= limit]  # same test as LabelOption.over_size_limit
    if not within:
        return f"Beyond label size limit ({max_height_in:g} in vs {max(limits):g} in)"
    limit = min(within)
    if max_height_in >= URGENT_HEIGHT_FACTOR * limit:
        return f"Treat within ~3 days ({max_height_in:g} in, label limit {limit:g} in)"
    return f"Treat within ~7 days ({max_height_in:g} in, label limit {limit:g} in)"
