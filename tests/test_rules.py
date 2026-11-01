from scout.agronomy.rules import decide, urgency
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import LabelOption, StopResult, UseRecord

BOOK = WeedBook({
    "waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="A. tuberculatus",
                          type="broadleaf", zero_tolerance=True, sources=["src"]),
    "velvetleaf": WeedInfo(id="velvetleaf", common_name="velvetleaf", scientific_name="A. theophrasti",
                           type="broadleaf", action_density=2.0, sources=["src"]),
    "giant_foxtail": WeedInfo(id="giant_foxtail", common_name="giant foxtail", scientific_name="S. faberi",
                              type="grass"),
})


def _stop(i, species=None, cover=None, density=None, heights=None, status="ok", approximate=False, view="row"):
    species, cover = species or {}, cover or {}
    return StopResult(stop_index=i, image_hash=f"h{i}", species=species, species_cover_pct=cover,
                      species_density=density, species_max_height_in=heights or {}, view_type=view,
                      weed_cover_pct=sum(cover.values()), weed_count=sum(species.values()), status=status,
                      count_approximate=approximate)


def test_zero_tolerance_species_triggers_spot_treatment():
    d = decide([_stop(0, {"waterhemp": 1}, {"waterhemp": 0.2}), _stop(1), _stop(2), _stop(3)], BOOK)
    assert d.decision == "treat" and d.treat_mode == "spot"
    assert "zero tolerance" in d.rule_trace[0]


def test_density_threshold_uses_top_down_stops():
    stops = [_stop(0, {"velvetleaf": 3}, {"velvetleaf": 2.0}, density={"velvetleaf": 3.0}),
             _stop(1, {"velvetleaf": 1}, {"velvetleaf": 1.0}, density={"velvetleaf": 1.0})]
    d = decide(stops, BOOK)
    assert d.summaries[0].mean_density_per_m2 == 2.0 and d.summaries[0].outcome == "treat"
    assert d.treat_mode == "broadcast"


def test_approximate_count_also_grades_cover():
    mat = [_stop(0, {"velvetleaf": 1}, {"velvetleaf": 40.0}, density={"velvetleaf": 0.5}, approximate=True)]
    d = decide(mat, BOOK)
    assert d.summaries[0].outcome == "treat" and "% cover" in d.rule_trace[0]
    exact = [_stop(0, {"velvetleaf": 1}, {"velvetleaf": 40.0}, density={"velvetleaf": 0.5})]
    assert decide(exact, BOOK).summaries[0].outcome == "no_action"


def test_default_cover_threshold_gives_scout_again():
    stops = [_stop(0, {"giant_foxtail": 4}, {"giant_foxtail": 3.0}),
             _stop(1, {"giant_foxtail": 2}, {"giant_foxtail": 2.0})]
    d = decide(stops, BOOK)
    assert d.summaries[0].mean_cover_pct == 2.5 and d.decision == "scout_again"
    assert "tool default" in d.rule_trace[0]


def test_density_species_seen_only_where_density_unknown_is_graded_on_cover():
    stops = [_stop(0, {"velvetleaf": 5}, {"velvetleaf": 30.0}), _stop(1, {"velvetleaf": 5}, {"velvetleaf": 30.0}),
             _stop(2, density={}, view="top_down")]
    d = decide(stops, BOOK)
    assert d.summaries[0].mean_density_per_m2 == 0.0
    assert d.summaries[0].outcome == "treat" and "% cover" in d.rule_trace[0]


def test_close_up_counts_presence_but_not_cover():
    stops = [_stop(0, {"giant_foxtail": 1}, {"giant_foxtail": 1.0}), _stop(1), _stop(2),
             _stop(3, {"giant_foxtail": 1}, {"giant_foxtail": 60.0}, view="close_up")]
    d = decide(stops, BOOK)
    assert d.summaries[0].frequency == 0.5 and d.summaries[0].mean_cover_pct == 0.33
    assert d.decision == "no_action"


def test_spot_mode_counts_stops_with_any_treated_species():
    top = {"view": "top_down"}
    stops = [_stop(0, {"waterhemp": 1}, {"waterhemp": 0.5}), _stop(1, {"waterhemp": 1}, {"waterhemp": 0.5}),
             _stop(2, {"velvetleaf": 4}, {"velvetleaf": 0.5}, density={"velvetleaf": 4.0}, **top),
             _stop(3, {"velvetleaf": 4}, {"velvetleaf": 0.5}, density={"velvetleaf": 4.0}, **top),
             _stop(4, density={}, **top), _stop(5, density={}, **top)]
    d = decide(stops, BOOK)
    assert [s.outcome for s in d.summaries] == ["treat", "treat"]
    assert d.treat_mode == "broadcast"  # 4 of 6 stops hold a treated species, though each species is at 2 of 6


def test_no_weeds_and_no_ok_stops():
    assert decide([_stop(0), _stop(1)], BOOK).decision == "no_action"
    failed = decide([_stop(0, status="failed"), _stop(1, status="no_crop_found")], BOOK)
    assert failed.decision == "scout_again" and failed.summaries == []
    assert failed.rule_trace == ["No stop could be analyzed -> scout again with new photos"]


def test_partial_analysis_is_noted_in_trace():
    d = decide([_stop(0, {"giant_foxtail": 1}, {"giant_foxtail": 1.0}), _stop(1, status="failed")], BOOK)
    assert d.rule_trace[0] == "Based on 1 of 2 stops; the rest could not be analyzed"


def _option(limit):
    record = UseRecord(reg_no="1-1", product_name="P", active_ingredients=[], herbicide_groups=["14"], crop="soybean",
                       species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                       max_weed_height_in=limit, page=1, quote="q")
    return LabelOption(record=record)


def test_urgency_branches():
    assert urgency(4.5, [_option(4.0)]).startswith("Beyond label size limit")
    assert urgency(4.0, [_option(4.0)]).startswith("Treat within ~3 days")  # "up to 4 inches" includes 4
    assert urgency(3.0, [_option(4.0)]).startswith("Treat within ~3 days")
    assert urgency(1.0, [_option(4.0)]).startswith("Treat within ~7 days")
    assert urgency(None, [_option(4.0)]) is None and urgency(2.0, [_option(None)]) is None


def test_urgency_ignores_limits_the_weed_already_exceeds():
    assert urgency(4.0, [_option(2.0), _option(6.0)]).startswith("Treat within ~7 days")  # the 6-in label fits
    assert urgency(7.0, [_option(2.0), _option(6.0)]) == "Beyond label size limit (7 in vs 6 in)"
