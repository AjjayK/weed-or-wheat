import re

from scout.labels.epa import PDF_BASE
from scout.models import FieldReport, LabelOption, SpeciesSummary, StopResult, UseRecord
from scout.report.render import CSV_COLUMNS, report_csv, report_html
from tests.helpers import small_weedbook


def _report(gps=None):
    record = UseRecord(reg_no="100-1", product_name="Product A", active_ingredients=[], herbicide_groups=["14"],
                       crop="soybean", species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                       max_weed_height_in=4.0, rate_min=1.0, rate_max=1.5, rate_unit="pt/A", page=3,
                       quote="up to 4 inches", pdf_file="000100-00001-20240101.pdf")
    stop = StopResult(stop_index=0, image_hash="a", gps=gps, species={"waterhemp": 2},
                      species_cover_pct={"waterhemp": 1.2}, weed_cover_pct=1.2, weed_count=2,
                      flags=["Check s0-i6: low confidence"])
    summary = SpeciesSummary(species_id="waterhemp", stops_present=1, frequency=1.0, mean_cover_pct=1.2,
                             mean_density_per_m2=None, max_height_in=3.0, outcome="treat")
    return FieldReport(walk_id="w1", crop="soybean", stops=[stop], species_summary=[summary], decision="treat",
                       treat_mode="broadcast", urgency="Treat within ~3 days (3 in, label limit 4 in)",
                       rule_trace=["waterhemp: zero tolerance"], explanation="Waterhemp found.",
                       options=[LabelOption(record=record)])


def _with_stops(*stops):
    return _report().model_copy(update={"stops": list(stops)})


def _html(report):
    return report_html(report, small_weedbook(), lambda i: f"/o/{i}.png")


def test_html_has_decision_options_and_overlays():
    html = report_html(_report(), small_weedbook(), lambda i: f"/walks/w1/overlay/{i}.png")
    assert "Treat (broadcast)" in html and "Product A (100-1)" in html
    assert "/walks/w1/overlay/0.png" in html and "Read the label before use" in html
    assert f"{PDF_BASE}000100-00001-20240101.pdf#page=3" in html and "Check s0-i6: low confidence" in html
    assert "leaflet" not in html


def test_options_table_names_the_weed_each_row_is_for():
    assert re.search(r"<td>waterhemp</td>\s*<td>Product A \(100-1\)</td>", _html(_report()))


def test_options_table_says_when_a_product_needs_a_seed_trait():
    report = _report()
    record = report.options[0].record.model_copy(update={"requires_trait": "glyphosate-tolerant"})
    with_trait = report.model_copy(update={"options": [LabelOption(record=record)]})
    assert "Requires glyphosate-tolerant seed" in _html(with_trait)
    assert "seed</div>" not in _html(report)


def test_html_shows_map_when_gps_present():
    html = _html(_report(gps=(41.5, -93.6)))
    assert "leaflet" in html and "41.5" in html and "Pins are shifted for privacy." in html


def test_leaflet_assets_are_pinned_with_integrity_hashes():
    html = _html(_report(gps=(41.5, -93.6)))
    assert html.count("leaflet@1.9.4") == 2
    assert html.count('integrity="sha256-') == 2 and html.count('crossorigin=""') == 2


def test_mean_cover_card_skips_close_ups_like_the_rules():
    html = _html(_with_stops(StopResult(stop_index=0, image_hash="a", weed_cover_pct=2.0),
                             StopResult(stop_index=1, image_hash="b", weed_cover_pct=60.0, view_type="close_up")))
    assert "Mean weed cover<b>2.0%</b>" in html


def test_failed_stop_is_not_shown_as_zeros():
    html = _html(_with_stops(StopResult(stop_index=0, image_hash="a", weed_cover_pct=2.0),
                             StopResult(stop_index=1, image_hash="b", status="failed",
                                        flags=["This stop could not be analyzed"])))
    assert "Not analyzed; not used in the field numbers." in html and "Weed cover 0.0%" not in html
    assert "This stop could not be analyzed" in html


def test_model_text_is_escaped():
    report = _report().model_copy(update={"explanation": "<script>alert(1)</script>"})
    report.stops[0].crop_stage = "<img src=x onerror=alert(1)>"
    html = _html(report)
    assert "<script>alert(1)" not in html and "<img src=x" not in html and "&lt;script&gt;" in html


def test_csv_has_one_row_per_stop():
    lines = report_csv(_report()).strip().splitlines()
    assert lines[0].split(",") == CSV_COLUMNS and len(lines) == 2
    assert "waterhemp:2" in lines[1]


def test_csv_leaves_a_failed_stops_numbers_blank():
    report = _with_stops(StopResult(stop_index=2, image_hash="c", status="failed",
                                    flags=["This stop could not be analyzed"]))
    assert report_csv(report).strip().splitlines()[-1] == "3,failed,,,,,,,,,,This stop could not be analyzed"


def test_crop_mismatch_banner_comes_before_the_decision():
    warning = "1 of 1 photos looks like corn, but this walk was scouted as soybean."
    html = _html(_report().model_copy(update={"crop_mismatch": warning}))
    assert warning in html and html.index(warning) < html.index("Decision")
    assert "looks like corn" not in _html(_report())


def test_legend_explains_blue_weed_areas():
    assert "Blue box: weed area measured by green pixels" in _html(_report())
