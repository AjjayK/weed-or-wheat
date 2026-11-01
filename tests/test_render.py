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


def test_html_has_decision_options_and_overlays():
    html = report_html(_report(), small_weedbook(), lambda i: f"/walks/w1/overlay/{i}.png")
    assert "Treat (broadcast)" in html and "Product A (100-1)" in html
    assert "/walks/w1/overlay/0.png" in html and "Read the label before use" in html
    assert "#page=3" in html and "Check s0-i6: low confidence" in html
    assert "leaflet" not in html


def test_html_shows_map_when_gps_present():
    html = report_html(_report(gps=(41.5, -93.6)), small_weedbook(), lambda i: f"/o/{i}.png")
    assert "leaflet" in html and "41.5" in html


def test_csv_has_one_row_per_stop():
    lines = report_csv(_report()).strip().splitlines()
    assert lines[0].split(",") == CSV_COLUMNS and len(lines) == 2
    assert "waterhemp:2" in lines[1]
