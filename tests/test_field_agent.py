from scout.agent.field_agent import build_report, crop_mismatch, numbers_ok
from scout.agent.llm import LlmTurn
from scout.agronomy.weeds import WeedBook, WeedInfo
from scout.models import StopResult, UseRecord
from tests.helpers import FakeLlm

BOOK = WeedBook({"waterhemp": WeedInfo(id="waterhemp", common_name="waterhemp", scientific_name="A. tuberculatus",
                                       type="broadleaf", zero_tolerance=True, sources=["src"])})
RECORD = UseRecord(reg_no="100-1", product_name="Product A", active_ingredients=["fomesafen"], herbicide_groups=["14"],
                   crop="soybean", species_id="waterhemp", weed_label_text="waterhemp", timing="postemergence",
                   max_weed_height_in=4.0, page=3, quote="up to 4 inches")


def _stops():
    return [StopResult(stop_index=0, image_hash="a", species={"waterhemp": 2}, species_cover_pct={"waterhemp": 1.2},
                       species_max_height_in={"waterhemp": 3.0}, weed_cover_pct=1.2, weed_count=2, crop_stage="V3"),
            StopResult(stop_index=1, image_hash="b", crop_stage="V3")]


def test_numbers_ok():
    assert numbers_ok("Waterhemp at 1 of 2 stops, 3 in tall.", '{"a": 1, "b": 2.0, "h": 3.0}')
    assert not numbers_ok("Spray 12 oz.", '{"a": 1}')


def test_report_uses_llm_text_when_numbers_check_out():
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found and has zero tolerance. Read the label before use.")])
    report = build_report("w1", "soybean", _stops(), BOOK, [RECORD], llm)
    assert report.decision == "treat" and report.treat_mode == "broadcast"
    assert [o.record.reg_no for o in report.options] == ["100-1"]
    assert report.urgency.startswith("Treat within ~3 days")
    assert report.explanation.startswith("Waterhemp was found")


def test_explanation_input_names_a_required_seed_trait():
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found. Read the label before use.")])
    record = RECORD.model_copy(update={"requires_trait": "glufosinate-tolerant"})
    build_report("w1", "soybean", _stops(), BOOK, [record], llm)
    assert '"requires_trait": "glufosinate-tolerant"' in llm.requests[0]["input"][0]["content"][0]["text"]


def test_report_falls_back_to_template_on_invented_numbers():
    llm = FakeLlm([LlmTurn("r1", [], "Spray 99 acres."), LlmTurn("r2", [], "Spray 98 acres.")])
    report = build_report("w1", "soybean", _stops(), BOOK, [RECORD], llm)
    assert report.explanation.startswith("Treat") and report.explanation.endswith("Read the label before use.")


def test_no_label_match_is_stated():
    report = build_report("w1", "soybean", _stops(), BOOK, [], FakeLlm([]))
    assert report.options == []
    assert any("No product in the current label set covers waterhemp on soybean" in line for line in report.rule_trace)


def test_one_product_listed_for_each_treated_weed_and_unidentified_asks_for_an_id():
    book = WeedBook({**{sid: BOOK.get(sid) for sid in BOOK.ids()},
                     "velvetleaf": WeedInfo(id="velvetleaf", common_name="velvetleaf", scientific_name="A. theophrasti",
                                            type="broadleaf", zero_tolerance=True, sources=["src"]),
                     "unidentified_broadleaf": WeedInfo(id="unidentified_broadleaf", common_name="unidentified broadleaf",
                                                        scientific_name="", type="broadleaf", zero_tolerance=True,
                                                        sources=["src"])})
    stops = [StopResult(stop_index=0, image_hash="a", species={"waterhemp": 1, "velvetleaf": 1, "unidentified_broadleaf": 1},
                        species_cover_pct={"waterhemp": 1.0, "velvetleaf": 1.0, "unidentified_broadleaf": 1.0},
                        weed_cover_pct=3.0, weed_count=3)]
    velvetleaf = RECORD.model_copy(update={"species_id": "velvetleaf", "weed_label_text": "velvetleaf"})
    report = build_report("w1", "soybean", stops, book, [RECORD, velvetleaf], FakeLlm([]))
    assert [(o.record.reg_no, o.record.species_id) for o in report.options] == [("100-1", "velvetleaf"), ("100-1", "waterhemp")]
    assert any(line.startswith("Identify the unidentified broadleaf") for line in report.rule_trace)
    assert not any("covers unidentified" in line for line in report.rule_trace)


def test_report_warns_when_photos_show_another_crop():
    stops = [_stops()[0].model_copy(update={"crop_seen": "corn"}),
             _stops()[1].model_copy(update={"crop_seen": "unsure"})]
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found. Read the label before use.")])
    report = build_report("w1", "soybean", stops, BOOK, [RECORD], llm)
    assert report.crop_mismatch == ("1 of 2 photos looks like corn, but this walk was scouted as soybean. "
                                    "The label options below are for soybean. Re-run with the right crop.")
    assert report.decision == "treat"  # the farmer's pick still decides


def test_no_crop_warning_when_the_crop_matches_or_is_unclear():
    stops = [_stops()[0].model_copy(update={"crop_seen": "soybean"}), _stops()[1]]
    llm = FakeLlm([LlmTurn("r1", [], "Waterhemp was found. Read the label before use.")])
    assert build_report("w1", "soybean", stops, BOOK, [RECORD], llm).crop_mismatch is None


def test_crop_warning_wording_for_several_photos_and_other_crops():
    two = [StopResult(stop_index=i, image_hash=str(i), crop_seen="corn") for i in range(2)]
    assert crop_mismatch("soybean", two).startswith("2 of 2 photos look like corn")
    other = [StopResult(stop_index=0, image_hash="a", crop_seen="other")]
    assert crop_mismatch("corn", other).startswith("1 of 1 photo doesn't look like corn. The label options below are for corn.")


def test_crop_warning_says_plainly_when_photos_show_different_crops():
    mixed = [StopResult(stop_index=0, image_hash="a", crop_seen="corn"),
             StopResult(stop_index=1, image_hash="b", crop_seen="alfalfa")]
    assert crop_mismatch("soybean", mixed).startswith("2 of 2 photos don't look like soybean")


def test_crop_warning_counts_stops_that_found_no_crop_but_not_failed_ones():
    stops = [StopResult(stop_index=0, image_hash="a", crop_seen="corn", status="no_crop_found"),
             StopResult(stop_index=1, image_hash="b", crop_seen="corn", status="no_crop_found"),
             StopResult(stop_index=2, image_hash="c", crop_seen="corn", status="failed")]
    assert crop_mismatch("soybean", stops).startswith("2 of 2 photos look like corn")


def test_crop_warning_plural_other_form():
    others = [StopResult(stop_index=i, image_hash=str(i), crop_seen="other") for i in range(2)]
    assert crop_mismatch("corn", others).startswith("2 of 2 photos don't look like corn")
