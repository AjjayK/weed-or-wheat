import json
from dataclasses import replace

from scout.agent.llm import LlmTurn
from scout.labels.epa import ProductFacts
from scout.labels.extract import USE_RECORDS_FORMAT, extract_use_records, verify
from scout.models import UseRecord
from tests.helpers import FakeLlm, small_weedbook

FACTS = ProductFacts(reg_no="100-1", product_name="Test Herbicide", active_ingredients=["fomesafen"],
                     sites=["SOYBEANS (FOLIAR TREATMENT)"], pests=[], types=["HERBICIDE TERRESTRIAL"],
                     latest_pdf="000100-00001-20240101.pdf")


def _item(**over):
    base = {"crop": "soybean", "species_id": "waterhemp", "weed_label_text": "Waterhemp", "timing": "postemergence",
            "herbicide_groups": ["14"], "max_weed_height_in": 4, "crop_stage_min": None, "crop_stage_max": "R1",
            "rate_min": 1.0, "rate_max": 1.5, "rate_unit": "pt/A", "phi_days": None, "rei_hours": 24, "page": 3,
            "quote": "Apply to waterhemp up to 4 inches tall"}
    base.update(over)
    return base


def test_extract_maps_species_and_fills_product_facts(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    llm = FakeLlm([LlmTurn("r1", [], json.dumps({"records": [
        _item(), _item(species_id="Marestail"), _item(species_id="dragonweed")]}))])
    records = extract_use_records(pdf, FACTS, llm, small_weedbook())
    assert [r.species_id for r in records] == ["waterhemp", "horseweed"]
    assert records[0].reg_no == "100-1" and records[0].pdf_file == "000100-00001-20240101.pdf"
    request = llm.requests[0]
    assert request["text_format"] == USE_RECORDS_FORMAT
    assert request["input"][0]["content"][1]["type"] == "input_file"


def _prompt(llm, n):
    return llm.requests[n]["input"][0]["content"][0]["text"]


def test_extract_asks_once_per_labeled_crop_and_keeps_only_that_crop(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    facts = replace(FACTS, sites=["CORN (FIELD) (FOLIAR TREATMENT)", "SOYBEANS (FOLIAR TREATMENT)"])
    llm = FakeLlm([LlmTurn("r1", [], json.dumps({"records": [_item(crop="corn"), _item(crop="soybean")]})),
                   LlmTurn("r2", [], json.dumps({"records": [_item(species_id="Marestail")]}))])
    records = extract_use_records(pdf, facts, llm, small_weedbook())
    assert len(llm.requests) == 2
    assert "on corn only" in _prompt(llm, 0) and "on soybean only" in _prompt(llm, 1)
    assert [(r.crop, r.species_id) for r in records] == [("corn", "waterhemp"), ("soybean", "horseweed")]


def test_extract_crops_default_to_all_three_and_skip_unlabeled_ones(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    empty = json.dumps({"records": []})
    llm = FakeLlm([LlmTurn(f"r{n}", [], empty) for n in range(3)])
    extract_use_records(pdf, replace(FACTS, sites=[]), llm, small_weedbook())
    assert len(llm.requests) == 3
    llm = FakeLlm([])
    assert extract_use_records(pdf, replace(FACTS, sites=["WHEAT"]), llm, small_weedbook()) == []
    assert llm.requests == []


def test_extract_matches_crop_sites_as_whole_words(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    llm = FakeLlm([LlmTurn("r1", [], json.dumps({"records": []}))])
    extract_use_records(pdf, replace(FACTS, sites=["ACORN SQUASH", "CORN SALAD", "SOYBEANS (FOLIAR TREATMENT)"]),
                        llm, small_weedbook())
    assert len(llm.requests) == 1 and "on soybean only" in _prompt(llm, 0)


def test_extract_reduces_herbicide_groups_to_digits(tmp_path):
    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    item = _item(herbicide_groups=["Group 15", "Group 27 (HPPD)", "27"])
    llm = FakeLlm([LlmTurn("r1", [], json.dumps({"records": [item]}))])
    assert extract_use_records(pdf, FACTS, llm, small_weedbook())[0].herbicide_groups == ["15", "27"]


def _record(page, quote):
    return UseRecord(reg_no="1", product_name="P", active_ingredients=[], herbicide_groups=[], crop="soybean",
                     species_id="waterhemp", weed_label_text="w", timing="postemergence", page=page, quote=quote)


def test_verify_keeps_found_quotes_and_fixes_adjacent_page():
    texts = ["Cover page", "Apply to  waterhemp up to\n4 inches tall.", "Other text"]
    kept, dropped = verify([_record(2, "apply to waterhemp up to 4 inches tall"),
                            _record(3, "Apply to waterhemp up to 4 inches tall"),
                            _record(1, "never printed anywhere on label")], texts)
    assert [r.page for r in kept] == [2, 2]
    assert len(dropped) == 1


PAGE3 = ("Page 1 \nuse after a mesotrione-\ncontaining herbicide.  Tank mix with Dual II Ma gnum® \n"
         "If dry, apply ½ to 1 inch of water.  Do not exceed 3.0 fl oz per acre.")


def test_verify_survives_pdf_text_artifacts_and_printed_page_numbers():
    texts = ["EPA acceptance letter", "letter page 2", PAGE3]
    quotes = ["use after a mesotrione-containing herbicide", "Tank mix with Dual II Magnum®",
              "apply 1/2 to 1 inch of water", "Do not exceed 3.0 fl oz per acre"]
    kept, dropped = verify([_record(1, q) for q in quotes], texts)  # cited by the printed "Page 1"
    assert dropped == [] and [r.page for r in kept] == [3, 3, 3, 3]


def test_verify_rejects_changed_words_and_numbers():
    texts = ["EPA acceptance letter", "letter page 2", PAGE3]
    kept, _ = verify([_record(3, "Do not exceed 30 fl oz per acre"), _record(3, "apply 12 to 1 inch of water"),
                      _record(3, "Tank mix with Dual Magnum"), _record(3, "Do exceed 3.0 fl oz per acre")], texts)
    assert kept == []


def test_verify_picks_the_nearest_page_when_a_quote_repeats():
    texts = ["Do not apply more than twice"] + ["other"] * 3 + ["Do not apply more than twice"]
    kept, _ = verify([_record(4, "Do not apply more than twice")], texts)
    assert kept[0].page == 5


def test_verify_drops_short_quotes():
    kept, dropped = verify([_record(1, "Cover")], ["Cover page"])
    assert kept == [] and len(dropped) == 1
