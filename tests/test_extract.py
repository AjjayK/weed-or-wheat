import json

from scout.agent.llm import LlmTurn
from scout.labels.epa import ProductFacts
from scout.labels.extract import USE_RECORDS_FORMAT, extract_use_records, verify
from scout.models import UseRecord
from tests.helpers import FakeLlm, small_weedbook

FACTS = ProductFacts(reg_no="100-1", product_name="Test Herbicide", active_ingredients=["fomesafen"], sites=[],
                     pests=[], types=["HERBICIDE"], latest_pdf="000100-00001-20240101.pdf")


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


def test_verify_drops_short_quotes():
    kept, dropped = verify([_record(1, "Cover")], ["Cover page"])
    assert kept == [] and len(dropped) == 1
