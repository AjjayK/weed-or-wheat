from scout.labels.epa import latest_pdf, parse_product

ITEM = {
    "eparegno": "524-549", "productname": "RD 1617 HERBICIDE",
    "active_ingredients": [{"pc_code": "103613", "active_ing": "Glyphosate, potassium salt", "active_ing_percent": 48.7}],
    "sites": [{"site": "CORN, FIELD"}], "pests": [{"pest": "KOCHIA"}], "types": [{"type": "HERBICIDE"}],
    "pdffiles": [{"pdffile": "000524-00549-20200225.pdf"}, {"pdffile": "000524-00549-20230406.pdf"},
                 {"pdffile": "odd-name.pdf"}],
}


def test_parse_product():
    facts = parse_product(ITEM)
    assert facts.reg_no == "524-549" and facts.product_name == "RD 1617 HERBICIDE"
    assert facts.active_ingredients == ["Glyphosate, potassium salt"]
    assert facts.sites == ["CORN, FIELD"] and facts.pests == ["KOCHIA"]
    assert facts.is_herbicide and facts.latest_pdf == "000524-00549-20230406.pdf"


def test_latest_pdf_handles_missing_dates():
    assert latest_pdf([{"pdffile": "odd.pdf"}]) is None
    assert latest_pdf([]) is None
