from scout.labels.build_labels import read_products, with_traits


def test_read_products_keeps_file_order_and_drops_blank_cells(tmp_path):
    path = tmp_path / "focus.csv"
    path.write_text("reg_no,brand,trait_corn,trait_soybean,trait_alfalfa,note\n"
                    "524-659,Roundup PowerMAX 3,glyphosate-tolerant,glyphosate-tolerant,,Roundup\n"
                    "241-310,,Clearfield,,,Pursuit\n"
                    "100-1131,,,,,\n"
                    " ,,,,,blank row\n", encoding="utf-8")
    products = read_products(path)
    assert list(products) == ["524-659", "241-310", "100-1131"]
    assert products["524-659"] == {"brand": "Roundup PowerMAX 3", "trait_corn": "glyphosate-tolerant",
                                   "trait_soybean": "glyphosate-tolerant", "note": "Roundup"}
    assert products["241-310"] == {"trait_corn": "Clearfield", "note": "Pursuit"} and products["100-1131"] == {}


def test_with_traits_sets_each_records_trait_for_its_crop():
    records = [{"reg_no": "241-310", "crop": "corn"}, {"reg_no": "241-310", "crop": "soybean", "requires_trait": "old"}]
    assert [r["requires_trait"] for r in with_traits(records, {"241-310": {"trait_corn": "Clearfield"}})] == [
        "Clearfield", None]
