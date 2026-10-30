from scout.agronomy.weeds import WeedBook


def test_default_book_loads_all_species():
    book = WeedBook.load()
    assert len(book.ids()) == 35
    assert "unidentified_broadleaf" in book and "unidentified_grass" in book


def test_resolve_names_and_synonyms():
    book = WeedBook.load()
    assert book.resolve("Marestail") == "horseweed"
    assert book.resolve("Amaranthus tuberculatus") == "waterhemp"
    assert book.resolve("giant foxtail") == "giant_foxtail"
    assert book.resolve("not a weed") is None
    assert book.resolve("") is None


def test_shipped_thresholds_have_sources():
    assert WeedBook.load().validate() == []
