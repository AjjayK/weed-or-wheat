import pytest

from scout.agronomy.weeds import WeedBook, WeedInfo


def test_default_book_loads_all_species():
    book = WeedBook.load()
    assert len(book.ids()) == 36
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


def _info(sid, **kw):
    return WeedInfo(id=sid, common_name=kw.pop("common_name", sid), scientific_name="", type="broadleaf", **kw)


@pytest.mark.parametrize("phrase, expected", [
    ("waterhemp pigweed", "waterhemp"),
    ("Waterhemp (Amaranthus tuberculatus)", "waterhemp"),
    ("dandelion flower", "dandelion"),
    ("Dandelions", "dandelion"),
    ("green foxtail seedlings", "green_foxtail"),
    ("giant_foxtail", "giant_foxtail"),
    ("Lamb's-quarters", "common_lambsquarters"),
    ("lambs quarters", "common_lambsquarters"),
    ("velvet leaf", "velvetleaf"),
    ("morning glories", "morningglory"),
    ("Palmer's amaranth", "palmer_amaranth"),
    ("horsetail weed", "field_horsetail"),
    ("Marestail", "horseweed"),
])
def test_resolve_tolerates_model_phrasing(phrase, expected):
    assert WeedBook.load().resolve(phrase) == expected


@pytest.mark.parametrize("phrase", ["pigweed", "ragweed", "giant ragweed or common ragweed", "not a weed", "", None])
def test_resolve_refuses_unknown_and_ambiguous(phrase):
    assert WeedBook.load().resolve(phrase) is None


def test_every_name_belongs_to_one_species():
    assert [p for p in WeedBook.load().validate() if "several species" in p] == []


def test_validate_flags_bad_entries():
    book = WeedBook({
        "a": _info("a", zero_tolerance=True),
        "b": WeedInfo(id="b", common_name="b", scientific_name="", type="weedy", action_density=1.5, sources=["s"]),
        "c": _info("c", action_cover_pct="1-2", sources=["s"]),
        "d": _info("d", synonyms=["shared"]), "e": _info("e", synonyms=["shared"]),
    })
    problems = "\n".join(book.validate())
    for fragment in ("a: threshold without a source", "b: type 'weedy'", "c: action_cover_pct", "'shared'"):
        assert fragment in problems


def test_load_errors_name_the_species(tmp_path):
    path = tmp_path / "weeds.yaml"
    path.write_text("waterhemp: {common_name: x, scientific_name: y, type: broadleaf, zero_tolerence: true}\n")
    with pytest.raises(ValueError, match="waterhemp"):
        WeedBook.load(path)
    path.write_text("waterhemp: {common_name: x, scientific_name: y, type: broadleaf, synonyms: common waterhemp}\n")
    with pytest.raises(ValueError, match="waterhemp"):
        WeedBook.load(path)
    path.write_text("")
    with pytest.raises(ValueError):
        WeedBook.load(path)


def test_contains_ignores_non_strings():
    book = WeedBook.load()
    assert "waterhemp" in book and ["waterhemp"] not in book and None not in book


def test_identify_maps_known_names_and_registers_others():
    book = WeedBook.load()
    assert book.identify("waterhemp") == "waterhemp"  # an id
    assert book.identify("Marestail") == "horseweed"  # a synonym
    assert book.identify("pigweed", "Amaranthus palmeri") == "palmer_amaranth"  # the scientific name decides
    sid = book.identify("Wild Buckwheat (Polygonum convolvulus)", weed_type="broadleaf")
    assert sid == "wild_buckwheat"
    info = book.get(sid)
    assert (info.common_name, info.scientific_name, info.type, info.catalog) == (
        "wild buckwheat", "Polygonum convolvulus", "broadleaf", False)
    assert info.action_density is None and info.action_cover_pct is None and not info.zero_tolerance
    assert book.identify("wild buckwheats") == book.identify("wildbuckwheat") == sid  # plurals, spacing
    assert book.identify("black bindweed", "Polygonum convolvulus") == sid  # the same plant by another name


@pytest.mark.parametrize("phrase, weed_type, expected", [
    ("small broadleaf seedling", None, "unidentified_broadleaf"),
    ("unknown grass", "grass", "unidentified_grass"),
    ("", "sedge", "unidentified_grass"),
    (None, None, "unidentified_broadleaf"),
])
def test_identify_a_description_is_an_unidentified_weed(phrase, weed_type, expected):
    assert WeedBook.load().identify(phrase, None, weed_type) == expected


@pytest.mark.parametrize("phrase", ["soybean", "Corn", "x9 weed", "a b c d e f", "<script>"])
def test_identify_refuses_names_that_are_no_weed(phrase):
    with pytest.raises(ValueError):
        WeedBook.load().identify(phrase)


def test_copy_keeps_registered_weeds_out_of_the_original():
    book = WeedBook.load()
    walk = book.copy()
    walk.register("burcucumber", "Sicyos angulatus")
    assert "burcucumber" in walk and "burcucumber" not in book
    assert "burcucumber" not in walk.catalog_ids() and "waterhemp" in walk.catalog_ids()
    assert walk.validate() == []


def test_register_returns_the_weed_already_known_by_that_name():
    book = WeedBook.load()
    first = book.register("Bur Cucumber", "Sicyos angulatus")
    assert book.register("burcucumber") == book.register("x", "Sicyos angulatus") == first == "bur_cucumber"
    assert book.get(first).common_name == "bur cucumber"
    assert book.register("Canada thistle") == "canada_thistle"  # a catalog weed is never registered twice
