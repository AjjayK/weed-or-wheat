import ssl

import pytest

from scout.labels.epa import download_pdf, epa_session, fetch_product, latest_pdf, parse_product

ITEM = {
    "eparegno": "524-549", "productname": "RD 1617 HERBICIDE",
    "active_ingredients": [{"pc_code": "103613", "active_ing": "Glyphosate, potassium salt", "active_ing_percent": 48.7}],
    "sites": [{"site": "CORN, FIELD"}], "pests": [{"pest": "KOCHIA"}], "types": [{"type": "HERBICIDE TERRESTRIAL"}],
    "pdffiles": [{"pdffile": "000524-00549-20200225.pdf"}, {"pdffile": "000524-00549-20230406.pdf"},
                 {"pdffile": "odd-name.pdf"}],
}


class _Response:
    def __init__(self, data=None, content=b""):
        self._data, self.content = data, content

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    def __init__(self, response):
        self.response, self.urls = response, []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.response


def test_parse_product():
    facts = parse_product(ITEM)
    assert facts.reg_no == "524-549" and facts.product_name == "RD 1617 HERBICIDE"
    assert facts.active_ingredients == ["Glyphosate, potassium salt"]
    assert facts.sites == ["CORN, FIELD"] and facts.pests == ["KOCHIA"]
    assert facts.is_herbicide and facts.latest_pdf == "000524-00549-20230406.pdf"


def test_is_herbicide_reads_ppls_types():
    def facts(*types):
        return parse_product({**ITEM, "types": [{"type": t} for t in types]})

    assert facts().is_herbicide  # some herbicides list no type at all
    assert not facts("FUNGICIDE", "INSECTICIDE").is_herbicide


def test_latest_pdf_handles_missing_dates():
    assert latest_pdf([{"pdffile": "odd.pdf"}]) is None
    assert latest_pdf([]) is None
    assert latest_pdf([{"pdffile": None}, {}]) is None
    assert parse_product({**ITEM, "pdffiles": [{"pdffile": None}]}).latest_pdf is None


def test_fetch_product_keeps_requested_number_when_ppls_lists_a_transfer(capsys):
    transferred = {**ITEM, "eparegno": "105211-77"}
    facts = fetch_product("524-549", _Session(_Response({"items": [transferred]})))
    assert facts.reg_no == "524-549" and facts.latest_pdf == "000524-00549-20230406.pdf"
    assert "105211-77" in capsys.readouterr().out


def test_fetch_product_is_quiet_for_a_current_product_and_warns_for_a_cancelled_one(capsys):
    assert fetch_product("524-549", _Session(_Response({"items": [ITEM]}))).reg_no == "524-549"
    assert capsys.readouterr().out == ""
    fetch_product("524-549", _Session(_Response({"items": [{**ITEM, "cancel_flag": "Yes"}]})))
    assert "cancel" in capsys.readouterr().out.lower()


def test_fetch_product_unknown_number():
    with pytest.raises(LookupError):
        fetch_product("1-1", _Session(_Response({"items": []})))


def test_download_pdf_saves_pdfs_and_refuses_anything_else(tmp_path):
    saved = download_pdf("a.pdf", tmp_path, _Session(_Response(content=b"%PDF-1.7 body")))
    assert saved.read_bytes() == b"%PDF-1.7 body"
    with pytest.raises(ValueError):
        download_pdf("b.pdf", tmp_path, _Session(_Response(content=b"<html>Service unavailable</html>")))
    assert not (tmp_path / "b.pdf").exists()
    session = _Session(_Response(content=b"%PDF-new"))
    assert download_pdf("a.pdf", tmp_path, session).read_bytes() == b"%PDF-1.7 body" and session.urls == []


def test_epa_session_retries_and_still_verifies_certificates():
    session = epa_session()
    for host in ("https://ordspub.epa.gov/ords/x", "https://www3.epa.gov/pesticides/x"):
        adapter = session.get_adapter(host)
        assert adapter.max_retries.total == 3 and 503 in adapter.max_retries.status_forcelist
        assert adapter.poolmanager.connection_pool_kw["ssl_context"].verify_mode == ssl.CERT_REQUIRED
    assert session.verify is True
