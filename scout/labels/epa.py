"""EPA Pesticide Product Label System (PPLS): product facts and the latest label PDF."""
import re
import ssl
from dataclasses import dataclass, replace
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

PPLS_URL = "https://ordspub.epa.gov/ords/pesticides/cswu/ppls/{reg_no}"
PDF_BASE = "https://www3.epa.gov/pesticides/chem_search/ppls/"
OP_LEGACY_SERVER_CONNECT = 0x4
_PDF_DATE = re.compile(r"-(\d{8})\.pdf$", re.IGNORECASE)


class LegacyTLSAdapter(HTTPAdapter):
    """Keeps certificate checks on, but allows the legacy renegotiation some EPA hosts need under OpenSSL 3."""

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= OP_LEGACY_SERVER_CONNECT
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def epa_session() -> requests.Session:
    retry = Retry(total=3, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.mount("https://ordspub.epa.gov", LegacyTLSAdapter(max_retries=retry))
    session.mount("https://www3.epa.gov", LegacyTLSAdapter(max_retries=retry))
    return session


@dataclass
class ProductFacts:
    reg_no: str
    product_name: str
    active_ingredients: list[str]
    sites: list[str]
    pests: list[str]
    types: list[str]
    latest_pdf: str | None

    @property
    def is_herbicide(self) -> bool:
        """PPLS types read like 'HERBICIDE TERRESTRIAL', and some herbicides list no type at all."""
        return not self.types or any("HERBICIDE" in t.upper() for t in self.types)


def latest_pdf(pdffiles: list[dict]) -> str | None:
    dated = [(m.group(1), f["pdffile"]) for f in pdffiles if (m := _PDF_DATE.search(f.get("pdffile") or ""))]
    return max(dated)[1] if dated else None


def parse_product(item: dict) -> ProductFacts:
    return ProductFacts(
        reg_no=item["eparegno"],
        product_name=item["productname"],
        active_ingredients=[a["active_ing"] for a in item.get("active_ingredients") or []],
        sites=[s["site"] for s in item.get("sites") or []],
        pests=[p["pest"] for p in item.get("pests") or []],
        types=[t["type"] for t in item.get("types") or []],
        latest_pdf=latest_pdf(item.get("pdffiles") or []),
    )


def fetch_product(reg_no: str, session: requests.Session) -> ProductFacts:
    response = session.get(PPLS_URL.format(reg_no=reg_no), timeout=60)
    response.raise_for_status()
    items = response.json().get("items") or []
    if not items:
        raise LookupError(f"EPA has no product {reg_no}")
    facts = parse_product(items[0])
    if facts.reg_no != reg_no:
        print(f"{reg_no}: PPLS now lists this product as {facts.reg_no}; keeping {reg_no}, whose label PDFs these are")
        facts = replace(facts, reg_no=reg_no)
    if items[0].get("cancel_flag") == "Yes":
        print(f"{reg_no}: warning, EPA lists this product as cancelled")
    return facts


def download_pdf(filename: str, dest_dir: Path, session: requests.Session) -> Path:
    path = Path(dest_dir) / filename
    if not path.exists():
        response = session.get(PDF_BASE + filename, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"{filename} is not a PDF (an error page, perhaps)")
        path.write_bytes(response.content)
    return path
