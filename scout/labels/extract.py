"""Read a label PDF with Muse Spark into UseRecords, then keep only records whose quote is really on the page."""
import base64
import json
import re
import unicodedata
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from scout.agent.llm import user_message
from scout.agronomy.weeds import WeedBook
from scout.labels.epa import ProductFacts
from scout.models import UseRecord

MIN_QUOTE_KEY_CHARS = 12  # letters and digits left in a quote after _key
SITE_PATTERN = {  # as PPLS sites name the crops; whole words, so ACORN SQUASH and CORN SALAD are not corn
    "corn": re.compile(r"\bCORN\b(?! SALAD)"), "soybean": re.compile(r"\bSOYBEANS?\b"),
    "alfalfa": re.compile(r"\bALFALFA\b")}


def _nullable(kind: str) -> dict:
    return {"type": [kind, "null"]}


_RECORD_PROPERTIES = {
    "crop": {"type": "string", "enum": ["corn", "soybean", "alfalfa"]},
    "species_id": {"type": "string"},
    "weed_label_text": {"type": "string"},
    "timing": {"type": "string", "enum": ["preemergence", "postemergence", "burndown"]},
    "herbicide_groups": {"type": "array", "items": {"type": "string"}},
    "max_weed_height_in": _nullable("number"),
    "crop_stage_min": _nullable("string"),
    "crop_stage_max": _nullable("string"),
    "rate_min": _nullable("number"),
    "rate_max": _nullable("number"),
    "rate_unit": _nullable("string"),
    "phi_days": _nullable("integer"),
    "rei_hours": _nullable("integer"),
    "page": {"type": "integer"},
    "quote": {"type": "string"},
}
USE_RECORDS_FORMAT = {
    "type": "json_schema",
    "name": "use_records",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"records": {"type": "array", "items": {
            "type": "object", "properties": _RECORD_PROPERTIES,
            "required": list(_RECORD_PROPERTIES), "additionalProperties": False}}},
        "required": ["records"],
        "additionalProperties": False,
    },
}

EXTRACT_PROMPT = """You are reading the EPA-registered label for {product} (EPA Reg. No. {reg_no}).
Extract one record per weed for in-crop postemergence uses on {crop} only (for corn: field corn). When the {crop} \
directions refer to a table of weeds controlled elsewhere in the label, take the weeds from that table.
species_id must be one of these ids; map label weed names to them and skip weeds that do not map:
{species}
Skip weeds the label rates only as partially controlled or suppressed by this product used alone.
For each record give: the maximum weed height in inches the label allows (null if none); crop stage limits such as \
V2 or V8 (null if none); the rate range and unit as printed; the pre-harvest interval in days; the restricted-entry \
interval in hours; the herbicide group number(s) printed on the label.
page is the 1-based page index within this PDF file where the record's facts appear. quote is a verbatim snippet \
of 15 to 200 characters copied from that page that supports the record. Copy the quote character for character \
from one place on the page: no paraphrase, no ellipses.
Return only records the label clearly supports."""


def pdf_data_url(path: Path) -> str:
    return "data:application/pdf;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def page_texts(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _key(text: str) -> str:
    """Letters and digits only, so pypdf noise (line breaks, hyphenation, kerning spaces, ®, quote styles,
    ligatures) cannot break a match. Separators between digits survive, so 1.5, 15, 1/2 and 7-10 stay distinct."""
    # NFKC writes ½ as 1, the fraction slash U+2044, 2; the slash is made plain here
    text = unicodedata.normalize("NFKC", text).replace("⁄", "/").lower()
    text = re.sub(r"(?<=\d)\s*([./])\s*(?=\d)", r"\1", text)
    text = re.sub(r"(?<=\d)[^a-z0-9./]+(?=\d)", "_", text)
    return re.sub(r"[^a-z0-9./_]+|(?<!\d)[./]|[./](?!\d)", "", text)


def verify(records: list[UseRecord], texts: list[str]) -> tuple[list[UseRecord], list[UseRecord]]:
    """Keep a record only if its quote is on some page of the PDF. Its page becomes the nearest page holding the
    quote: models often cite the printed page number, and EPA PDFs open with a 1-3 page acceptance letter."""
    pages = [_key(t) for t in texts]
    kept, dropped = [], []
    for record in records:
        quote = _key(record.quote)
        hits = ([n for n, page in enumerate(pages, start=1) if quote in page]
                if len(quote) >= MIN_QUOTE_KEY_CHARS else [])
        if hits:
            kept.append(record.model_copy(update={"page": min(hits, key=lambda n: abs(n - record.page))}))
        else:
            dropped.append(record)
    return kept, dropped


def extract_use_records(pdf_path: Path, facts: ProductFacts, llm, weeds: WeedBook) -> list[UseRecord]:
    species = [sid for sid in weeds.ids() if not sid.startswith("unidentified")]
    listing = "\n".join(f"- {sid}: {', '.join([weeds.get(sid).common_name, *weeds.get(sid).synonyms])}"
                        for sid in species)
    file = (Path(pdf_path).name, pdf_data_url(pdf_path))
    crops = [crop for crop, pattern in SITE_PATTERN.items()
             if not facts.sites or any(pattern.search(s.upper()) for s in facts.sites)]
    records = []
    for crop in crops:  # one smaller answer per crop: a single answer for a glyphosate label can run past the timeout
        prompt = EXTRACT_PROMPT.format(product=facts.product_name, reg_no=facts.reg_no, species=listing, crop=crop)
        turn = llm.turn([user_message(prompt, file=file)], text_format=USE_RECORDS_FORMAT)
        for item in json.loads(turn.text)["records"]:
            sid = item["species_id"] if item["species_id"] in weeds else weeds.resolve(item["species_id"])
            if sid is None or sid.startswith("unidentified") or item.get("crop") != crop:
                continue
            groups = sorted({d for g in item["herbicide_groups"] for d in re.findall(r"\d+", g)})
            try:
                records.append(UseRecord(reg_no=facts.reg_no, product_name=facts.product_name,
                                         active_ingredients=facts.active_ingredients, pdf_file=facts.latest_pdf,
                                         **{**item, "species_id": sid, "herbicide_groups": groups}))
            except ValidationError:
                continue
    return records
