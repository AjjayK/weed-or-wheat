"""Read a label PDF with Muse Spark into UseRecords, then keep only records whose quote is really on the page."""
import base64
import json
import re
from pathlib import Path

from pydantic import ValidationError
from pypdf import PdfReader

from scout.agent.llm import user_message
from scout.agronomy.weeds import WeedBook
from scout.labels.epa import ProductFacts
from scout.models import UseRecord

MIN_QUOTE_CHARS = 15


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
Extract one record per (crop, weed, timing) for these crops only: corn (field corn), soybean, alfalfa.
species_id must be one of these ids; map label weed names to them and skip weeds that do not map:
{species}
For each record give: the maximum weed height in inches the label allows (null if none); crop stage limits such as \
V2 or V8 (null if none); the rate range and unit as printed; the pre-harvest interval in days; the restricted-entry \
interval in hours; the herbicide group number(s) printed on the label.
page is the 1-based page index within this PDF file where the record's facts appear. quote is a verbatim snippet \
of 15 to 200 characters copied from that page that supports the record.
Return only records the label clearly supports."""


def pdf_data_url(path: Path) -> str:
    return "data:application/pdf;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def page_texts(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _norm(text: str) -> str:
    text = text.lower()
    for old, new in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def verify(records: list[UseRecord], texts: list[str]) -> tuple[list[UseRecord], list[UseRecord]]:
    """Keep a record only if its quote appears on its page (or the page next to it, which then becomes its page)."""
    normed = [_norm(t) for t in texts]
    kept, dropped = [], []
    for record in records:
        quote = _norm(record.quote)
        found = None
        if len(quote) >= MIN_QUOTE_CHARS:
            for page in (record.page, record.page - 1, record.page + 1):
                if 1 <= page <= len(normed) and quote in normed[page - 1]:
                    found = page
                    break
        if found is None:
            dropped.append(record)
        else:
            kept.append(record.model_copy(update={"page": found}))
    return kept, dropped


def extract_use_records(pdf_path: Path, facts: ProductFacts, llm, weeds: WeedBook) -> list[UseRecord]:
    species = [sid for sid in weeds.ids() if not sid.startswith("unidentified")]
    listing = "\n".join(f"- {sid}: {', '.join([weeds.get(sid).common_name, *weeds.get(sid).synonyms])}"
                        for sid in species)
    prompt = EXTRACT_PROMPT.format(product=facts.product_name, reg_no=facts.reg_no, species=listing)
    turn = llm.turn([user_message(prompt, file=(Path(pdf_path).name, pdf_data_url(pdf_path)))],
                    text_format=USE_RECORDS_FORMAT)
    records = []
    for item in json.loads(turn.text)["records"]:
        sid = item["species_id"] if item["species_id"] in weeds else weeds.resolve(item["species_id"])
        if sid is None or sid.startswith("unidentified"):
            continue
        try:
            records.append(UseRecord(reg_no=facts.reg_no, product_name=facts.product_name,
                                     active_ingredients=facts.active_ingredients, pdf_file=facts.latest_pdf,
                                     **{**item, "species_id": sid}))
        except ValidationError:
            continue
    return records
