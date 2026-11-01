"""Build use_records.json from focus_products.csv.

Usage: python -m scout.labels.build_labels [--only REG_NO ...]
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from scout.agent.llm import Llm
from scout.agronomy.weeds import WeedBook
from scout.config import get_settings
from scout.labels.epa import download_pdf, epa_session, fetch_product
from scout.labels.extract import extract_use_records, page_texts, verify

HERE = Path(__file__).parent


def read_products(path: Path) -> dict[str, dict[str, str]]:
    """The CSV's rows by registration number, in file order, without blank cells. Optional columns: brand (the
    name on the jug when PPLS lists another, such as a master label's code name) and trait_<crop> (the seed trait
    the crop must carry for this product)."""
    with open(path, encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["reg_no"].strip()]
    return {row["reg_no"].strip(): {k: v.strip() for k, v in row.items() if k != "reg_no" and v and v.strip()}
            for row in rows}


def with_traits(records: list[dict], products: dict[str, dict[str, str]]) -> list[dict]:
    """Set every record's seed trait from the CSV, so editing a trait needs no new extraction."""
    return [{**r, "requires_trait": products.get(r["reg_no"], {}).get(f"trait_{r['crop']}")} for r in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    llm, weeds, session = Llm.from_settings(get_settings()), WeedBook.load(), epa_session()
    pdf_dir = HERE / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    out_path = HERE / "use_records.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    products = read_products(HERE / "focus_products.csv")
    reg_nos = [r for r in products if not args.only or r in args.only]
    records = [r for r in existing if r["reg_no"] not in reg_nos]
    for reg_no in reg_nos:
        previous = [r for r in existing if r["reg_no"] == reg_no]
        try:
            facts = fetch_product(reg_no, session)
            if not facts.is_herbicide or not facts.latest_pdf:
                print(f"{reg_no}: skipped (not a herbicide or no label PDF)")
                continue
            if "brand" in products[reg_no]:
                facts = replace(facts, product_name=products[reg_no]["brand"])
            pdf = download_pdf(facts.latest_pdf, pdf_dir, session)
            kept, dropped = verify(extract_use_records(pdf, facts, llm, weeds), page_texts(pdf))
            records += [r.model_dump() for r in kept]
            print(f"{reg_no} {facts.product_name}: kept {len(kept)}, dropped {len(dropped)}")
        except Exception as error:  # one bad product must not stop the batch, nor cost it its earlier records
            records += previous
            print(f"{reg_no}: failed ({error}); kept {len(previous)} earlier records")
    out_path.write_text(json.dumps(with_traits(records, products), indent=1), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
