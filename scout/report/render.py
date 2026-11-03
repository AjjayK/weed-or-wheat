"""Report page (HTML) and per-stop CSV."""
import csv
import io
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from scout.agronomy.weeds import WeedBook
from scout.labels.epa import PDF_BASE
from scout.models import FieldReport, StopResult

_ENV = Environment(loader=FileSystemLoader(Path(__file__).with_name("templates")),
                   autoescape=select_autoescape(["html"]))
OUTCOME_TEXT = {"no_action": "No action", "scout_again": "Scout again in 5–7 days", "treat": "Treat"}
CSV_COLUMNS = ["stop", "status", "view_type", "weed_cover_pct", "crop_cover_pct", "weed_count", "count_approximate",
               "density_per_m2", "in_row_weed_share", "species", "max_height_in", "flags"]


def _blank(value) -> object:
    return "" if value is None else value


def decision_text(report: FieldReport) -> str:
    """The decision as people read it, e.g. "Treat (spot)"."""
    return OUTCOME_TEXT[report.decision] + (f" ({report.treat_mode})" if report.treat_mode else "")


def report_html(report: FieldReport, weeds: WeedBook, overlay_url: Callable[[int], str],
                step_url: Callable[[str], str] | None = None) -> str:
    ok = [s for s in report.stops if s.status == "ok"]
    measured = [s for s in ok if s.view_type != "close_up"] or ok  # the stops rules.py averages
    densities = [s.density_per_m2 for s in ok if s.density_per_m2 is not None]
    names = {s.species_id: weeds.get(s.species_id).common_name if s.species_id in weeds else s.species_id
             for s in report.species_summary}
    return _ENV.get_template("report.html").render(
        report=report,
        mean_cover=sum(s.weed_cover_pct for s in measured) / len(measured) if measured else 0.0,
        density_text=f"{sum(densities) / len(densities):.1f} /m²" if densities else "—",
        stops_with_weeds=sum(1 for s in ok if s.weed_count),
        ok_count=len(ok),
        decision_text=decision_text(report),
        names=names,
        outcome_text=OUTCOME_TEXT,
        overlay_url=overlay_url,
        step_url=step_url or (lambda name: None),
        pdf_base=PDF_BASE,
    )


def _csv_row(s: StopResult) -> list:
    flags = " | ".join(s.flags)
    if s.status == "failed":  # the zeros of a failed stop are placeholders, not measurements
        return [s.stop_index + 1, s.status] + [""] * (len(CSV_COLUMNS) - 3) + [flags]
    return [s.stop_index + 1, s.status, s.view_type, s.weed_cover_pct, s.crop_cover_pct, s.weed_count,
            s.count_approximate, _blank(s.density_per_m2), _blank(s.in_row_weed_share),
            ";".join(f"{k}:{v}" for k, v in s.species.items()), _blank(s.max_height_in), flags]


def report_csv(report: FieldReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for s in report.stops:
        writer.writerow(_csv_row(s))
    return buf.getvalue()
