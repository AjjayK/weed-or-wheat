"""Field step: rules decide, the label table supplies options, Muse Spark explains using only known numbers."""
import json
import re
from collections import Counter

from scout.agent.llm import user_message
from scout.agent.prompts import FIELD_INSTRUCTIONS
from scout.agronomy.rules import decide, urgency
from scout.agronomy.weeds import WeedBook
from scout.labels.lookup import find_label_options
from scout.models import FieldReport, LabelOption, StopResult, UseRecord

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
OUTCOME_WORDS = {"no_action": "No action needed", "scout_again": "Scout again in 5-7 days", "treat": "Treat"}


def _canon(number: str) -> str:
    return f"{float(number):g}"


def numbers_ok(text: str, allowed: str) -> bool:
    """True when every number in `text` also appears in `allowed`."""
    known = {_canon(n) for n in _NUMBER.findall(allowed)}
    return all(_canon(n) in known for n in _NUMBER.findall(text))


def _urgency_rank(text: str | None) -> int:
    if text is None:
        return 9
    return 0 if text.startswith("Beyond") else 1 if "~3 days" in text else 2


def template_explanation(decision: str, treat_mode: str | None, rule_trace: list[str]) -> str:
    head = OUTCOME_WORDS[decision] + (f" ({treat_mode})" if treat_mode else "") + "."
    return " ".join([head, *[line + "." for line in rule_trace[:4]], "Read the label before use."])


def build_report(walk_id: str, crop: str, stops: list[StopResult], weeds: WeedBook, records: list[UseRecord],
                 llm) -> FieldReport:
    d = decide(stops, weeds)
    stages = Counter(s.crop_stage for s in stops if s.status == "ok" and s.crop_stage)
    stage = stages.most_common(1)[0][0] if stages else None
    options: list[LabelOption] = []
    seen: set[tuple[str, str]] = set()
    urgency_text = None
    for summary in d.summaries:
        if summary.outcome != "treat":
            continue
        name = weeds.get(summary.species_id).common_name
        if summary.species_id.startswith("unidentified"):
            d.rule_trace.append(f"Identify the {name} to choose a product; labels name specific weeds")
            continue
        found = find_label_options(records, crop, summary.species_id, summary.max_height_in, stage)
        if not found:
            d.rule_trace.append(f"No product in the current label set covers {name} on {crop}")
        candidate = urgency(summary.max_height_in, found)
        if _urgency_rank(candidate) < _urgency_rank(urgency_text):
            urgency_text = candidate
        for option in found:
            key = (option.record.reg_no, option.record.species_id)  # one product can cover several treated weeds
            if key not in seen:
                seen.add(key)
                options.append(option)
    payload = json.dumps({
        "crop": crop, "decision": d.decision, "treat_mode": d.treat_mode, "urgency": urgency_text,
        "species": [s.model_dump() for s in d.summaries], "rule_trace": d.rule_trace,
        "options": [{"product": o.record.product_name, "groups": o.record.herbicide_groups,
                     "max_weed_height_in": o.record.max_weed_height_in, "over_size_limit": o.over_size_limit}
                    for o in options],
    }, ensure_ascii=False)
    explanation = None
    for _ in range(2):
        try:
            text = llm.turn([user_message(payload)], instructions=FIELD_INSTRUCTIONS).text.strip()
        except Exception:  # the template below is a complete fallback
            break
        if text and numbers_ok(text, payload):
            explanation = text
            break
    return FieldReport(
        walk_id=walk_id, crop=crop, stops=stops, species_summary=d.summaries, decision=d.decision,
        treat_mode=d.treat_mode, urgency=urgency_text, rule_trace=d.rule_trace,
        explanation=explanation or template_explanation(d.decision, d.treat_mode, d.rule_trace), options=options,
    )
