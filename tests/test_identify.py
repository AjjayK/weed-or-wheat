import json
import threading
import time
from functools import partial

import pytest

import scout.agent.tools as tools
from scout.agent.identify import ID_FORMAT, Vote, identify_kinds, settled, tally
from scout.agent.llm import LlmTurn
from scout.agent.state import StopState
from scout.agent.tools import ToolRunner
from tests.helpers import FakeSam, make_image, small_weedbook


@pytest.fixture(autouse=True)
def calls_answer_in_order(monkeypatch):
    """The identify tool's calls answer one at a time, in the order of the answer lists below: the panel stops once
    its votes settle the winner, so which votes count must not depend on which thread runs first."""
    monkeypatch.setattr(tools, "identify_kinds", partial(identify_kinds, workers=1))


def _answer(name, scientific=None, weed_type="broadleaf", confidence="high"):
    return json.dumps({"features": [f"{name} leaf"], "look_alikes": [], "common_name": name,
                       "scientific_name": scientific, "weed_type": weed_type, "confidence": confidence})


class PanelLlm:
    """Answers identification calls from a list, in turn, from any thread; an exception in the list is raised."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        self._lock = threading.Lock()

    def turn(self, input, **kwargs):
        with self._lock:
            self.requests.append({"input": input, **kwargs})
            answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return LlmTurn("r", [], answer)


def _vote(name, scientific=None, confidence="high", weed_type="broadleaf"):
    return Vote(name, scientific, weed_type, confidence, [f"{name} leaf"])


def test_the_majority_names_the_weed_however_it_is_written():
    book = small_weedbook()
    verdict = tally([_vote("burcucumber", "Sicyos angulatus"), _vote("Bur cucumber", "Sicyos angulatus"),
                     _vote("bur-cucumber"), _vote("tall morningglory", "Ipomoea purpurea", "medium"),
                     _vote("honeyvine milkweed", "Cynanchum laeve", "medium")], book)
    assert (verdict.votes, verdict.answered, verdict.confidence) == (3, 5, "medium")
    assert verdict.species_id == "burcucumber" and book.get("burcucumber").scientific_name == "Sicyos angulatus"
    assert verdict.others == ["tall morningglory (1)", "honeyvine milkweed (1)"]
    assert verdict.features == ["burcucumber leaf"]


def test_known_species_vote_together_by_synonym_and_scientific_name():
    verdict = tally([_vote("marestail"), _vote("horseweed"), _vote("Canada fleabane", "Conyza canadensis"),
                     _vote("waterhemp")], small_weedbook())
    assert (verdict.species_id, verdict.votes, verdict.confidence) == ("horseweed", 3, "medium")


def test_a_unanimous_sure_panel_is_high_confidence_and_a_split_one_low():
    book = small_weedbook()
    assert tally([_vote("waterhemp")] * 5, book).confidence == "high"
    split = tally([_vote("waterhemp"), _vote("horseweed", confidence="low"), _vote(None)], book)
    assert split.confidence == "low" and split.species_id == "waterhemp"  # a tie goes to the surer voters


def test_votes_that_name_nothing_make_an_unidentified_weed_by_type():
    verdict = tally([_vote(None, weed_type="grass"), _vote("grass seedling", weed_type="grass"),
                     _vote("giant foxtail", weed_type="grass", confidence="low")], small_weedbook())
    assert verdict.species_id == "unidentified_grass" and verdict.common_name is None and verdict.votes == 2


def test_every_kind_is_asked_in_parallel_and_failed_calls_cost_only_their_vote():
    image = make_image(200, 200)
    llm = PanelLlm([_answer("waterhemp"), RuntimeError("timeout"), _answer("waterhemp"),
                    RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    verdicts = identify_kinds(llm, "corn", [[image.work], [image.work, image.work]], small_weedbook(), samples=3,
                              effort="high")
    assert len(llm.requests) == 6 and all(r["text_format"] == ID_FORMAT and r["effort"] == "high"
                                          for r in llm.requests)
    assert {len(r["input"][0]["content"]) for r in llm.requests} <= {2, 3}  # the prompt and each kind's close-ups
    named = [v for v in verdicts if v is not None]
    assert len(named) == 1 and named[0].species_id == "waterhemp" and named[0].answered == 2
    assert verdicts.count(None) == 1  # every call for that kind failed


def test_the_prompt_asks_for_features_before_the_name_and_allows_any_species():
    llm = PanelLlm([_answer("waterhemp")])
    identify_kinds(llm, "soybean", [[make_image().work]], small_weedbook(), samples=1, effort=None)
    text = llm.requests[0]["input"][0]["content"][0]["text"]
    assert "soybean field" in text and "First describe" in text and "any weed species may be the answer" in text
    assert ID_FORMAT["schema"]["required"][:2] == ["features", "look_alikes"]  # reasons come before the name


def _runner(llm, samples=3):
    state = StopState(make_image(100, 100), "corn")
    runner = ToolRunner(state, FakeSam(), small_weedbook(), llm=llm, id_samples=samples, id_effort="high")
    runner.run("mark", {"kind": "weed", "boxes": [{"box": [600, 600, 700, 700], "label": "vine"},
                                                  {"box": [800, 100, 900, 200], "label": "vine"}]})
    return runner, state


def test_identify_tool_returns_the_verdict_and_classify_falls_back_on_it():
    llm = PanelLlm([_answer("burcucumber", "Sicyos angulatus"), _answer("velvetleaf"),
                    _answer("burcucumber", "Sicyos angulatus")])  # unsettled until the last vote
    runner, state = _runner(llm)
    out = json.loads(runner.run("identify", {"groups": [{"instance_ids": ["s0-i1", "s0-i2"], "label": "vine"}]})
                     .output)
    group = out["identified"][0]
    assert group["species_id"] == "burcucumber" and group["votes"] == "2 of 3"
    assert group["other_names"] == ["velvetleaf (1)"] and group["scientific_name"] == "Sicyos angulatus"
    step = state.steps[-1]
    assert step.tool == "identify" and step.summary == "vine: burcucumber, 2 of 3 votes; also velvetleaf (1)"
    runner.run("classify", {"items": [
        {"instance_id": "s0-i1", "role": "weed", "species_id": None, "id_confidence": "medium"},
        {"instance_id": "s0-i2", "role": "weed", "species_id": "waterhemp", "id_confidence": "high"}]})
    assert state.instances["s0-i1"].species_id == "burcucumber"  # no name of its own: the panel's stands
    assert state.instances["s0-i2"].species_id == "waterhemp"  # the agent's own name wins


@pytest.mark.parametrize("groups, message", [
    ([], "1 to 6 groups"),
    ([{"instance_ids": [], "label": "x"}], "list the instance_ids of this kind"),
    ([{"instance_ids": ["s0-i9"], "label": "x"}], "unknown instance s0-i9"),
])
def test_identify_tool_rejects_bad_groups(groups, message):
    runner, _ = _runner(PanelLlm([]))
    out = runner.run("identify", {"groups": groups}).output
    assert out.startswith("error:") and message in out


def test_identify_tool_without_a_panel_says_to_name_the_weeds_yourself():
    state = StopState(make_image(100, 100), "corn")
    out = ToolRunner(state, FakeSam(), small_weedbook()).run("identify", {"groups": [{"instance_ids": ["x"]}]})
    assert "name the weeds in classify yourself" in out.output


def test_cannot_tell_answers_do_not_outvote_a_species_the_panel_named():
    verdict = tally([_vote(None), _vote(None), _vote("unknown broadleaf"), _vote("burcucumber", "Sicyos angulatus"),
                     _vote("burcucumber")], small_weedbook())
    assert (verdict.species_id, verdict.votes, verdict.answered, verdict.confidence) == ("burcucumber", 2, 5, "low")
    assert verdict.others == ["unidentified (3)"]
    lone = tally([_vote(None), _vote(None), _vote(None), _vote(None), _vote("velvetleaf")], small_weedbook())
    assert lone.species_id == "unidentified_broadleaf"  # one name in five is too few


def test_small_close_ups_are_enlarged_and_big_ones_shrunk():
    from PIL import Image

    from scout.agent.identify import ID_MAX_PX, ID_MIN_PX, fit
    assert max(fit(Image.new("RGB", (200, 100))).size) == ID_MIN_PX
    assert fit(Image.new("RGB", (4000, 3000))).size == (ID_MAX_PX, 768)
    assert fit(Image.new("RGB", (900, 600))).size == (900, 600)


def test_identify_sends_each_plant_up_close_and_in_its_surroundings():
    llm = PanelLlm([_answer("waterhemp")] * 3)
    runner, _ = _runner(llm)
    runner.run("identify", {"groups": [{"instance_ids": ["s0-i1"], "label": "pigweed"}]})
    images = [part for part in llm.requests[0]["input"][0]["content"] if part["type"] == "input_image"]
    assert len(images) == 2 and "a wider view of the plant" in llm.requests[0]["input"][0]["content"][0]["text"]


def test_identify_shows_the_first_three_plants_and_names_every_one_listed():
    from scout.agent.state import StopState
    llm = PanelLlm([_answer("waterhemp")] * 3)
    state = StopState(make_image(100, 100), "corn")
    runner = ToolRunner(state, FakeSam(), small_weedbook(), llm=llm, id_samples=3)
    boxes = [[600, 600, 700, 700], [800, 100, 900, 200], [260, 400, 360, 500], [400, 750, 1000, 1000]]
    runner.run("mark", {"kind": "weed", "boxes": [{"box": b, "label": "pigweed"} for b in boxes]})
    ids = list(state.instances)
    runner.run("identify", {"groups": [{"instance_ids": ids, "label": "pigweed"}]})
    images = [p for p in llm.requests[0]["input"][0]["content"] if p["type"] == "input_image"]
    assert len(images) == 6  # three plants, two views each
    runner.run("classify", {"items": [{"instance_id": i, "role": "weed", "species_id": None, "id_confidence": "medium"}
                                      for i in ids]})
    assert {state.instances[i].species_id for i in ids} == {"waterhemp"}


def test_identify_shows_the_largest_plants_whatever_order_they_are_listed_in(monkeypatch):
    llm = PanelLlm([_answer("waterhemp")] * 3)
    state = StopState(make_image(100, 100), "corn")
    runner = ToolRunner(state, FakeSam(), small_weedbook(), llm=llm, id_samples=3)
    boxes = [[600, 600, 700, 700], [800, 100, 900, 200], [400, 750, 1000, 1000], [260, 400, 360, 500]]
    runner.run("mark", {"kind": "weed", "boxes": [{"box": b, "label": "pigweed"} for b in boxes]})
    shown = []
    real = runner._close_up
    monkeypatch.setattr(runner, "_close_up", lambda box, margin=0.15: shown.append(box) or real(box, margin))
    ids = list(state.instances)
    runner.run("identify", {"groups": [{"instance_ids": ids, "label": "pigweed"}]})
    biggest = max(ids, key=lambda i: state.instances[i].area_px)
    assert shown[0] == state.instances[biggest].box and len(shown) == 6


def test_votes_too_slow_for_the_panel_are_not_counted():
    import time

    class SlowPanel(PanelLlm):
        def turn(self, input, **kwargs):
            with self._lock:
                n = len(self.requests)
                self.requests.append(kwargs)
            if n >= 2:
                time.sleep(1.0)
            return LlmTurn("r", [], _answer("waterhemp" if n < 2 else "horseweed"))

    began = time.monotonic()
    verdicts = identify_kinds(SlowPanel([]), "corn", [[make_image().work]], small_weedbook(), samples=5,
                              effort=None, workers=5, timeout_s=0.3)
    assert time.monotonic() - began < 0.9
    assert verdicts[0].species_id == "waterhemp" and verdicts[0].answered == 2


def test_a_panel_is_settled_once_no_vote_still_out_could_change_the_winner():
    book = small_weedbook()
    bc, vl, unsure = _vote("burcucumber", "Sicyos angulatus"), _vote("velvetleaf"), _vote(None)
    assert settled([bc] * 4, 3, book)  # 4 of 7: the other 3 can't pass it
    assert not settled([bc] * 3 + [vl], 3, book)  # velvetleaf could still reach 4
    assert settled([bc] * 4 + [vl], 2, book)
    assert not settled([bc] * 2 + [unsure] * 2, 3, book)  # 3 more for one species would pass both
    assert settled([unsure] * 5, 2, book)  # 2 votes can't lift a species to the 40% bar (2.8 of 7)
    assert not settled([unsure] * 4, 3, book)  # 3 votes for one species would beat the abstentions
    assert settled([], 0, book) and not settled([], 3, book) and settled([vl, bc], 0, book)


def test_the_panel_stops_waiting_once_the_votes_settle_it():
    release = threading.Event()

    class StuckPanel(PanelLlm):
        def turn(self, input, **kwargs):
            with self._lock:
                n = len(self.requests)
                self.requests.append(kwargs)
            if n >= 4:
                release.wait(5)  # calls still out: the panel must not wait for these
            return LlmTurn("r", [], _answer("burcucumber", "Sicyos angulatus"))

    began = time.monotonic()
    try:
        verdicts = identify_kinds(StuckPanel([]), "corn", [[make_image().work]], small_weedbook(), samples=7,
                                  effort=None, timeout_s=10)
    finally:
        release.set()
    assert time.monotonic() - began < 2
    assert (verdicts[0].species_id, verdicts[0].votes, verdicts[0].answered) == ("burcucumber", 4, 4)


def test_a_weed_left_unclassified_when_time_ran_out_keeps_the_panels_name():
    from scout.agent.stop_agent import finalize
    runner, state = _runner(PanelLlm([_answer("burcucumber", "Sicyos angulatus")] * 3))
    runner.run("identify", {"groups": [{"instance_ids": ["s0-i1", "s0-i2"], "label": "vine"}]})
    result = finalize(state, FakeSam())
    assert result.species == {"burcucumber": 2}
    assert "Check s0-i1: marked but never classified" in result.flags


def test_a_name_from_a_split_panel_stays_flagged_for_checking():
    llm = PanelLlm([_answer("redroot pigweed"), _answer("redroot pigweed"), _answer(None), _answer("waterhemp"),
                    _answer(None)])  # unsettled until the last vote
    runner, state = _runner(llm, samples=5)
    out = json.loads(runner.run("identify", {"groups": [{"instance_ids": ["s0-i1", "s0-i2"], "label": "pigweed"}]})
                     .output)["identified"][0]
    assert (out["votes"], out["confidence"]) == ("2 of 5", "low")
    runner.run("classify", {"items": [
        {"instance_id": "s0-i1", "role": "weed", "species_id": "redroot pigweed", "id_confidence": "high"},
        {"instance_id": "s0-i2", "role": "weed", "species_id": "waterhemp", "id_confidence": "high"}]})
    first, second = state.instances["s0-i1"], state.instances["s0-i2"]
    assert (first.id_confidence, first.check_this) == ("low", True)  # the panel's doubt carries over
    assert (second.id_confidence, second.check_this) == ("high", False)  # the agent's own different name is its own
