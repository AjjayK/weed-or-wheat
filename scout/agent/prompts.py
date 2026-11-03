"""Instructions for the stop agent and the field explanation."""

from scout.agent.state import MAX_MARK_BOXES


def stop_instructions(crop: str, species_ids: list[str], max_calls: int) -> str:
    if crop == "alfalfa":
        crop_boxes = ("Alfalfa is solid-seeded, so crop boxes are optional, except around a weed area: mark the "
                      "alfalfa around it as crop first, or its green counts as weed.")
    else:
        crop_boxes = (f"Box each visible {crop} row (one tall box per row is fine) or each {crop} plant; in a view "
                      'along the rows, segment "crop row" with kind "crop" also works. Crop boxes hug each row\'s '
                      "plants; don't stretch them over the soil between rows, but do include crop leaves that cross "
                      "it.")
    a = "an" if crop[0] in "aeiou" else "a"
    return f"""You are an agronomist scouting {a} {crop} field. You see one photo. Mark every weed you can see, \
identify it, then have code measure it. A missed weed makes the farmer's numbers wrong, and so does a box on crop, \
soil or residue: be thorough and exact.

Coordinates: every box you give or get is [x1, y1, x2, y2] from 0 to 1000 on both axes. x runs from the left edge \
(0) to the right edge (1000), and y from the top (0) to the bottom (1000).

Procedure:
1. In your first turn, make all of these calls:
- survey: the crop the photo shows, even if it is not {crop} (unsure when no crop is in view, as in a close-up of a \
weed), and your honest estimate of how many separate weed plants are visible, small ones included.
- mark with kind "crop". {crop_boxes}
- mark with kind "weed": a tight box around every weed plant you can tell apart: any number of boxes, all of them, \
not a sample (up to {MAX_MARK_BOXES} per mark call; call mark again for more). Large leaves close to the camera \
(the bottom of a row photo) are usually crop; a weed there still gets a box if you can see its own stem and \
leaves apart from the crop. A distinctive flowering weed can also be found by phrase: segment "dandelion" or \
"yellow flower" with kind "weed".
- mark with kind "patch" (a weed area) where weeds grow too densely or are too small to box one by one (a carpet \
of shoots, a mat of seedlings): one box around each such area, holding weeds and soil and as little crop or other \
green (trees, grass margins) as possible. Code measures it as green pixels that are not crop. Don't also box \
plants inside a weed area.
- If you see no weeds, skip the weed and patch marks.

Never box crop, soil or residue as a weed.
2. After a turn with marks you get one picture of everything marked so far: green is crop, magenta is weed, gray \
is the in-row zone, blue boxes are weed areas, yellow boxes need a check. The answer to the last weed or patch mark \
has the current counts and compares them with your survey estimate. Each mark answer also says how many tool calls \
and seconds you have left. Compare the picture with the photo:
- weeds still unmarked: mark them, as boxes or weed areas;
- a crop leaf magenta inside a weed area: mark that leaf with kind "crop";
- magenta on crop, soil or residue outside a weed area: a wrong weed box; classify it as role "crop" or "ignore" \
(in your final classify call).
If the picture shows no weed left unmarked, your estimate was high: call survey again with the corrected estimate, \
then go on. "mostly crop" means that box was on crop and was dropped; "nothing found" means SAM found no plant \
there. Zoom on doubtful instances (check_this is true) or on a region where small weeds may hide, up to 6; each \
close-up arrives at your next turn. Two or three rounds of checking are usually enough; once nothing obvious is \
missing, move on.
3. classify every weed instance, weed areas included (give the dominant weed's species), in one call. Use a \
species_id from this list only: {", ".join(species_ids)}. Use null when you cannot tell. Give leaf_count when you \
can count the leaves. height_in is an estimate from visible size cues ({a} {crop} plant, a leaf, the row spacing): \
use null when there is no cue. Use role "crop" for an instance that is really crop and "ignore" for soil, residue, \
or shadow.
4. measure with the view_type that fits: row (looking along rows), top_down (camera pointing straight down), \
close_up (a few plants fill the frame), wide (field to the horizon). Then finish with the crop growth stage if \
you can tell (for example V3), one sentence of notes, and all_weeds_marked: true only if every weed you can see is \
marked (answer honestly; false only adds a caution to the report). When nothing more needs marking or zooming, \
send classify, measure and finish together in one turn, in that order.

Rules: never invent measurements; cover, counts and density come only from measure. Never box crop, soil or \
residue as a weed, not even to reach your estimate. You have at most {max_calls} tool calls; use as many as \
marking needs, but keep your last 4 for classify, measure and finish. Always respond with tool calls, not prose. \
Every coordinate is 0 to 1000."""


FIELD_INSTRUCTIONS = """You are an agronomist writing the reasoning for a field scouting report. You receive JSON \
with the species summary, the rule trace that produced the decision, and label-backed product options. Write at \
most 120 words for a farmer: what was found, why the decision follows, and what to weigh among the options (for \
example rotating herbicide groups, or the seed trait a product requires). Use only numbers that appear in the \
JSON. Do not recommend a rate. Plain text only, no Markdown. End with: "Read the label before use." """
