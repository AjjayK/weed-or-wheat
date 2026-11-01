from scout.labels.lookup import find_label_options, load_records, stage_index
from scout.models import UseRecord


def _r(reg, groups, limit=None, stage_min=None, stage_max=None, species="waterhemp", crop="soybean",
       timing="postemergence"):
    return UseRecord(reg_no=reg, product_name=f"P{reg}", active_ingredients=[], herbicide_groups=groups, crop=crop,
                     species_id=species, weed_label_text=species, timing=timing, max_weed_height_in=limit,
                     crop_stage_min=stage_min, crop_stage_max=stage_max, page=1, quote="q")


def test_stage_index_orders_stages():
    assert stage_index("V3") < stage_index("R1") and stage_index("v3") == stage_index("V3")
    assert stage_index("bloom") is None and stage_index(None) is None


def test_stage_index_reads_label_wording():
    assert stage_index("V4-V5") == stage_index("V5") and stage_index("up to V8") == stage_index("V8")
    assert stage_index("8-leaf") == stage_index("V8") and stage_index("V 4") == stage_index("V4")
    assert stage_index("R1 (beginning bloom)") == stage_index("R1")
    assert stage_index("30 inches") is None and stage_index("very early") is None


def test_filters_crop_species_timing_and_stage():
    records = [_r("1", ["14"], 4), _r("2", ["14"], crop="corn"), _r("3", ["15"], timing="preemergence"),
               _r("4", ["10"], stage_min="VE", stage_max="V2"), _r("5", ["9"], species="velvetleaf")]
    options = find_label_options(records, "soybean", "waterhemp", 2.0, "V3")
    assert [o.record.reg_no for o in options] == ["1"]


def test_stage_filter_checks_each_bound_the_label_gives():
    records = [_r("max", ["14"], stage_max="V8"), _r("min", ["10"], stage_min="V3"), _r("none", ["9"])]

    def reg_nos(stage):
        return sorted(o.record.reg_no for o in find_label_options(records, "soybean", "waterhemp", None, stage))

    assert reg_nos("V10") == ["min", "none"] and reg_nos("V2") == ["max", "none"]
    assert reg_nos("V6") == ["max", "min", "none"] and reg_nos(None) == ["max", "min", "none"]


def test_over_size_kept_last_and_groups_diversified():
    records = [_r("a", ["14"], 2), _r("b", ["14"], 6), _r("c", ["10"], 5), _r("b", ["14"], 3)]
    options = find_label_options(records, "soybean", "waterhemp", 4.0, None)
    assert [o.record.reg_no for o in options] == ["b", "c", "a"]
    assert options[-1].over_size_limit and not options[0].over_size_limit
    assert options[0].record.max_weed_height_in == 6


def test_load_records_missing_file(tmp_path):
    assert load_records(tmp_path / "none.json") == []
