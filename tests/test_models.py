from scout.models import FieldReport, Instance, StopResult


def test_instance_defaults():
    inst = Instance(id="s0-i1", source="concept", prompt="plant", box=(0, 0, 10, 10), area_px=100)
    assert inst.role == "unknown" and inst.check_this is False


def test_stop_result_round_trip():
    stop = StopResult(stop_index=2, image_hash="abc", species={"waterhemp": 3}, species_cover_pct={"waterhemp": 1.5})
    again = StopResult.model_validate_json(stop.model_dump_json())
    assert again == stop and again.status == "ok"


def test_field_report_minimal():
    report = FieldReport(walk_id="w", crop="soybean", stops=[], species_summary=[], decision="no_action",
                         rule_trace=[], explanation="", options=[])
    assert report.treat_mode is None and report.urgency is None


def test_weed_areas_and_crop_seen_fields():
    area = Instance(id="s0-i9", source="patch", prompt="horsetail", box=(0, 0, 10, 10), area_px=40, plant_count=6)
    assert area.plant_count == 6
    assert Instance(id="s0-i1", source="prompt", prompt="w", box=(0, 0, 2, 2), area_px=4).plant_count == 1
    assert StopResult(stop_index=0, image_hash="a").crop_seen is None
    assert StopResult(stop_index=0, image_hash="a", crop_seen="corn").crop_seen == "corn"
    assert "crop_mismatch" in FieldReport.model_fields and FieldReport.model_fields["crop_mismatch"].default is None
