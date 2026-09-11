import pytest
from pydantic import ValidationError

from app.schemas import ProductV1Output, VideoAnalysisV1Output


def test_product_v1_output_requires_matching_schema_version():
    parsed = ProductV1Output.model_validate({"schema_version": "product/v1", "core_selling_points": ["保温"]})
    assert parsed.core_selling_points == ["保温"]
    with pytest.raises(ValidationError):
        ProductV1Output.model_validate({"schema_version": "product/v0"})


def test_video_analysis_v1_validates_typed_timeline_events():
    parsed = VideoAnalysisV1Output.model_validate({
        "schema_version": "video_analysis/v1",
        "narrative_structure": {"timeline_events": [{"start": 0, "end": 2, "event_name": "hook"}]},
    })
    assert parsed.narrative_structure.timeline_events[0].event_name == "hook"
    with pytest.raises(ValidationError):
        VideoAnalysisV1Output.model_validate({"schema_version": "video_analysis/v1", "narrative_structure": {"timeline_events": [{"start": "bad"}]}})
