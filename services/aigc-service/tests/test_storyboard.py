from __future__ import annotations

import numpy as np
from PIL import Image

from app.evaluation.storyboard import evaluate_storyboard


def _frames(count: int = 16):
    output = []
    masks = []
    for index in range(count):
        image = np.full((64, 48, 3), 220, dtype=np.uint8)
        left = 16 + index // 4
        image[18:46, left : left + 12] = (30, 80, 120)
        output.append(Image.fromarray(image, mode="RGB"))
        mask = np.zeros((64, 48), dtype=bool)
        mask[18:46, left : left + 12] = True
        masks.append(mask)
    return output, masks


def test_storyboard_returns_event_rows_and_evidence(tmp_path):
    frames, masks = _frames()
    events = [
        {"id": "event_01", "start_s": 0.0, "end_s": 0.5, "camera_motion": "macro_push_in"},
        {"id": "event_02", "start_s": 0.5, "end_s": 1.0, "camera_motion": "orbit"},
    ]
    result = evaluate_storyboard(frames, masks, 16.0, events, tmp_path)
    assert result["status"] == "succeeded"
    assert len(result["events"]) == 2
    assert result["events"][0]["evidence_frames"]
    assert result["weights"]["event_completion"] == 0.45


def test_storyboard_empty_input_is_unavailable(tmp_path):
    result = evaluate_storyboard([], [], 24.0, [], tmp_path)
    assert result["status"] == "unavailable"
