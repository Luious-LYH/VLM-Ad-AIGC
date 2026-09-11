from __future__ import annotations

import numpy as np
from PIL import Image

from app.evaluation.segmentation import ellipse_mask, segment_product, track_video_masks


def test_alpha_mask_is_preferred_over_background_heuristic():
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    pixels = rgba.load()
    for y in range(8, 24):
        for x in range(10, 22):
            pixels[x, y] = (20, 40, 60, 255)
    result = segment_product(rgba)
    assert result.status == "succeeded"
    assert result.backend == "alpha"
    assert result.bbox == (10, 8, 22, 24)
    assert result.confidence == 1.0


def test_contrast_product_uses_real_component_not_ellipse():
    array = np.full((120, 100, 3), 245, dtype=np.uint8)
    array[30:90, 35:65] = (10, 20, 30)
    result = segment_product(Image.fromarray(array, mode="RGB"))
    assert result.status == "succeeded"
    assert result.backend == "saliency_component"
    assert result.bbox is not None
    assert result.area_fraction is not None and result.area_fraction < 0.35
    assert result.fallback_reason is not None


def test_empty_video_is_explicitly_unavailable():
    frames, masks, metadata = track_video_masks([])
    assert frames == [] and masks == []
    assert metadata["status"] == "segmentation_failed"
    assert metadata["error"] == "empty_video"


def test_legacy_ellipse_has_expected_shape():
    mask = ellipse_mask((100, 120))
    assert mask.shape == (120, 100)
    assert mask[60, 50]
