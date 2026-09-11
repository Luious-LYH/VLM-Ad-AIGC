"""Local, explainable evaluation helpers for the VLM-Ad-AIGC v0.1 CLI."""

from .segmentation import SegmentResult, segment_product, track_video_masks
from .storyboard import evaluate_storyboard

__all__ = [
    "SegmentResult",
    "segment_product",
    "track_video_masks",
    "evaluate_storyboard",
]
