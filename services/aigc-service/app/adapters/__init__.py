"""Stable role-level interfaces used by the v0.1 generation/evaluation stack.

Concrete deployments may live in isolated runtimes (for example the official
LTX worker), so these protocols intentionally describe the boundary rather
than importing any heavyweight ML framework.
"""

from .interfaces import (
    ImageGenerator,
    MetricEvaluator,
    ProductSegmenter,
    StoryboardEvaluator,
    VideoGenerator,
    VideoTracker,
    VLMBackend,
)

__all__ = [
    "VLMBackend", "ImageGenerator", "VideoGenerator", "ProductSegmenter",
    "VideoTracker", "MetricEvaluator", "StoryboardEvaluator",
]
