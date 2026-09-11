"""Dependency-light protocols for the VLM-Ad-AIGC pipeline.

The protocols are deliberately small: orchestration code can depend on these
roles while each implementation chooses its own framework/runtime and returns
JSON-serializable provenance.  ``Any`` is used at the media boundary so the
API service stays importable without torch/diffusers installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence


class VLMBackend(Protocol):
    model_id: str

    def analyze_product(self, image: Any, prompt: str) -> dict[str, Any]: ...

    def polish_prompt(self, product: dict[str, Any], prompt: str) -> dict[str, Any]: ...


class ImageGenerator(Protocol):
    model_id: str

    def generate_keyframe(self, *, prompt: str, negative_prompt: str | None,
                          reference: Path, seed: int, width: int, height: int,
                          **kwargs: Any) -> dict[str, Any]: ...


class VideoGenerator(Protocol):
    model_id: str

    def generate_video(self, *, prompt: str, negative_prompt: str | None,
                       keyframe: Path, seed: int, width: int, height: int,
                       fps: int, num_frames: int, **kwargs: Any) -> dict[str, Any]: ...


class ProductSegmenter(Protocol):
    backend: str

    def segment(self, image: Any, **kwargs: Any) -> dict[str, Any]: ...


class VideoTracker(Protocol):
    def track(self, frames: Sequence[Any], initial_mask: Any, **kwargs: Any) -> dict[str, Any]: ...


class MetricEvaluator(Protocol):
    def evaluate(self, *, reference: Any, keyframe: Any, frames: Sequence[Any],
                 masks: Sequence[Any], **kwargs: Any) -> dict[str, Any]: ...


class StoryboardEvaluator(Protocol):
    def evaluate(self, *, frames: Sequence[Any], masks: Sequence[Any],
                 events: Sequence[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...
