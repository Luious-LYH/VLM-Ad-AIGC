from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import settings
from .security import UnsafeAssetUri, validate_asset_uri


def _asset(value: str) -> str:
    try:
        return validate_asset_uri(value, settings)
    except UnsafeAssetUri as exc:
        raise ValueError(str(exc)) from exc


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContractOutput(BaseModel):
    """Validated VLM output while preserving contract extensions for the UI."""

    model_config = ConfigDict(extra="allow")


class ProductV1Output(ContractOutput):
    schema_version: Literal["product/v1"]
    product_name: str | None = None
    category: str | None = None
    visual_description: str | None = None
    usage_method: str | None = None
    core_selling_points: list[str] = Field(default_factory=list)
    target_audience: str | None = None
    usage_scene: str | None = None
    pain_point_gain_point: str | None = None
    ingredients_material: str | None = None
    spec_size: str | None = None


class VideoTimelineEventOutput(ContractOutput):
    start: float
    end: float
    event_name: str
    description: str | None = None


class VideoNarrativeOutput(ContractOutput):
    primary_type: str | None = None
    timeline_events: list[VideoTimelineEventOutput] = Field(default_factory=list)


class VideoAnalysisV1Output(ContractOutput):
    schema_version: Literal["video_analysis/v1"]
    meta_info: dict[str, Any] = Field(default_factory=dict)
    narrative_structure: VideoNarrativeOutput = Field(default_factory=VideoNarrativeOutput)
    rhythm_and_density: dict[str, Any] = Field(default_factory=dict)
    camera_and_composition: dict[str, Any] = Field(default_factory=dict)
    on_screen_texts: list[dict[str, Any]] = Field(default_factory=list)
    audio_and_beats: dict[str, Any] = Field(default_factory=dict)
    visual_and_color: dict[str, Any] = Field(default_factory=dict)


class RunMetadata(StrictModel):
    run_id: str
    backend: str
    provider: str = "local"
    model_id: str
    model_revision: str
    seed: int | None = None
    latency_ms: int
    deterministic: bool


class SyncResponse(StrictModel):
    success: Literal[True] = True
    result: dict[str, Any]
    run: RunMetadata


class ReferenceAnalyzeRequest(StrictModel):
    video_url: str
    frame_paths: list[str] = Field(default_factory=list, max_length=12)
    fps: float = Field(default=1.0, gt=0, le=4)
    max_frames: int = Field(default=96, ge=1, le=256)
    prompt: str | None = Field(default=None, max_length=4000)

    _safe_video = field_validator("video_url")(_asset)

    @field_validator("frame_paths")
    @classmethod
    def safe_frames(cls, values: list[str]) -> list[str]:
        return [_asset(value) for value in values]


class ProductParseRequest(StrictModel):
    product_description: str = Field(default="", max_length=8000)
    product_image_urls: list[str] = Field(default_factory=list, max_length=8)
    product_video_urls: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("product_image_urls", "product_video_urls")
    @classmethod
    def safe_assets(cls, values: list[str]) -> list[str]:
        return [_asset(value) for value in values]

    @model_validator(mode="after")
    def at_least_one_input(self) -> "ProductParseRequest":
        if not self.product_description.strip() and not self.product_image_urls and not self.product_video_urls:
            raise ValueError("provide text, an image, or a video")
        return self


class MaterialAnalyzeRequest(StrictModel):
    video_url: str
    product: dict[str, Any] | None = None
    max_highlights: int = Field(default=5, ge=1, le=20)

    _safe_video = field_validator("video_url")(_asset)


class ScriptStitchRequest(StrictModel):
    video_analysis: dict[str, Any]
    product: dict[str, Any]
    gap_plan: dict[str, Any] = Field(default_factory=lambda: {"gaps": [], "resolutions": []})
    stage_overrides: list[dict[str, Any]] = Field(default_factory=list)
    version_type: Literal["high_click", "high_conversion", "high_pace", "high_quality"] | None = None


POLISH_FIELDS = {
    "product_name", "category", "visual_description", "usage_method",
    "core_selling_points", "target_audience", "usage_scene", "pain_point_gain_point",
}


class ProductPolishRequest(StrictModel):
    product: dict[str, Any]
    dirty_fields: list[str] = Field(min_length=1, max_length=8)

    @field_validator("dirty_fields")
    @classmethod
    def valid_fields(cls, values: list[str]) -> list[str]:
        invalid = sorted(set(values) - POLISH_FIELDS)
        if invalid:
            raise ValueError(f"unsupported polish fields: {', '.join(invalid)}")
        return list(dict.fromkeys(values))


class DirectorIntentRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    manifest: dict[str, Any] | None = None
    context: dict[str, Any] | None = None


class ImageGenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    reference_images: list[str] = Field(default_factory=list, max_length=8)
    width: int = Field(default=1024, ge=256, le=2048, multiple_of=8)
    height: int = Field(default=1024, ge=256, le=2048, multiple_of=8)
    num_candidates: int = Field(default=4, ge=1, le=8)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    model: str | None = Field(default=None, max_length=200)
    conditioning: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reference_images")
    @classmethod
    def safe_images(cls, values: list[str]) -> list[str]:
        return [_asset(value) for value in values]


class VideoGenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=8000)
    image_url: str | None = None
    negative_prompt: str | None = Field(default=None, max_length=4000)
    duration_seconds: float = Field(default=4.0, gt=0, le=10)
    fps: int = Field(default=24, ge=8, le=30)
    width: int = Field(default=1280, ge=256, le=1920, multiple_of=8)
    height: int = Field(default=720, ge=256, le=1080, multiple_of=8)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    model: str | None = Field(default=None, max_length=200)

    @field_validator("image_url")
    @classmethod
    def safe_optional_image(cls, value: str | None) -> str | None:
        return _asset(value) if value is not None else None


class LocalVideoGenerationRequest(StrictModel):
    """Public I2V contract used by the Next.js local provider."""

    model: Literal["ltxv_2b", "wan22_ti2v_5b", "hunyuanvideo_15_i2v"] = "ltxv_2b"
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    input_image: str
    width: int = Field(default=576, ge=256, le=1920, multiple_of=8)
    height: int = Field(default=1024, ge=256, le=1920, multiple_of=8)
    num_frames: int = Field(default=97, ge=17, le=241)
    fps: int = Field(default=24, ge=8, le=30)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    # Optional Phase 3 ranking switch.  ``temporal`` preserves the original
    # Phase 2.5 selector; ``identity_temporal`` adds DINOv2 product matching.
    candidate_ranking: Literal["temporal", "identity_temporal"] = "temporal"
    identity_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    _safe_input = field_validator("input_image")(_asset)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class JobRecord(StrictModel):
    id: str
    kind: Literal["image_generation", "video_generation"]
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    run: RunMetadata
    result: dict[str, Any] | None = None
    error: str | None = None


class JobSubmission(StrictModel):
    id: str
    status: JobStatus
    status_url: str
    reused: bool = False
    run: RunMetadata
