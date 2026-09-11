from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .backends import create_backend
from .config import settings
from .jobs import IdempotencyConflict, JobStore
from .schemas import (
    DirectorIntentRequest, ImageGenerationRequest, JobRecord, JobSubmission,
    MaterialAnalyzeRequest, ProductParseRequest, ProductPolishRequest,
    ReferenceAnalyzeRequest, ScriptStitchRequest, SyncResponse, VideoGenerationRequest,
    LocalVideoGenerationRequest,
)

backend = create_backend(settings)
jobs = JobStore(backend, settings)


async def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    if settings.api_token and authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")


app = FastAPI(
    title="MetaCut Local AIGC Service",
    version="0.1.0",
    description="Local-first multimodal understanding and generation gateway.",
    dependencies=[Depends(authorize)],
)


@app.exception_handler(IdempotencyConflict)
async def idempotency_conflict(_: Request, exc: IdempotencyConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/healthz", tags=["system"])
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "service": settings.service_name, "backend": backend.name}


@app.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    return await healthz()


@app.get("/v1/capabilities", tags=["system"])
async def capabilities() -> dict[str, Any]:
    return backend.capabilities()


def _sync(operation: str, model: Any) -> SyncResponse:
    result, run = backend.invoke(operation, model.model_dump(mode="json"))
    return SyncResponse(result=result, run=run)


@app.post("/v1/reference/analyze", response_model=SyncResponse, tags=["understanding"])
async def reference_analyze(request: ReferenceAnalyzeRequest) -> SyncResponse:
    return _sync("reference.analyze", request)


@app.post("/v1/product/parse", response_model=SyncResponse, tags=["understanding"])
async def product_parse(request: ProductParseRequest) -> SyncResponse:
    return _sync("product.parse", request)


@app.post("/v1/material/analyze", response_model=SyncResponse, tags=["understanding"])
async def material_analyze(request: MaterialAnalyzeRequest) -> SyncResponse:
    return _sync("material.analyze", request)


@app.post("/v1/script/stitch", response_model=SyncResponse, tags=["reasoning"])
async def script_stitch(request: ScriptStitchRequest) -> SyncResponse:
    return _sync("script.stitch", request)


@app.post("/v1/product/polish", response_model=SyncResponse, tags=["reasoning"])
async def product_polish(request: ProductPolishRequest) -> SyncResponse:
    return _sync("product.polish", request)


@app.post("/v1/director/intent", response_model=SyncResponse, tags=["reasoning"])
async def director_intent(request: DirectorIntentRequest) -> SyncResponse:
    return _sync("director.intent", request)


async def _submit(kind: str, payload: dict[str, Any], key: str | None) -> JobSubmission:
    job, reused = await jobs.submit(kind, payload, key)
    return JobSubmission(
        id=job.id, status=job.status, status_url=f"/v1/jobs/{job.id}",
        reused=reused, run=job.run,
    )


@app.post("/v1/image/generations", response_model=JobSubmission, status_code=202, tags=["generation"])
async def image_generation(
    request: ImageGenerationRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> JobSubmission:
    return await _submit("image_generation", request.model_dump(mode="json"), idempotency_key)


@app.post("/v1/keyframes", response_model=JobSubmission, status_code=202, tags=["generation"])
async def keyframes(
    request: ImageGenerationRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> JobSubmission:
    """Compatibility-stable local keyframe contract for SDXL/IP-Adapter and FLUX.2."""
    return await _submit("image_generation", request.model_dump(mode="json"), idempotency_key)


@app.post("/v1/video/generations", response_model=JobSubmission, status_code=202, tags=["generation"])
async def video_generation(
    request: VideoGenerationRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> JobSubmission:
    return await _submit("video_generation", request.model_dump(mode="json"), idempotency_key)


@app.post("/v1/video", response_model=JobSubmission, status_code=202, tags=["generation"])
async def local_video_generation(
    request: LocalVideoGenerationRequest,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> JobSubmission:
    payload = request.model_dump(mode="json")
    payload["image_url"] = payload["input_image"]
    payload["duration_seconds"] = payload["num_frames"] / payload["fps"]
    return await _submit("video_generation", payload, idempotency_key)


@app.get("/v1/jobs/{job_id}", response_model=JobRecord, tags=["generation"])
async def get_job(job_id: str) -> JobRecord:
    job = await jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
