from app.backends.local import BackendUnavailable, LocalBackend
from app.config import Settings


def backend(tmp_path, *, allow_explicit_fallback=False):
    return LocalBackend(Settings(
        backend="local", output_root=tmp_path, asset_roots=(tmp_path,),
        allow_explicit_fallback=allow_explicit_fallback,
    ))


def test_flux_failure_retries_stable_sdxl_and_marks_provenance(tmp_path, monkeypatch):
    worker = backend(tmp_path, allow_explicit_fallback=True)
    seen = []

    def generate(payload, _run):
        seen.append(payload["model"])
        if payload["model"] == "flux2_klein":
            raise BackendUnavailable("Flux2KleinPipeline missing")
        return {"model_id": "stabilityai/stable-diffusion-xl-base-1.0", "selected_image": "/uploads/a.png"}

    monkeypatch.setattr(worker, "_generate_image", generate)
    result = worker.generate("image_generation", {"model": "flux2_klein", "seed": 7}, worker.generation_run("image_generation", {"model": "flux2_klein", "seed": 7}))

    assert seen == ["flux2_klein", "sdxl_ip_adapter"]
    assert result["requested_model"] == "flux2_klein"
    assert "FLUX.2-klein unavailable" in result["fallback_reason"]


def test_wan_failure_retries_ltx_and_marks_provenance(tmp_path, monkeypatch):
    worker = backend(tmp_path, allow_explicit_fallback=True)
    seen = []

    def generate(payload, _run):
        seen.append(payload["model"])
        if payload["model"] == "wan22_ti2v_5b":
            raise BackendUnavailable("CUDA out of memory")
        return {"model_id": "Lightricks/LTX-Video:ltxv-2b-0.9.8-distilled", "video_url": "/uploads/a.mp4"}

    monkeypatch.setattr(worker, "_generate_video", generate)
    result = worker.generate("video_generation", {"model": "wan22_ti2v_5b", "seed": 7}, worker.generation_run("video_generation", {"model": "wan22_ti2v_5b", "seed": 7}))

    assert seen == ["wan22_ti2v_5b", "ltxv_2b"]
    assert result["requested_model"] == "wan22_ti2v_5b"
    assert "Wan2.2 TI2V unavailable" in result["fallback_reason"]


def test_model_failure_is_not_silently_replaced(tmp_path, monkeypatch):
    worker = backend(tmp_path)

    def generate(_payload, _run):
        raise BackendUnavailable("requested model unavailable")

    monkeypatch.setattr(worker, "_generate_video", generate)
    try:
        worker.generate("video_generation", {"model": "wan22_ti2v_5b", "seed": 7}, worker.generation_run("video_generation", {"model": "wan22_ti2v_5b", "seed": 7}))
    except BackendUnavailable as error:
        assert "requested model unavailable" in str(error)
    else:
        raise AssertionError("requested model was silently replaced")


def test_product_polish_is_a_local_vlm_operation(tmp_path, monkeypatch):
    worker = backend(tmp_path)
    monkeypatch.setattr(worker, "_qwen_json", lambda _prompt, _images, _schema: {
        "schema_version": "product/v1", "visual_description": "polished factual description",
    })
    result, run = worker.invoke("product.polish", {
        "product": {"schema_version": "product/v1", "visual_description": "old"},
        "dirty_fields": ["visual_description"],
    })
    assert result["visual_description"] == "polished factual description"
    assert result["polish_source"] == "qwen3_vl"
    assert run.model_id
