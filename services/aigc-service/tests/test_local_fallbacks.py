from app.backends.local import BackendUnavailable, LocalBackend
from app.config import Settings


def backend(tmp_path):
    return LocalBackend(Settings(backend="local", output_root=tmp_path, asset_roots=(tmp_path,)))


def test_flux_failure_retries_stable_sdxl_and_marks_provenance(tmp_path, monkeypatch):
    worker = backend(tmp_path)
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
    worker = backend(tmp_path)
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
